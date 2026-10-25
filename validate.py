#!/usr/bin/env python3
"""FLEURS-mini integrity validator.

Checks (all must pass):
 1. checksums.sha256 matches every listed file (and no manifest/audio file is missing from it)
 2. every JSONL parses and has the required schema fields
 3. every audio file is 16 kHz mono PCM16; |declared - actual duration| <= 0.1s
 4. per-language ASR totals within [590, 610]s; LangID per-class totals too
 5. translation: 20 direction files, non-empty references, both-side sids present
    in the corresponding ASR manifests, unique pair_ids
 6. core sids present in all 5 languages; cascade-10 sids absent everywhere
 7. no duplicate utt_ids; no orphan audio files (unreferenced wavs)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

import soundfile as sf

HERE = Path(__file__).resolve().parent
LANGS = ["en", "zh", "es", "pt", "it"]
WINDOW = (590.0, 610.0)
CASCADE10_SIDS = {"1669", "1673", "1699", "1771", "1832", "1877", "1914", "1915", "1949", "1980"}

ASR_FIELDS = {"utt_id", "sent_id", "lang", "locale", "audio", "duration_s", "transcript",
              "transcript_norm", "gender", "in_core", "source_tsv", "source_filename"}
LANGID_FIELDS = {"utt_id", "audio", "duration_s", "lang"}
TRANS_FIELDS = {"pair_id", "direction", "sent_id", "source_lang", "target_lang",
                "source_audio", "source_transcript", "source_transcript_norm",
                "target_reference", "target_reference_norm", "reference_quality"}

errors: list[str] = []
warnings: list[str] = []


def err(msg: str):
    errors.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                err(f"{path.name}:{i}: JSON parse error: {e}")
    return out


def main():
    # ---- 1. checksums ----
    ck_path = HERE / "checksums.sha256"
    if not ck_path.exists():
        err("checksums.sha256 missing")
    else:
        listed, seen = {}, set()
        with open(ck_path, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=" ", quoting=csv.QUOTE_NONE):
                row = [c for c in row if c]
                if len(row) != 2:
                    continue
                listed[row[1]] = row[0]
        for rel, digest in listed.items():
            p = HERE / rel
            if not p.exists():
                err(f"checksummed file missing: {rel}")
                continue
            seen.add(rel)
            if sha256(p) != digest:
                err(f"checksum mismatch: {rel}")
        # manifests + audio must all be covered
        for sub, pat in (("asr", "asr_*.jsonl"), ("langid", "langid.jsonl"),
                         ("translation", "*_*.jsonl"), ("audio", "**/*.wav")):
            for p in (HERE / sub).glob(pat):
                rel = str(p.relative_to(HERE))
                if rel not in seen:
                    err(f"not in checksums.sha256: {rel}")

    # ---- 2/3/4. ASR manifests ----
    asr = {}
    audio_actual = {}
    for lang in LANGS:
        path = HERE / "asr" / f"asr_{lang}.jsonl"
        if not path.exists():
            err(f"missing {path.name}")
            continue
        entries = load_jsonl(path)
        asr[lang] = entries
        for e in entries:
            missing = ASR_FIELDS - set(e)
            if missing:
                err(f"asr_{lang}: {e.get('utt_id')}: missing fields {sorted(missing)}")
                continue
            if e["lang"] != lang:
                err(f"asr_{lang}: {e['utt_id']}: lang={e['lang']}")
            if not e["transcript"].strip():
                err(f"asr_{lang}: {e['utt_id']}: empty transcript")
            if e["sent_id"] in CASCADE10_SIDS:
                err(f"asr_{lang}: cascade-10 sid leaked: {e['sent_id']}")
            ap = HERE / e["audio"]
            if not ap.exists():
                err(f"asr_{lang}: {e['utt_id']}: audio missing: {e['audio']}")
                continue
            info = sf.info(ap)
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
                err(f"asr_{lang}: {e['utt_id']}: audio {info.samplerate}Hz {info.channels}ch {info.subtype}")
                continue
            actual = info.frames / info.samplerate
            audio_actual[e["audio"]] = actual
            if abs(actual - e["duration_s"]) > 0.1:
                err(f"asr_{lang}: {e['utt_id']}: duration {e['duration_s']} vs actual {actual:.3f}")
        tot = sum(e["duration_s"] for e in entries)
        if not (WINDOW[0] <= tot <= WINDOW[1]):
            err(f"asr_{lang}: total {tot:.1f}s outside window {WINDOW}")
        uids = [e["utt_id"] for e in entries]
        dup = [u for u, c in Counter(uids).items() if c > 1]
        if dup:
            err(f"asr_{lang}: duplicate utt_ids {dup[:5]}")
        print(f"asr_{lang}: {len(entries)} utts, {tot/60:.2f} min")

    # ---- langid ----
    lpath = HERE / "langid" / "langid.jsonl"
    if not lpath.exists():
        err("missing langid.jsonl")
    else:
        lentries = load_jsonl(lpath)
        per_lang_tot = Counter()
        asr_uids = {e["utt_id"] for l in LANGS for e in asr.get(l, [])}
        for e in lentries:
            if LANGID_FIELDS - set(e):
                err(f"langid: {e.get('utt_id')}: missing fields")
                continue
            if e["lang"] not in LANGS:
                err(f"langid: {e['utt_id']}: bad lang {e['lang']}")
            if e["utt_id"] not in asr_uids:
                err(f"langid: {e['utt_id']}: not present in any asr manifest")
            if e["audio"] not in audio_actual:
                err(f"langid: {e['utt_id']}: audio not in asr set: {e['audio']}")
                continue
            if abs(audio_actual[e["audio"]] - e["duration_s"]) > 0.1:
                err(f"langid: {e['utt_id']}: duration mismatch")
            per_lang_tot[e["lang"]] += e["duration_s"]
        for lang in LANGS:
            t = per_lang_tot.get(lang, 0.0)
            if not (WINDOW[0] <= t <= WINDOW[1]):
                err(f"langid: class {lang} total {t:.1f}s outside window")
        print(f"langid: {len(lentries)} items, per-class minutes:",
              {k: round(v / 60, 2) for k, v in sorted(per_lang_tot.items())})

    # ---- 5. translation ----
    pair_files = 0
    sids_by_lang = {l: {e["sent_id"] for e in asr.get(l, [])} for l in LANGS}
    audio_by_lang = {l: {e["audio"] for e in asr.get(l, [])} for l in LANGS}
    all_pair_ids = []
    for src in LANGS:
        for tgt in LANGS:
            if src == tgt:
                continue
            path = HERE / "translation" / f"{src}_{tgt}.jsonl"
            if not path.exists():
                err(f"missing translation file: {path.name}")
                continue
            pair_files += 1
            rows = load_jsonl(path)
            n_ok = 0
            for r in rows:
                if TRANS_FIELDS - set(r):
                    err(f"{path.name}: {r.get('pair_id')}: missing fields")
                    continue
                if r["direction"] != f"{src}_{tgt}" or r["source_lang"] != src or r["target_lang"] != tgt:
                    err(f"{path.name}: {r['pair_id']}: direction mismatch")
                if not r["target_reference"].strip():
                    err(f"{path.name}: {r['pair_id']}: empty reference")
                if r["reference_quality"] != "human_verified":
                    err(f"{path.name}: {r['pair_id']}: quality={r['reference_quality']}")
                if r["sent_id"] not in sids_by_lang[src] or r["sent_id"] not in sids_by_lang[tgt]:
                    err(f"{path.name}: {r['pair_id']}: sid not in both ASR manifests")
                if r["source_audio"] not in audio_by_lang[src]:
                    err(f"{path.name}: {r['pair_id']}: source audio not in asr_{src}")
                if r["sent_id"] in CASCADE10_SIDS:
                    err(f"{path.name}: cascade-10 sid leaked: {r['sent_id']}")
                n_ok += 1
            all_pair_ids.extend(r.get("pair_id", "") for r in rows)
            if n_ok < 30:
                err(f"{path.name}: only {n_ok} pairs")
    if pair_files != 20:
        err(f"translation: {pair_files}/20 direction files")
    dup = [p for p, c in Counter(all_pair_ids).items() if c > 1 and p]
    if dup:
        err(f"translation: duplicate pair_ids {dup[:5]}")
    print(f"translation: {pair_files} files, {len(all_pair_ids)} pairs total")

    # ---- 6. core sids in all langs ----
    meta = json.load(open(HERE / "metadata.json", encoding="utf-8"))
    core_size = meta["construction"]["core"]["core_size"]
    core_sids = None
    for lang in LANGS:
        cs = {e["sent_id"] for e in asr.get(lang, []) if e.get("in_core")}
        if core_sids is None:
            core_sids = cs
        elif cs != core_sids:
            err(f"core sid set differs for {lang}: "
                f"+{sorted(cs - core_sids)[:3]} -{sorted(core_sids - cs)[:3]}")
    if core_sids is not None and len(core_sids) != core_size:
        err(f"core size {len(core_sids)} != metadata {core_size}")
    print(f"core: {len(core_sids) if core_sids else 0} sids shared by all 5 langs")

    # ---- 7. orphan audio ----
    referenced = {e["audio"] for l in LANGS for e in asr.get(l, [])}
    orphans = [str(p.relative_to(HERE)) for p in (HERE / "audio").rglob("*.wav")
               if str(p.relative_to(HERE)) not in referenced]
    if orphans:
        err(f"orphan audio files: {orphans[:5]}")

    # ---- verdict ----
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"\nFAIL: {len(errors)} error(s)")
        for e in errors[:30]:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: all checks green")


if __name__ == "__main__":
    main()
