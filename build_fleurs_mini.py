#!/usr/bin/env python3
"""Build FLEURS-mini: five-language (zh/en/pt/es/it) mini benchmark, ~10 min per
language, drawn from the LOCAL FLEURS test split (google/fleurs).

Tasks covered: Speech LangID, ASR, Translation (20 directions).

Design
------
- Pool = sentence_ids with >=1 valid recording in ALL five locales (5-way
  parallel), excluding the 10 sids used by the cascade en->zh manual
  validation (docs/cascade-validation-results.md).
- Core = greedy subset of pool (ordered by it/en min-duration ratio) with
  it_min_sum <= 480 s. Every language's manifest contains all core sids ->
  same-sentence cross-lingual comparability.
- Per-language fill: iterate core sids then extension candidates (pool sids
  first, overlap-preferred), picking per sid the recording closest to
  min(lang_median, TARGET - running_total), stop at >= 595 s, hard cap 610 s.
- Audio: every selected wav is re-encoded to 16 kHz mono PCM16 (en sources are
  float32; zh/es/pt/it prefer the existing _16k.wav variants).

Fully deterministic (no RNG). Re-running reproduces identical manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

import soundfile as sf

parser = argparse.ArgumentParser(description="Build FLEURS-mini from a local FLEURS checkout")
parser.add_argument("--src", default=None,
                    help="FLEURS data dir with <locale>/test.tsv + <locale>/audio/test/ "
                         "(default: qwen3-asr-wp0 dev layout at data/fleurs/data)")
args = parser.parse_args()

REPO = Path(__file__).resolve().parents[2]          # ~/github/qwen3-asr-wp0 (dev checkout)
SRC = REPO / "data" / "fleurs" / "data"            # default: dev layout
OUT = Path(__file__).resolve().parent               # .../data/FLEURS-mini

LOCALES = {"cmn_hans_cn": "zh", "en_us": "en", "es_419": "es", "pt_br": "pt", "it_it": "it"}
LOC_BY_LANG = {v: k for k, v in LOCALES.items()}
LANGS = ["en", "zh", "es", "pt", "it"]              # fill order (en hardest to fill)

TARGET_SECONDS = 600.0
STOP_WHEN = 595.0
HARD_CAP = 610.0
CORE_IT_MIN_CAP = 480.0
CASCADE10_SIDS = {"1669", "1673", "1699", "1771", "1832", "1877", "1914", "1915", "1949", "1980"}


def read_tsv(loc: str):
    rows, skipped = [], 0
    with open(SRC / loc / "test.tsv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 7 and row[5].isdigit() and row[0] and row[1] and row[2]:
                rows.append({"sid": row[0], "fn": row[1], "raw": row[2], "norm": row[3],
                             "ns": int(row[5]), "gender": row[6]})
            else:
                skipped += 1
    return rows, skipped


def src_audio(loc: str, fn: str):
    """Existing source wav for a TSV row (prefer _16k variant)."""
    d = SRC / loc / "audio" / "test" / "test"
    if not d.is_dir():
        d = SRC / loc / "audio" / "test"
    p16 = d / fn.replace(".wav", "_16k.wav")
    if p16.exists():
        return p16
    p = d / fn
    return p if p.exists() else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global SRC
    if args.src:
        SRC = Path(args.src).expanduser().resolve()
    if not (SRC / "cmn_hans_cn" / "test.tsv").exists():
        raise SystemExit(f"FLEURS source not found: {SRC} (pass --src /path/to/fleurs/data)")
    created = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")

    # ---------- 1. parse ----------
    recs = {}          # lang -> sid -> [row, ...]  (valid rows with existing audio)
    skipped_rows = {}
    tsv_sha = {}
    for loc, lang in LOCALES.items():
        rows, skipped = read_tsv(loc)
        skipped_rows[lang] = skipped
        tsv_sha[lang] = sha256(SRC / loc / "test.tsv")
        by_sid = {}
        for r in rows:
            ap = src_audio(loc, r["fn"])
            if ap is None:
                continue
            r["src"] = ap
            by_sid.setdefault(r["sid"], []).append(r)
        recs[lang] = by_sid
        print(f"[parse] {loc} ({lang}): {len(rows)} valid rows, {skipped} malformed skipped, "
              f"{sum(len(v) for v in by_sid.values())} with audio, {len(by_sid)} sids")

    # ---------- 2. pool & core ----------
    pool = set.intersection(*[set(recs[l]) for l in LANGS]) - CASCADE10_SIDS
    print(f"[pool] 5-way parallel sids (cascade-10 excluded): {len(pool)}")

    def dmin(lang, sid):
        return min(r["ns"] for r in recs[lang][sid]) / 16000.0

    core_order = sorted(pool, key=lambda s: (dmin("it", s) / max(dmin("en", s), 0.1), int(s)))
    core, it_sum = [], 0.0
    for sid in core_order:
        d = dmin("it", sid)
        if it_sum + d > CORE_IT_MIN_CAP:
            continue
        core.append(sid)
        it_sum += d
    core_set = set(core)
    print(f"[core] {len(core)} sids, it_min_sum={it_sum:.0f}s")

    # ---------- 3. per-language selection ----------
    lang_median = {l: sorted(r["ns"] for s in recs[l] for r in recs[l][s])[len([r for s in recs[l] for r in recs[l][s]]) // 2] / 16000.0
                   for l in LANGS}

    selected = {}      # lang -> {sid: row}
    all_selected_sids = set()
    for lang in LANGS:
        chosen = {}
        total = 0.0
        # candidate order: core sids first (numeric), then pool ext (overlap-preferred), then rest
        ext_pool = sorted(pool - core_set, key=int)
        rest = sorted(set(recs[lang]) - pool - core_set - CASCADE10_SIDS, key=int)
        overlap_first = [s for s in ext_pool if s in all_selected_sids]
        non_overlap = [s for s in ext_pool if s not in all_selected_sids]
        order = sorted(core_set, key=int) + overlap_first + non_overlap + rest
        for sid in order:
            if total >= STOP_WHEN:
                break
            want = min(lang_median[lang], TARGET_SECONDS - total)
            cands = sorted(recs[lang][sid], key=lambda r: abs(r["ns"] / 16000.0 - want))
            row = cands[0]
            d = row["ns"] / 16000.0
            if total + d > HARD_CAP:
                row = min(recs[lang][sid], key=lambda r: r["ns"])
                d = row["ns"] / 16000.0
                if total + d > HARD_CAP:
                    if sid in core_set:
                        raise RuntimeError(f"core sid {sid} cannot fit under cap for {lang}")
                    continue
            chosen[sid] = row
            total += d
        assert total >= 590.0, f"{lang}: total {total:.1f}s below window"
        selected[lang] = chosen
        all_selected_sids |= set(chosen)
        print(f"[fill] {lang}: {len(chosen)} utts ({sum(1 for s in chosen if s in core_set)} core "
              f"+ {sum(1 for s in chosen if s not in core_set)} ext), total={total:.1f}s")

    # ---------- 4. audio + manifests ----------
    for sub in ("audio", "asr", "langid", "translation"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    audio_index = {}   # (lang, sid) -> relative audio path
    asr_entries = {l: [] for l in LANGS}
    for lang in LANGS:
        loc = LOC_BY_LANG[lang]
        (OUT / "audio" / lang).mkdir(exist_ok=True)
        for sid in sorted(selected[lang], key=int):
            row = selected[lang][sid]
            stem = row["fn"].replace(".wav", "").replace("_16k", "")
            dst = OUT / "audio" / lang / f"{stem}.wav"
            data, sr = sf.read(row["src"], dtype="float32")
            assert sr == 16000 and data.ndim == 1, f"{row['src']}: sr={sr} ndim={data.ndim}"
            sf.write(dst, data, 16000, subtype="PCM_16")
            rel = f"audio/{lang}/{stem}.wav"
            audio_index[(lang, sid)] = rel
            info = sf.info(dst)
            asr_entries[lang].append({
                "utt_id": f"fleursmini_{lang}_{sid}",
                "sent_id": sid,
                "lang": lang,
                "locale": loc,
                "audio": rel,
                "duration_s": round(info.frames / info.samplerate, 3),
                "num_samples_tsv": row["ns"],
                "transcript": row["raw"],
                "transcript_norm": row["norm"],
                "gender": row["gender"],
                "in_core": sid in core_set,
                "source_tsv": f"data/fleurs/data/{loc}/test.tsv",
                "source_filename": row["fn"],
            })
        with open(OUT / "asr" / f"asr_{lang}.jsonl", "w", encoding="utf-8") as f:
            for e in asr_entries[lang]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tot = sum(e["duration_s"] for e in asr_entries[lang])
        print(f"[asr] {lang}: {len(asr_entries[lang])} utts, {tot:.1f}s")

    # LangID: union of all asr entries
    with open(OUT / "langid" / "langid.jsonl", "w", encoding="utf-8") as f:
        for lang in LANGS:
            for e in asr_entries[lang]:
                f.write(json.dumps({"utt_id": e["utt_id"], "audio": e["audio"],
                                    "duration_s": e["duration_s"], "lang": lang},
                                   ensure_ascii=False) + "\n")

    # Translation: 20 directions
    pair_counts = {}
    for src_l in LANGS:
        for tgt_l in LANGS:
            if src_l == tgt_l:
                continue
            shared = sorted(set(selected[src_l]) & set(selected[tgt_l]), key=int)
            rows = []
            for sid in shared:
                srow, trow = selected[src_l][sid], selected[tgt_l][sid]
                rows.append({
                    "pair_id": f"fleursmini_{src_l}_{tgt_l}_{sid}",
                    "direction": f"{src_l}_{tgt_l}",
                    "sent_id": sid,
                    "source_lang": src_l,
                    "target_lang": tgt_l,
                    "source_audio": audio_index[(src_l, sid)],
                    "source_transcript": srow["raw"],
                    "source_transcript_norm": srow["norm"],
                    "target_reference": trow["raw"],
                    "target_reference_norm": trow["norm"],
                    "reference_quality": "human_verified",
                })
            with open(OUT / "translation" / f"{src_l}_{tgt_l}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            pair_counts[f"{src_l}->{tgt_l}"] = len(rows)

    # ---------- 5. metadata ----------
    def lang_stats(lang):
        es = asr_entries[lang]
        durs = sorted(e["duration_s"] for e in es)
        g = Counter(e["gender"] for e in es)
        return {
            "n_utterances": len(es),
            "n_sids": len({e['sent_id'] for e in es}),
            "n_core_sids": sum(1 for e in es if e["in_core"]),
            "total_duration_s": round(sum(durs), 1),
            "duration_min_s": round(durs[0], 1),
            "duration_median_s": round(durs[len(durs) // 2], 1),
            "duration_max_s": round(durs[-1], 1),
            "gender": dict(g),
        }

    try:
        git_rev = subprocess.run(["git", "-C", str(OUT), "describe", "--always", "--dirty"],
                                 capture_output=True, text=True).stdout.strip()
        if not git_rev:
            git_rev = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        git_rev = "unknown"

    metadata = {
        "name": "FLEURS-mini",
        "version": "1.0.0",
        "created_at": created,
        "builder": "build_fleurs_mini.py",
        "repo_git_rev": git_rev,
        "source": {
            "dataset": "google/fleurs (FLEURS, arXiv 2205.12446)",
            "split": "test",
            "locales": LOCALES,
            "source_dir": str(SRC),
            "tsv_sha256": tsv_sha,
        },
        "construction": {
            "deterministic": True,
            "seed": None,
            "target_seconds_per_lang": TARGET_SECONDS,
            "accept_window_s": [590, 610],
            "core": {
                "definition": "sentence_ids with >=1 valid recording in all 5 locales",
                "pool_size": len(pool),
                "core_size": len(core),
                "it_min_sum_cap_s": CORE_IT_MIN_CAP,
                "ordering": "it/en min-duration ratio ascending, then sid",
            },
            "fill_order": LANGS,
            "recording_choice": "per sid, recording closest to min(lang_median, 600 - running_total)",
            "excluded_sids_cascade10": sorted(CASCADE10_SIDS),
            "malformed_tsv_rows_skipped": skipped_rows,
        },
        "audio_format": "wav, 16 kHz, mono, PCM16 (re-encoded from FLEURS sources)",
        "tasks": {
            "langid": "langid/langid.jsonl (5 classes, duration-balanced)",
            "asr": "asr/asr_<lang>.jsonl x 5",
            "translation": "translation/<src>_<tgt>.jsonl x 20 directions",
        },
        "stats": {
            "per_lang": {l: lang_stats(l) for l in LANGS},
            "langid_total": sum(len(asr_entries[l]) for l in LANGS),
            "translation_pairs": pair_counts,
        },
        "contamination_notes": [
            "FLEURS test split was never used in LoRA training or pseudolabels "
            "(pseudolabels.jsonl: 0/14123 items reference test-split sids).",
            "Overlaps the GEMBA-WP0-v1 eval universe (same FLEURS test split) by construction.",
            "The 10 cascade-validation sids are excluded from all tasks.",
        ],
        "langid_gender_confound": {
            "upstream_distribution": {"zh": "70% M", "en": "36% M", "es": "67% M",
                                      "pt": "100% M (913/913 test rows)", "it": "4% M (37/862 test rows)"},
            "implication": "pt and it are nearly gender-segregated in upstream FLEURS test, so "
                           "LangID accuracy on this set can exploit a speaker-gender/pitch shortcut "
                           "for pt vs it. Report per-class accuracy and treat pt/it separation as "
                           "upper-bound; do not attribute it purely to language modeling.",
        },
        "license": "CC-BY-4.0 (upstream FLEURS)",
    }
    with open(OUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ---------- 6. README ----------
    per_lang_rows = "\n".join(
        f"| {l} | {s['n_utterances']} | {s['n_sids']} | {s['n_core_sids']} | "
        f"{s['total_duration_s']/60:.2f} min | {s['duration_min_s']}–{s['duration_max_s']}s "
        f"(中位 {s['duration_median_s']}s) | {s['gender'].get('MALE',0)}M/{s['gender'].get('FEMALE',0)}F |"
        for l, s in metadata["stats"]["per_lang"].items())
    trans_rows = "\n".join(
        f"| {d.replace('->','→')} | {n} |" for d, n in pair_counts.items())

    readme = f"""# FLEURS-mini

从本地 FLEURS（google/fleurs，test split）抽取的五语迷你基准：**中(zh)、英(en)、葡(pt)、西(es)、意(it)**，
每语约 **10 分钟** 音频，覆盖 **Speech LangID / ASR / Translation** 三种任务。
由 `build_fleurs_mini.py` 全自动确定性构建（无随机数，重跑可复现），构建时间 {created}，repo rev `{git_rev}`。

## 设计

- **平行核心**：五语 test split 共有 {len(pool)} 个五方平行 sentence_id（同一句话五种语言版本），
  排除级联验证用过的 10 个 sid 后，贪心选出 **{len(core)} 句核心集**（按 it/en 最短时长比升序，
  it 最短时长累计 ≤ {CORE_IT_MIN_CAP:.0f}s 封顶）。**核心集五语全包含** → 同一句话可跨语言直接对比。
- **各语补齐**：在核心集之上，按 en→zh→es→pt→it 顺序补足到每语 ~600s（接受区间 590–610s），
  每句选"最接近 min(该语中位时长, 剩余预算)"的那条录音；扩展句优先选其他语言已选中的 sid（最大化翻译对重叠）。
- **时长配平的动机**：五语句长差异大（en 句均 ~9.9s，it ~14.7s），严格"同句集五语全等长"不可行；
  核心集保证平行性，录音选择吸收时长差异。
- **排除项**：级联 en→zh 手工验证的 10 个 sid（{sorted(CASCADE10_SIDS, key=int)}）全部任务排除；
  TSV 畸形行（<7 列）跳过：{skipped_rows}。
- **音频统一重编码**为 16 kHz 单声道 PCM16 wav（en 源为 float32，其余优先用 _16k 源）。

## 目录结构

```
FLEURS-mini/
├── audio/{{lang}}/*.wav        # 16kHz mono PCM16，文件名=源 FLEURS 录音 stem
├── asr/asr_{{en,zh,es,pt,it}}.jsonl
├── langid/langid.jsonl       # 五分类 LangID（时长均衡）
├── translation/{{src}}_{{tgt}}.jsonl  # 20 个方向，源音频+目标语人工参考译文
├── metadata.json             # 构建参数、统计、来源 sha256、污染声明
├── checksums.sha256          # 全部音频+清单校验和
├── build_fleurs_mini.py      # 构建脚本（本文件）
├── validate.py               # 完整性校验（checksum/格式/schema/一致性）
└── README.md
```

## 任务格式

**ASR**（`asr/asr_zh.jsonl`，每语一个文件）：

```json
{{"utt_id": "fleursmini_zh_1779", "sent_id": "1779", "lang": "zh", "locale": "cmn_hans_cn",
  "audio": "audio/zh/10325559490685159122.wav", "duration_s": 15.16,
  "transcript": "特朗普与土耳其总统…", "transcript_norm": "…", "gender": "MALE",
  "in_core": true, "source_tsv": "data/fleurs/data/cmn_hans_cn/test.tsv", "source_filename": "…"}}
```

评分：`transcript`（raw）为主参考；zh 用 **CER**，en/es/pt/it 用 WER（§4.5：WER/CER 只作趋势指标，不作语义门禁）。

**Speech LangID**（`langid/langid.jsonl`）：`utt_id / audio / duration_s / lang`，五分类，
各类时长均衡（~10 min/类），可避免"时长捷径"。评分：accuracy（必要时加 per-class recall）。
⚠️ **性别混淆警告**：上游 FLEURS test 的 pt_br 全部为男声（913/913）、it_it 几乎全为女声（4% 男），
本数据继承该分布（pt 49M/0F，it 1M/44F）——LangID 模型可借音高/性别捷径区分 pt↔it，
pt/it 分类的准确率应视为上界，不能完全归因于语言建模。详见 `metadata.json` 的 `langid_gender_confound`。

**Translation**（`translation/zh_es.jsonl` = zh→es，共 20 方向）：

```json
{{"pair_id": "fleursmini_zh_es_1779", "direction": "zh_es", "sent_id": "1779",
  "source_lang": "zh", "target_lang": "es",
  "source_audio": "audio/zh/….wav", "source_transcript": "…", "source_transcript_norm": "…",
  "target_reference": "…", "target_reference_norm": "…", "reference_quality": "human_verified"}}
```

参考译文来自 FLEURS 平行文本（human_verified，非机翻）。评分建议沿用 GEMBA-WP0-v1
（`wp0.data.gemba_scorer`，裁判 deepseek-v4-flash-0731，adequacy≥7 门禁）；
若用 BLEU 辅助，中文目标必须 `sacrebleu … tokenize='zh'`。

## 统计

### 每语（ASR/LangID 视角）

| 语种 | 句数 | sid 数 | 核心 sid | 总时长 | 单句时长范围(中位) | 性别 |
| --- | --- | --- | --- | --- | --- | --- |
{per_lang_rows}

LangID 合计 {metadata['stats']['langid_total']} 条，5 类。

### 翻译方向（20 方向句对数，均 ≥ 核心集 {len(core)}）

| 方向 | 句对 | 方向 | 句对 |
| --- | --- | --- | --- |
""" + "\n".join(
        f"| {d} | {pair_counts[d]} | {d2} | {pair_counts[d2]} |"
        for d, d2 in zip(sorted(pair_counts), sorted(pair_counts, reverse=True))) + f"""

## 污染与重叠声明

1. FLEURS **test split 从未进入 LoRA 训练或伪标签**（`pseudolabels.jsonl` 14123 条中 0 条引用 test sid）；
   G8 held-out 是 train wav 名单，与本数据无交集。
2. 与 GEMBA-WP0-v1 评测同源（都是 test split），有意重叠——mini 可视作其快速子集。
3. 级联验证 10 sid 已排除，与 `docs/cascade-validation-results.md` 的实测样本无重叠。
4. **en 是新增正式覆盖**：此前 en 仅有级联验证 10 句，未进 GEMBA 6 方向；本基准 en 参与全部三任务。

## 复现与校验

```bash
# 校验下载内容的完整性（checksum / 音频格式 / schema / 时长窗口 / 平行性）
python validate.py

# （可选）从本地 FLEURS 全量 test split 重新构建，产物确定性可复现
python build_fleurs_mini.py --src /path/to/fleurs/data   # 需要五语 test.tsv + audio/test/
```

## 来源与许可

- FLEURS: Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations
  of Speech*, arXiv:2205.12446，https://huggingface.co/datasets/google/fleurs （test split）。
  重建需要本地 FLEURS 副本（`--src` 指向含五语 `test.tsv` + `audio/test/` 的目录）；TSV sha256 见 `metadata.json`。
- 许可：CC-BY-4.0（随上游 FLEURS）。
"""
    with open(OUT / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # ---------- 7. checksums ----------
    lines = []
    for sub in ("audio", "asr", "langid", "translation"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                lines.append(f"{sha256(p)}  {p.relative_to(OUT)}")
    for name in ("README.md", "metadata.json", "build_fleurs_mini.py", "validate.py"):
        p = OUT / name
        if p.exists():
            lines.append(f"{sha256(p)}  {name}")
    with open(OUT / "checksums.sha256", "w") as f:
        f.write("\n".join(sorted(lines)) + "\n")

    n_audio = sum(len(asr_entries[l]) for l in LANGS)
    size = sum(p.stat().st_size for p in (OUT / "audio").rglob("*.wav"))
    print(f"\n[done] {n_audio} wavs ({size/2**20:.0f} MiB), 20 translation files, "
          f"{metadata['stats']['langid_total']} langid items -> {OUT}")


if __name__ == "__main__":
    main()
