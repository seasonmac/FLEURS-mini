# FLEURS-mini

从本地 FLEURS（google/fleurs，test split）抽取的五语迷你基准：**中(zh)、英(en)、葡(pt)、西(es)、意(it)**，
每语约 **10 分钟** 音频，覆盖 **Speech LangID / ASR / Translation** 三种任务。
由 `build_fleurs_mini.py` 全自动确定性构建（无随机数，重跑可复现），构建时间 2026-10-25T17:26:49+08:00，repo rev `6eeb46b-dirty`。

## 设计

- **平行核心**：五语 test split 共有 330 个五方平行 sentence_id（同一句话五种语言版本），
  排除级联验证用过的 10 个 sid 后，贪心选出 **43 句核心集**（按 it/en 最短时长比升序，
  it 最短时长累计 ≤ 480s 封顶）。**核心集五语全包含** → 同一句话可跨语言直接对比。
- **各语补齐**：在核心集之上，按 en→zh→es→pt→it 顺序补足到每语 ~600s（接受区间 590–610s），
  每句选"最接近 min(该语中位时长, 剩余预算)"的那条录音；扩展句优先选其他语言已选中的 sid（最大化翻译对重叠）。
- **时长配平的动机**：五语句长差异大（en 句均 ~9.9s，it ~14.7s），严格"同句集五语全等长"不可行；
  核心集保证平行性，录音选择吸收时长差异。
- **排除项**：级联 en→zh 手工验证的 10 个 sid（['1669', '1673', '1699', '1771', '1832', '1877', '1914', '1915', '1949', '1980']）全部任务排除；
  TSV 畸形行（<7 列）跳过：{'zh': 0, 'en': 2, 'es': 0, 'pt': 6, 'it': 3}。
- **音频统一重编码**为 16 kHz 单声道 PCM16 wav（en 源为 float32，其余优先用 _16k 源）。

## 目录结构

```
FLEURS-mini/
├── audio/{lang}/*.wav        # 16kHz mono PCM16，文件名=源 FLEURS 录音 stem
├── asr/asr_{en,zh,es,pt,it}.jsonl
├── langid/langid.jsonl       # 五分类 LangID（时长均衡）
├── translation/{src}_{tgt}.jsonl  # 20 个方向，源音频+目标语人工参考译文
├── metadata.json             # 构建参数、统计、来源 sha256、污染声明
├── checksums.sha256          # 全部音频+清单校验和
├── build_fleurs_mini.py      # 构建脚本（本文件）
├── validate.py               # 完整性校验（checksum/格式/schema/一致性）
└── README.md
```

## 任务格式

**ASR**（`asr/asr_zh.jsonl`，每语一个文件）：

```json
{"utt_id": "fleursmini_zh_1779", "sent_id": "1779", "lang": "zh", "locale": "cmn_hans_cn",
  "audio": "audio/zh/10325559490685159122.wav", "duration_s": 15.16,
  "transcript": "特朗普与土耳其总统…", "transcript_norm": "…", "gender": "MALE",
  "in_core": true, "source_tsv": "data/fleurs/data/cmn_hans_cn/test.tsv", "source_filename": "…"}
```

评分：`transcript`（raw）为主参考；zh 用 **CER**，en/es/pt/it 用 WER（§4.5：WER/CER 只作趋势指标，不作语义门禁）。

**Speech LangID**（`langid/langid.jsonl`）：`utt_id / audio / duration_s / lang`，五分类，
各类时长均衡（~10 min/类），可避免"时长捷径"。评分：accuracy（必要时加 per-class recall）。
⚠️ **性别混淆警告**：上游 FLEURS test 的 pt_br 全部为男声（913/913）、it_it 几乎全为女声（4% 男），
本数据继承该分布（pt 49M/0F，it 1M/44F）——LangID 模型可借音高/性别捷径区分 pt↔it，
pt/it 分类的准确率应视为上界，不能完全归因于语言建模。详见 `metadata.json` 的 `langid_gender_confound`。

**Translation**（`translation/zh_es.jsonl` = zh→es，共 20 方向）：

```json
{"pair_id": "fleursmini_zh_es_1779", "direction": "zh_es", "sent_id": "1779",
  "source_lang": "zh", "target_lang": "es",
  "source_audio": "audio/zh/….wav", "source_transcript": "…", "source_transcript_norm": "…",
  "target_reference": "…", "target_reference_norm": "…", "reference_quality": "human_verified"}
```

参考译文来自 FLEURS 平行文本（human_verified，非机翻）。评分建议沿用 GEMBA-WP0-v1
（`wp0.data.gemba_scorer`，裁判 deepseek-v4-flash-0731，adequacy≥7 门禁）；
若用 BLEU 辅助，中文目标必须 `sacrebleu … tokenize='zh'`。

## 统计

### 每语（ASR/LangID 视角）

| 语种 | 句数 | sid 数 | 核心 sid | 总时长 | 单句时长范围(中位) | 性别 |
| --- | --- | --- | --- | --- | --- | --- |
| en | 53 | 53 | 43 | 10.04 min | 6.9–20.7s (中位 10.4s) | 36M/17F |
| zh | 54 | 54 | 43 | 10.03 min | 6.6–20.7s (中位 10.9s) | 39M/15F |
| es | 50 | 50 | 43 | 9.96 min | 7.6–21.5s (中位 11.5s) | 35M/15F |
| pt | 49 | 49 | 43 | 10.06 min | 8.5–20.1s (中位 11.6s) | 49M/0F |
| it | 45 | 45 | 43 | 10.01 min | 8.2–19.7s (中位 13.3s) | 1M/44F |

LangID 合计 251 条，5 类。

### 翻译方向（20 方向句对数，均 ≥ 核心集 43）

| 方向 | 句对 | 方向 | 句对 |
| --- | --- | --- | --- |
| en->es | 50 | zh->pt | 49 |
| en->it | 45 | zh->it | 45 |
| en->pt | 49 | zh->es | 50 |
| en->zh | 53 | zh->en | 53 |
| es->en | 50 | pt->zh | 49 |
| es->it | 45 | pt->it | 45 |
| es->pt | 49 | pt->es | 49 |
| es->zh | 50 | pt->en | 49 |
| it->en | 45 | it->zh | 45 |
| it->es | 45 | it->pt | 45 |
| it->pt | 45 | it->es | 45 |
| it->zh | 45 | it->en | 45 |
| pt->en | 49 | es->zh | 50 |
| pt->es | 49 | es->pt | 49 |
| pt->it | 45 | es->it | 45 |
| pt->zh | 49 | es->en | 50 |
| zh->en | 53 | en->zh | 53 |
| zh->es | 50 | en->pt | 49 |
| zh->it | 45 | en->it | 45 |
| zh->pt | 49 | en->es | 50 |

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
