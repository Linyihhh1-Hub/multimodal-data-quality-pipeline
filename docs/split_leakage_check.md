# Train/Eval 数据泄漏检测

## 为什么需要检查泄漏

训练集和评测集之间如果存在相同样本、相同图片或近重复图片，评测指标会被系统性抬高。模型可能没有真正学会泛化，只是在评测阶段重新遇到了训练时见过的视觉内容或高度相似内容。

多模态数据中，文本 caption 可能不同，但图片主体完全相同。如果同一张图或近重复图同时出现在训练集和评测集，会导致模型在评测时见过视觉内容，从而造成评测结果虚高。因此项目在导出训练集和评测集后增加 split leakage 检测，检查 sample_id、image_id、image_path 和 perceptual_hash 是否跨 split 重复。

## 检查口径

模块位置：

```bash
src/quality/split_leakage.py
```

默认检测 `train` 与 `val` / `eval` 之间的跨 split 重复，检查字段包括：

- `sample_id`：同一个样本不能同时进入训练和评测。
- `image_id`：同一张图片即使 caption 不同，也不能跨训练和评测。
- `image_path`：防止同一路径文件被重复分配。
- `perceptual_hash`：检查视觉近重复图片。
- `duplicate_group_id`：检查已经聚类出的重复图片组。

如果某些字段不存在，检测会降级处理，并在报告的 `missing_keys` 中记录，不会中断运行。

## COCO 一图多 Caption 场景

COCO 等图文数据集常见一个 `image_id` 对应多个 caption。如果按 caption 级别随机切分，可能出现同一张图的不同 caption 分别进入 train 和 eval。此时 eval 的文本没有泄漏，但视觉内容已经泄漏，VLM 评测仍然会偏高。

因此该模块把 `image_id` 作为核心泄漏键。只要同一个 `image_id` 同时出现在 train 和 eval，就应当视为风险样本，后续需要在图像级别重新划分或做去重重排。

## 使用方式

从 Parquet 质量元数据运行：

```bash
python -m src.quality.split_leakage \
  --metadata data/processed_clip_coco/processed_metadata_coco_clip_v1.1.parquet \
  --output data/processed_clip_coco/split_leakage_report.json
```

从导出的 JSONL 文件运行：

```bash
python -m src.quality.split_leakage \
  --train data/exports_clip_coco_v1.1/train.jsonl \
  --eval data/exports_clip_coco_v1.1/eval.jsonl \
  --output data/exports_clip_coco_v1.1/split_leakage_report.json
```

输出报告示例：

```json
{
  "has_leakage": false,
  "total_leakage_count": 0,
  "leakage_by_key": {
    "sample_id": 0,
    "image_id": 0,
    "image_path": 0,
    "perceptual_hash": 0,
    "duplicate_group_id": 0
  },
  "checked_keys": ["sample_id", "image_id", "image_path"],
  "missing_keys": ["perceptual_hash", "duplicate_group_id"],
  "leakage_examples": []
}
```

如果存在泄漏，`leakage_examples` 只保留前 20 条，避免报告过大。

## 面试讲法

可以这样介绍：

> 我没有只做 accepted/rejected 的样本过滤，还在训练集和评测集之间加了数据泄漏检测。多模态数据里 caption 不同不代表图片不同，所以模块会同时检查 sample_id、image_id、image_path、perceptual_hash 和 duplicate_group_id。它能在导出后生成 JSON 报告，告诉我们是否有同图或近重复图跨 split 出现，避免评测集被训练数据污染。

这部分体现的是数据工程链路里的评测可信度治理：自动质检解决样本质量，review 回流解决边界样本，而 split leakage 检测保证 train/eval 隔离，避免模型效果被高估。
