# 人工复核回流

## 为什么不能只依赖自动规则

自动规则、图像质量检测和 CLIP 图文相似度适合做大规模初筛，但它们无法稳定处理所有边界样本。例如 caption 描述比较抽象、图片主体不明显、图文存在弱相关、或者分数刚好落在阈值附近时，直接 accepted 或 rejected 都可能引入错误。

因此，AI 训练数据治理需要把样本分成三类：

- `accepted`：自动规则足够确定，可以进入训练或评测导出。
- `rejected`：明显不符合质量要求，保留拒绝留痕。
- `review`：自动规则不确定，需要人工复核。

## review 队列的作用

`review_samples.jsonl` 是治理闭环中的人工复核入口。它保存自动质检无法直接判断的样本，让人工只处理高价值边界样本，而不是重新检查所有数据。

review 队列的价值在于：

- 降低人工审核成本。
- 避免边界样本直接污染训练集。
- 保存复核原因，便于后续调整规则和阈值。
- 形成可追踪的数据版本演进记录。

## 人工复核如何回流为新版本数据

本项目新增 `src.review.apply_review_feedback`，读取原始质量元数据 Parquet 和人工复核 JSONL，把人工决策应用到原始状态为 `review` 的样本上，并输出新版本 Parquet。

示例命令：

```bash
python -m src.review.apply_review_feedback \
  --metadata data/processed_clip_coco/processed_metadata_coco_clip_v1.1.parquet \
  --feedback examples/review_decisions_demo.jsonl \
  --output data/processed_clip_coco/processed_metadata_coco_clip_v1.2_reviewed.parquet \
  --version v1.2
```

人工复核决策格式：

```json
{"sample_id":"demo_001","decision":"accepted","reviewer":"human_001","reason":"caption matches image","review_time":"2026-06-26T10:00:00"}
{"sample_id":"demo_002","decision":"rejected","reviewer":"human_001","reason":"caption does not match image","review_time":"2026-06-26T10:01:00"}
{"sample_id":"demo_003","decision":"keep_review","reviewer":"human_001","reason":"ambiguous sample","review_time":"2026-06-26T10:02:00"}
```

脚本会新增这些字段：

- `review_decision`
- `reviewer`
- `review_reason`
- `review_time`
- `status_before_review`
- `status_after_review`
- `review_applied`
- `version`

默认只更新原始状态为 `review` 的样本。已经是 `accepted` 或 `rejected` 的样本即使出现在反馈文件里，也不会被覆盖，除非显式使用 `--overwrite-non-review`。

## 如何避免污染训练集

review 样本不能直接进入训练集。推荐流程是：

1. 自动质检先生成 `accepted/review/rejected`。
2. 只有 `accepted` 进入训练、验证或评测导出。
3. `review` 样本进入人工复核队列，不直接参与训练。
4. 人工复核结果回流生成新版本，例如 `v1.2`。
5. 新版本中人工确认的 `accepted` 才能进入下一轮训练数据导出。

这样可以把不确定样本隔离在 review 队列中，避免模型训练被低置信样本污染。

## 面试时怎么讲

可以这样描述：

> 自动规则和 CLIP 分数适合做大规模初筛，但边界样本仍然需要人工复核。项目通过 review 队列保存不确定样本，再通过人工决策文件回流生成 v1.2 数据版本，从而形成可追踪、可复现的数据治理闭环。

如果继续展开，可以补充：

> 这个模块体现的是数据工程里的版本治理思路：原始质量元数据不被直接覆盖，人工反馈作为单独输入进入回流脚本，输出新的 Parquet 版本，并保留 `status_before_review`、`status_after_review`、`reviewer` 和 `review_time`。这样既能追踪人工决策，也能在后续看板或质量报告里比较自动质检版本和人工回流版本的差异。
