# 质量元数据聚合示例

## 为什么需要聚合质量元数据

AI 训练数据治理不能只停留在逐样本清洗。逐样本 Pipeline 可以判断图片是否可读、caption 是否有效、图文相似度是否达标，以及样本最终进入 accepted、review 或 rejected；但项目展示、质量复盘和版本对比还需要把这些逐样本元数据聚合成可解释的指标。

本项目新增的 DuckDB 聚合示例读取 Parquet 质量元数据，输出总样本数、状态占比、过滤原因 TopN、质量分区间、CLIP 图文相似度、重复图比例、source/split 分布和 caption 长度等统计结果。这些结果可以继续供给 Streamlit 看板、质量报告、版本对比表和面试演示。

## 本地 Pipeline 与 DuckDB/Spark 聚合层的关系

本地 Pipeline 负责逐样本生成质量元数据，DuckDB/Spark 聚合层负责对大规模 Parquet 元数据做质量统计、版本对比和看板供数，从而把项目从脚本处理升级为数据工程分析链路。

在当前实现中：

- `src.pipeline` 等模块负责构建和评分样本，并写出 Parquet/JSONL。
- `src.distributed.duckdb_quality_agg` 读取已经落地的 Parquet 元数据。
- DuckDB 在本机进程内执行 SQL 聚合，不需要单独部署服务。
- 输出 JSON 可以直接进入看板、报告或简历项目材料。

Spark 暂时不作为默认实现。后续当数据量进一步扩大、数据存储进入对象存储或湖仓目录时，可以把同样的统计口径迁移到 Spark SQL：输入仍是 Parquet，输出仍是质量汇总表或 JSON，只是执行引擎从单机 DuckDB 切换到分布式 Spark。

## 百万级数据时的分析方式

当数据从 5000 条扩大到百万级时，建议保持“逐样本元数据 Parquet 化 + 聚合层 SQL 化”的结构：

1. Pipeline 仍然按批次生成样本级质量元数据，字段包括状态、质量分、CLIP 相似度、过滤原因、source、split、重复图标记等。
2. 每个版本写入独立目录，例如 `data/processed_clip_coco/v1.1/metadata.parquet` 或湖仓表分区。
3. DuckDB 可以直接读取本地或挂载目录中的 Parquet，快速完成版本内质量统计和版本间对比。
4. 当单机内存或执行时间成为瓶颈时，用 Spark SQL 读取同样的 Parquet 元数据，按数据版本、source、split、日期或任务类型聚合。
5. 聚合结果写回 Parquet/JSON，作为看板和报告的数据源，而不是让看板实时扫描所有样本。

这种结构的好处是：样本级明细可追溯，聚合指标可复用，执行引擎可从本地 DuckDB 平滑扩展到 Spark。

## 使用方式

示例命令：

```bash
python -m src.distributed.duckdb_quality_agg \
  --input data/processed_clip_coco/processed_metadata_coco_clip_v1.1.parquet \
  --output data/processed_clip_coco/quality_summary_coco_clip_v1.1.json
```

输出 JSON 会包含：

- `total_samples`
- `status_counts` 和 `status_ratios`
- `avg_quality_score`、`min_quality_score`、`max_quality_score`
- `avg_clip_score` 和 `clip_score_buckets`
- `duplicate_count` 和 `duplicate_ratio`
- `source_distribution`
- `split_distribution`
- `avg_caption_length`
- `filter_reason_topn`

字段名做了兼容处理：状态字段支持 `final_status`、`status`、`filter_status`；CLIP 字段支持 `clip_score`、`image_text_similarity`、`similarity_score`；样本 ID 支持 `sample_id` 或 `image_id`。缺失字段会返回 `None`、`0` 或空结果，不会中断聚合。

## 面试时怎么讲

可以这样描述：

> 这个项目不是只写了一个单机清洗脚本。我把逐样本质量评估结果标准化为 Parquet 元数据，然后增加 DuckDB 聚合层，对 accepted/review/rejected、过滤原因、CLIP 相似度、重复图、source 和 split 做统一统计。这样看板和报告不需要重新扫原始图片，而是消费聚合后的质量指标。数据量小时用本地 DuckDB 就能完成分析；扩大到百万级时，同一套 Parquet 元数据和 SQL 口径可以迁移到 Spark，形成从样本处理到质量分析再到看板供数的数据工程链路。

如果被追问 Spark，可以补充：

> DuckDB 是当前本地优先实现，适合单机开发和面试演示。Spark 不是重新写业务规则，而是替换聚合执行引擎；核心口径仍然是 Parquet 元数据上的 SQL 聚合。
