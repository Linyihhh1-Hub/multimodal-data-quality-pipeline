# 图文多模态数据质量 Pipeline 面试讲稿

## 1 分钟项目介绍

这个项目面向视觉语言模型训练数据生产场景，目标是把原始图文数据加工成可用于训练、微调和评测的高质量数据集。我实现了从 COCO Captions 接入、图片和文本基础质检、CLIP 图文一致性评分、近重复检测、样本 accepted/rejected/review 分层、Parquet/JSONL 导出，到质量报告、静态样本画廊和版本对比的完整离线 Pipeline。

当前本地已经下载并处理 COCO val2017，跑通 5000 条 captions。基础规则版本通过 4960 条，拒绝 40 条；真实 CLIP 版本也已跑通，并通过 v1.0/v1.1 阈值迭代把 3677 条边界样本转入 review 队列，用于体现数据质量治理闭环。

详细项目指标见 `docs/project_showcase.md`。

## 简历 Bullet

- 构建面向视觉语言模型训练数据生产的图文多模态质量治理 Pipeline，完成 COCO Captions 5000 条样本接入、图片/文本质检、图文一致性评分、样本分层过滤和训练/评测 JSONL 导出。
- 基于 Pillow/OpenCV/imagehash 实现图片损坏、低分辨率、宽高比异常、模糊、亮度异常和近重复图片检测，并沉淀 `filter_reason`、`perceptual_hash`、`duplicate_group_size` 等质量元数据。
- 接入 HuggingFace CLIP，以 image/text embedding cosine similarity 计算图文一致性分数，并结合图片质量、文本质量构建综合质量评分。
- 设计 v1.0/v1.1 规则迭代机制，支持通过配置调整接受/复核阈值；在 COCO+CLIP 5000 条样本上将 3677 条边界样本转入 review 队列，生成版本对比报告。
- 构建 Streamlit 看板、Markdown 质量报告和静态样本画廊，展示通过率、过滤原因、图文相似度、重复图分布、低质样本案例和 caption 高频标签。

## 面试追问

### 为什么原始 COCO caption 不能直接训练？

原始数据不一定满足训练数据要求。即使 COCO 是高质量公开数据，也会存在低分辨率图、一图多 caption、caption 长短不一致、图文一致性分布不均等问题。训练前需要把这些问题显式量化，形成可追踪的质量元数据，而不是只把图片和 caption 拼成 JSONL。

### 为什么引入 CLIP？

规则质检只能判断图片是否可用、caption 是否为空或过短，但不能判断图片和文本是否语义一致。CLIP 提供了模型辅助的图文一致性分数，可以把样本从“格式可用”进一步评估到“语义是否适合训练”。项目里用 image/text embedding 的 cosine similarity，而不是单图单文 softmax，避免单样本 softmax 恒为高分的问题。

### 为什么不直接 reject 所有重复图？

COCO Captions 天然是一图多 caption。重复图不一定是坏数据，它可能提供不同语言描述角度。所以我先把近重复识别为元数据字段，例如 `is_duplicate_image` 和 `duplicate_group_size`，而不是默认过滤。后续可以按训练目标选择保留多 caption、只保留最高质量 caption，或者按 image_id 做评测集去重。

### v1.0 到 v1.1 体现了什么闭环？

v1.0 使用基础质量阈值，主要完成可用样本筛选。v1.1 基于 CLIP 分数分布提高接受阈值，把低于更高质量线但仍可人工判断的样本转入 review 队列。这个过程体现了从质量分布分析、规则调整、版本重跑到结果对比的数据闭环。

### 通过率高是不是说明项目没价值？

不是。COCO 本身是高质量公开数据，所以基础规则通过率高是合理的。项目价值不在于强行过滤很多数据，而在于建立可解释、可复现、可迭代的质量治理体系。换到业务采集数据、网页图文数据或 OCR 截图数据时，同一套 Pipeline 可以识别更多真实问题。
