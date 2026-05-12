# `eval_questions.jsonl` Schema

每行一个 JSON 对象，字段如下：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | ✓ | 全局唯一，格式 `qNNN` |
| `type` | enum | ✓ | `single_concept` / `single_table_lookup` / `cross_doc` / `formula_concept` / `refusal` |
| `doc_ids` | string[] | ✓ | 引用的源文档 id（对应 `download_eval_materials.py` 配置中的 `id`）|
| `question` | string | ✓ | 中文问题原文 |
| `evaluation` | enum | ✓ | `keyword_match` / `must_refuse` / `manual_rubric` |
| `expected_keywords` | (string \| string[])[] | △ | `keyword_match` 必填。外层 AND，内层 OR——示例 `[["0.9B","900M"],["ERNIE-4.5"]]` 表示"必须包含 0.9B 或 900M **且** 必须包含 ERNIE-4.5" |
| `must_cite_doc` | string \| null | △ | 答案必须引用的文档 id；跨文档题为 `null` |
| `rubric` | string | △ | `must_refuse` / `manual_rubric` 必填，给人工评测员的判定标准 |
| `tags` | string[] | △ | 自由标签，便于切片统计 |

## 5 类问题分布（参考 MMLongBench-Doc / OmniDocBench 设计）

| `type` | 占比建议 | 验证什么 |
|--------|---------|---------|
| `single_table_lookup` | ~35% | 表格行/列定位精度（项目核心卖点）|
| `single_concept` | ~25% | 检索 + 文本理解基线 |
| `cross_doc` | ~20% | 多文档融合，对应"客服跨手册回答"场景 |
| `refusal` | ~15% | 幻觉防御（RAGAS faithfulness 配合）|
| `formula_concept` | ~5% | 公式 / LaTeX 解析能力 |

## 评测指标对接

- `keyword_match` → 直接 Python `all(any(k in answer for k in group) for group in expected_keywords)`
- `must_refuse` → 检测"未提及/未讨论/无相关信息/抱歉"等拒答关键词；命中且未编造细节则通过
- `manual_rubric` → 人工打 0/0.5/1 分
- 全集再过一遍 **RAGAS**：`faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`
