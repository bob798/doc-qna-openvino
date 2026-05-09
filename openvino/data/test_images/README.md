# Phase 1 测试图片清单

按以下分类放入本目录，共 **10 张**，文件名前缀对应类型，方便脚本筛选与报告归档。

| 类型 | 数量 | 命名建议 | 说明 |
|------|------|----------|------|
| 纯文字 | 3 | `text_01.png` ~ `text_03.png` | 单栏/双栏纯文字段落，含中英混排 |
| 表格 | 3 | `table_01.png` ~ `table_03.png` | 含表头、合并单元格、密集数字 |
| 公式 | 2 | `formula_01.png` ~ `formula_02.png` | 含 LaTeX 风格数学公式 |
| 图文混排 | 2 | `mix_01.png` ~ `mix_02.png` | 含图片 + 文字 + 表格的混合页 |

> 来源建议：从产品规格书、学术论文、说明书中各抽取 1-2 页。

运行 `python scripts/benchmark_inference.py --image_dir data/test_images` 即可批量评测。
