# 许可与来源追踪

## NIST JARVIS-DFT

- 3D 原始归档：`raw/jarvis_dft3d_2025.zip`，Figshare 条目 6815699，文件 `jdft_3d-9-24-2025.json.zip`，数据集 DOI `10.6084/m9.figshare.6815699.v11`。
- 2D 原始归档：`raw/jarvis_dft2d_2022.zip`，Figshare 条目 6815705，文件 `d2-12-12-2022.json.zip`，数据集 DOI `10.6084/m9.figshare.6815705.v8`。
- Figshare 条目声明许可证为 CC BY 4.0。再分发或发表结果时应引用对应数据集与 JARVIS 论文。
- 官方入口：<https://jarvis.nist.gov/>；数据说明：<https://pages.nist.gov/jarvis/databases/>。

## OpenAlex

- `literature_openalex.*` 于 2026-08-02 通过 OpenAlex Works API 获取。
- 检索式保存在 `literature_collection_summary.json` 和采集脚本中。
- OpenAlex 元数据会持续更新，引用计数和开放链接是采集时快照。
- 官方 API：<https://docs.openalex.org/>。

## Europe PMC

- `literature_europe_pmc.csv`、`open_access_table_rows.csv` 和 `open_access_process_mentions.csv` 于 2026-08-02 通过 Europe PMC REST API 与 `fullTextXML` 接口获取。
- Europe PMC 中各文章的版权/开放许可证可能不同。抽取记录保留 `pmcid`、DOI 和 `source_url`；再利用时必须检查对应文章页面的具体许可证。
- 官方 API：<https://europepmc.org/RestfulWebService>。

## 人工整理主表

- `curated_ir_material_systems.csv` 是基于 `sources/curated_source_registry.csv` 中综述和开放资料的事实性汇总。
- 代表性范围是面向检索与方案设计的入口，不应替代对原始论文中温度、偏压、面积、带宽、噪声模型与测量条件的核查。
- 本项目生成的脚本、字段设计和说明文档可按项目自身许可使用；第三方原始数据和论文内容仍受其各自许可约束。

