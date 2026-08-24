# 数据字典

## 通用约定

- CSV 使用 UTF-8 with BOM，便于中文环境中的 Excel 直接打开。
- 空字符串表示来源未提供或不适用，不等于数值 0。
- 多值字段通常以分号 `;` 分隔；嵌套列表/表格单元格使用 JSON。
- 波长单位为 `µm`，带隙为 `eV`，温度若出现数值应回查表头确认单位。

## `curated_ir_material_systems.csv`

- `material_id`：本数据集稳定编号。
- `category`：元素、III-V、II-VI、IV-VI、超晶格、量子阱、CQD、二维、有机等类别。
- `material_system`：常用材料/平台名称。
- `formula_template`：化学式或层状结构模板；`x/y/m/n/N` 为变量。
- `composition_or_ratio`：代表性成分、合金分数或必须记录的比例参数。
- `tuning_variable`：主要光谱/性能调控变量。
- `target_spectral_region`：该平台常见目标波段，不代表每个样品都覆盖全部波段。
- `representative_cutoff_or_response_um`：综述层面的代表性范围，非保证值。
- `representative_bandgap_eV`：代表性或可调范围；超晶格/量子阱可为有效能隙/跃迁而非体材料带隙。
- `growth_or_synthesis`：常见生长或材料合成路线。
- `device_process_or_stack`：常见器件工艺、层栈或处理。
- `evidence_level`：`review-established` 或 `reported platform` 等证据级别。
- `numerical_scope_note`：数值的适用边界。
- `source_codes/source_dois/source_urls`：来源关联。

## `jarvis_*_ir_candidates.csv`

- `jarvis_id`：NIST JARVIS 唯一标识。
- `formula/elements/n_elements`：化学式、元素集合与元素数。
- `gap_selected_eV`：按 HSE → TBmBJ → OptB88vdW 选出的正带隙。
- `gap_method_selected`：所选带隙的计算方法。
- `cutoff_wavelength_um_from_gap`：按 `1.239841984/Eg` 推算的理想截止波长。
- `cutoff_region`：本项目定义的光谱分区。
- `optb88vdw_bandgap_eV/mbj_bandgap_eV/hse_bandgap_eV`：各计算方法原始值。
- `formation_energy_eV_per_atom`：每原子形成能。
- `energy_above_hull_eV_per_atom`：凸包上方能量；通常越低越接近热力学稳定。
- `density_g_cm3`：计算密度。
- `space_group_* / crystal_system`：晶体学信息。
- `dielectric_x/y/z/mean`：静态介电相关字段；具体计算设置见 JARVIS 文档。
- `avg_electron_mass_m0/avg_hole_mass_m0`：相对自由电子质量的平均有效质量字段。
- `bulk_modulus_GPa/shear_modulus_GPa`：弹性模量。
- `exfoliation_energy_meV_per_atom`：主要用于层状/二维材料。
- `max_IR_mode_cm-1/min_IR_mode_cm-1`：红外振动模式范围（有数据时）。
- `spillage`：SOC spillage 指标（有数据时）。
- `candidate_warning`：每条候选必须保留的使用警告。

## `literature_openalex.csv/.jsonl`

- `openalex_id/doi/title/authors/journal/publication_*`：文献标识与书目信息。
- `cited_by_count`：采集当日 OpenAlex 引用计数，不是永久值。
- `is_retracted`：OpenAlex 撤稿标记。
- `is_open_access/oa_status/landing_page_url/pdf_url`：开放获取与链接。
- `abstract/keywords`：OpenAlex 提供的摘要重建与关键词。
- `matched_queries`：该记录命中的本项目检索式，用于主题追踪。

## `open_access_table_rows.csv`

- `pmcid/doi/paper_title/publication_year`：来源论文。
- `table_index/table_label/table_caption`：论文内表格定位。
- `row_index`：表格内顺序，从 1 开始；通常首行是表头，但以原文为准。
- `cells_json`：完整单元格数组，推荐程序读取。
- `cells_text`：以 ` | ` 拼接的便于全文检索版本。
- `source_url`：Europe PMC 文章链接。

## `open_access_process_mentions.csv`

- `section`：证据所在 JATS 章节。
- `process_keywords`：命中的工艺词，如 MBE、MOCVD、CVD、ALD、退火、刻蚀、钝化等。
- `evidence_sentence`：开放全文中的定位句，最长 700 字符；复现前须回查原文上下文。

## `paper/abstract/abstracts.*`

- `paper_key`：去重主键；优先使用规范化 DOI，无 DOI 时使用 Semantic Scholar paper ID。
- `semantic_scholar_id/corpus_id/doi/pmid/pmcid`：论文标识符；来源未提供时为空。
- `title/abstract`：公开学术元数据接口返回的题名和摘要；不含 AI 生成摘要或正文片段。
- `publication_year/publication_date/work_type`：出版时间与文献类型。
- `authors/institutions/journal`：作者、机构和期刊元数据；多值字段以分号分隔。
- `is_open_access/oa_status`：论文全文的开放状态；`closed_or_unknown` 不影响摘要元数据收录。
- `cited_by_count`：采集时 Semantic Scholar 给出的引用计数，会随时间变化。
- `matched_terms/local_term_score`：题名或摘要直接命中的检索词及本地相关性分数；题名命中权重为 3，摘要命中权重为 1。
- `relevance_tier`：`core_title_match` 表示题名直接命中，`related_abstract_match` 表示摘要直接命中，`broad_query_match` 表示仅由批量检索召回，适合在高精度任务中排除。
- `topics/keywords`：Semantic Scholar 学科分类及 fields-of-study 字段。
- `search_scope/metadata_source/retrieved_at`：检索式、元数据来源与采集时间，供审计和复现。

同一数据提供三种形式：`abstracts.sqlite3` 用于断点续跑和 SQL 查询，
`abstracts.jsonl.gz` 用于机器学习/流式处理，`abstracts.csv.gz` 用于表格分析。
