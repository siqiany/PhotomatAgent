# VAE group-meeting package

本目录是一套可直接用于组会汇报的多性质条件 VAE 资料。

## 文件

- `figures/vae_inverse_generation_overview.svg|png|pdf`：方法总览图，SVG 文本和对象可编辑。
- `figures/vae_inverse_generation_overview_spec.json`：方法图的可复现源规格。
- `figures/vae_training_data_profile.svg|png|pdf|tiff`：训练数据覆盖率、演示条件位置和模型事实。
- `figures/make_vae_data_profile.py`：定量图生成脚本。
- `source_data/property_coverage.csv`：定量图源数据。
- `source_data/example_tool_call.json`：智能体工具调用参数。
- `source_data/example_result_summary.json`：固定随机种子的真实运行摘要。
- `run_agent_vae_example.py`：直接执行智能体 VAE 工具的可运行示例。
- `GROUP_MEETING_NOTES.md`：8–10 分钟讲解提纲、局限和常见问答。
- `FIGURE_QA.md`：图稿契约、数据完整性、模型指标定义和视觉 QA。

## 复现

```bash
.venv/bin/python docs/vae_group_meeting/figures/make_vae_data_profile.py
.venv/bin/python docs/vae_group_meeting/run_agent_vae_example.py

python3 <scientific-diagram-skill>/scripts/render_diagram.py \
  docs/vae_group_meeting/figures/vae_inverse_generation_overview_spec.json \
  -o docs/vae_group_meeting/figures/vae_inverse_generation_overview.svg
```

方法图导出 PNG/PDF 时使用 scientific-diagram 的 `export_svg.py`。图中所有数值均来自本项目打包数据、检查点或固定随机子的真实工具运行；没有使用模拟训练指标。
