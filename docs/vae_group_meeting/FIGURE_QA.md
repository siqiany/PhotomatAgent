# Figure contract and QA notes

## Figure claim

The packaged conditional VAE performs stochastic inverse generation of new material compositions from sparse combinations of trained material properties; the reliability of each condition is bounded by its label coverage, and every generated formula remains unvalidated until structure and physics calculations are completed.

## Figure contract

- Archetype: schematic-led group-meeting package with one quantitative evidence slide.
- Backend: Python only for plotting, previewing and export.
- Final size: 16:9 widescreen, 13.333 × 7.5 inches, optimized for group-meeting projection rather than a journal column.
- Editable outputs: SVG and PDF.
- Projection output: 300 dpi PNG.
- Archival raster output: 600 dpi LZW-compressed TIFF for the quantitative slide.

## Panel logic

- Method figure A: natural-language request is converted to a sparse multi-property tool input.
- Method figure B: training-only Encoder path is separated from inference-time prior sampling and Decoder generation.
- Method figure C: continuous composition fractions are integerized and filtered, then explicitly routed to future validation.
- Data figure a: property-label coverage establishes which conditions have strong or weak training support.
- Data figure b: the example targets are located relative to the robust training center and IQR.
- Data figure c: model size and training facts provide reproducibility context.

## Data integrity

- Source: all 11,240 rows in the packaged inverse-index arrays.
- Sampling: none.
- Row exclusion: none for coverage estimates. Property-specific statistics use the model's stored presence mask, and the valid count is shown for every field.
- Simulated data: none.
- The example candidate formulas come from an actual run of the packaged VAE checkpoint with seed 23.
- The model validation MAE is reported exactly as stored in the checkpoint. It is an element-fraction reconstruction MAE, not a property-prediction error.

## Statistics and ML reporting

- Split: random 80/20 train/validation split used by the training script.
- Training seeds: one reported seed (42); no uncertainty interval is available.
- Epochs: 10.
- Batch size: 256.
- Loss: composition reconstruction plus β-weighted KL divergence, β = 0.02.
- Sparse conditioning: 45% per-condition dropout during training.
- Validation metric: mean absolute error over the 89-dimensional reconstructed composition fractions.

## Visual QA

- Native SVG labels, arrows and shapes; no external or generated bitmap assets.
- Diagram SVG validator: pass with zero warnings.
- Quantitative-source validator: 13 passes, zero failures, one expected width warning because the output is 16:9 rather than 89/183 mm journal width.
- Font: explicitly registered Chinese sans-serif with Arial/Helvetica/DejaVu fallbacks.
- Color: restrained blue–violet–amber–green palette; labels and line styles preserve meaning without relying on color alone.
- Final PNG previews were visually inspected for missing glyphs, clipping, overlap and arrow direction.

## Reviewer and discussion risks

- Low label coverage for exfoliation energy, IR modes and spillage limits confidence in those conditioning dimensions.
- Long-tailed or anomalous values exist in several raw properties; robust median/IQR normalization and ±8 clipping reduce but do not eliminate this risk.
- Formula novelty only means absence from the configured reduced-formula reference set.
- Charge-neutrality heuristics do not establish thermodynamic stability, synthesizability or target-property attainment.
- No device-level target such as responsivity, detectivity, dark current or EQE is supported by this VAE.
