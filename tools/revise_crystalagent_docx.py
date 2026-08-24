from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "wp": WP, "a": A}
WQ = f"{{{W}}}"


def paragraph_text(p):
    return "".join(t.text or "" for t in p.xpath(".//w:t", namespaces=NS))


def find_paragraph(body, prefix):
    matches = [p for p in body.findall(f"{WQ}p") if paragraph_text(p).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph starting with {prefix!r}, found {len(matches)}")
    return matches[0]


def replace_text(p, text):
    ppr = p.find(f"{WQ}pPr")
    first_rpr = p.find(f".//{WQ}rPr")
    first_rpr = copy.deepcopy(first_rpr) if first_rpr is not None else None
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    r = etree.SubElement(p, f"{WQ}r")
    if first_rpr is not None:
        r.append(first_rpr)
    t = etree.SubElement(r, f"{WQ}t")
    t.text = text


def new_paragraph_like(reference, text):
    p = etree.Element(f"{WQ}p", nsmap=reference.nsmap)
    ppr = reference.find(f"{WQ}pPr")
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    r = etree.SubElement(p, f"{WQ}r")
    t = etree.SubElement(r, f"{WQ}t")
    t.text = text
    return p


def insert_after(body, reference, paragraph):
    idx = body.index(reference)
    body.insert(idx + 1, paragraph)
    return paragraph


def keep_lines(p):
    ppr = p.find(f"{WQ}pPr")
    if ppr is None:
        ppr = etree.Element(f"{WQ}pPr")
        p.insert(0, ppr)
    if ppr.find(f"{WQ}keepLines") is None:
        ppr.append(etree.Element(f"{WQ}keepLines"))


def resize_drawing(drawing, width_inches):
    target_cx = int(width_inches * 914400)
    extent = drawing.find(f".//{{{WP}}}extent")
    if extent is None:
        return
    old_cx = int(extent.get("cx"))
    old_cy = int(extent.get("cy"))
    target_cy = int(old_cy * target_cx / old_cx)
    extent.set("cx", str(target_cx))
    extent.set("cy", str(target_cy))
    for aext in drawing.findall(f".//{{{A}}}ext"):
        if aext.get("cx") and aext.get("cy"):
            aext.set("cx", str(target_cx))
            aext.set("cy", str(target_cy))


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src, "r") as zin:
        document_xml = zin.read("word/document.xml")
        root = etree.fromstring(document_xml)
        body = root.find(f"{WQ}body")
        if body is None:
            raise RuntimeError("DOCX body not found")

        replacements = {
            "The Myth of the Universal Model:":
                "Crystal-system-aware benchmarking and routing of graph neural networks for materials-property prediction",
            "Community benchmarks for crystal property prediction routinely report aggregate metrics;":
                "Community benchmarks for crystal-property prediction routinely report aggregate metrics, but no single graph neural network (GNN) configuration performs optimally across all crystal-symmetry classes. Here we evaluate six released model configurations for formation energy and four for band-gap prediction across 154,373 Materials Project structures spanning all seven crystal systems. The routing-construction split was used to define and freeze the model-selection table, and formation-energy performance was then assessed on a disjoint held-out split. We find pronounced system dependence: SevenNet leads in triclinic structures; MEGNet in monoclinic, trigonal and hexagonal structures; M3GNet in tetragonal and cubic structures; and ALIGNN in orthorhombic structures. CrystalAgent converts this complementarity into an auditable routing rule, reducing formation-energy MAE by 20.1% relative to the best aggregate single-model baseline. In the band-gap analysis, which used a supervised CHGNet head whose loss was monitored on the evaluation split, routing reduced MAE by 5.9% relative to the best aggregate single model. Controlled structural analyses further show that model responses vary with lattice geometry, although simple descriptors explain only part of the complementarity. These results establish crystal-system-resolved evaluation as a practical basis for comparing and routing released model configurations within the evaluated model pool and Materials Project distribution.",
            "We ask a more targeted question:":
                "We ask a deployment-focused question: given a crystal structure's symmetry class, which available GNN configuration should be used for a specified property? Crystal system provides a simple and auditable conditioning variable that can expose performance heterogeneity hidden by aggregate metrics. Because the evaluated checkpoints differ in training data, target conventions and inference implementations, however, the resulting rankings describe deployable configurations and should not be interpreted as isolating architectural effects.",
            "Our contributions are threefold.":
                "Our contributions are threefold. First, we provide a crystal-system-resolved benchmark showing that the ranking of released model configurations changes across symmetry classes. Second, controlled lattice-perturbation case studies and a structural-descriptor analysis quantify model-response heterogeneity while showing that simple descriptors explain only part of the observed complementarity. Third, we encode the routing-construction results in CrystalAgent, an automated framework that freezes one model choice for each crystal-system-property pair before evaluation on the disjoint split. Figure 1 summarizes the workflow.",
            "Figure 1:Overview":
                "Figure 1. Overview of the CrystalAgent framework. Input crystal structures (CIF files; formula-only queries require a resolved structure) are parsed using pymatgen and spglib to determine crystal system and space group. A routing table constructed from crystal-system-resolved MAEs selects a model for the requested property. The selected model is executed and its output is passed to the reporting and physical-consistency checks.",
            "All structures were retrieved from the Materials Project database":
                "All structures were retrieved from the Materials Project database, yielding 154,373 inorganic crystal structures with DFT-computed formation energies and PBE band gaps. Structures were partitioned at the structure level using a crystal-system-stratified 70/30 split across the seven systems (triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal and cubic). The routing-construction set (108,059 structures; 70%) was used to construct and freeze the routing table, whereas the non-overlapping evaluation set (46,314 structures; 30%) was reserved for subsequent assessment. Each crystal system was proportionally represented in both partitions, and no evaluation-set structure informed route construction or route reselection. As disclosed in Methods 2.2, evaluation-set band-gap loss was monitored during training of the CHGNet regression head, although the loss decreased through the fixed 100-epoch cap and early stopping was not triggered. Space-group numbers were extracted from the CIF metadata. The seven crystal systems are illustrated in Figure 2.",
            "2.2 Model Selection":
                "2.2 Model configuration, target harmonization and prediction failures",
            "Figure 3. Architecture":
                "Figure 3. Architecture of the CrystalAgent knowledge graph (44 nodes, 132 edges and 7 relation types). Layer 1 (routing query) represents crystal-system-model edges weighted by routing-construction MAE. Layer 2 (tool tracing) represents model-to-preprocessing-tool dependencies. Layer 3 (physics chain) represents property-to-property implication edges used for consistency checks. The lower callouts show the sparse-data fallback (n < 50, inverse-MAE-weighted ensemble) and incremental model addition.",
            "2.5 Statistical analysis.":
                "2.5 Statistical analysis",
            "The routing table and the per-system winner/runner-up pairs are predefined":
                "The routing table and the per-system winner-runner-up pairs were predefined on the routing-construction split (108,059 structures; 70%), and the evaluation split (46,314 structures; 30%) was used to test these preselections. Paired bootstrap tests were applied to per-structure absolute-error differences. In each of B = 10,000 resamples, structures were sampled with replacement while retaining identical structure indices across the compared models, and the statistic was the mean difference |ŷA - y| - |ŷB - y|. Two-sided empirical p values were computed from the bootstrap distribution and are reported as p < 0.001 when no resample crossed zero; exact values are provided in Supplementary Table S6f. The 14 winner-runner-up comparisons (seven crystal systems by two properties) were adjusted for multiplicity using the Holm procedure. Structural-descriptor classification performance is reported as the mean ± s.d. across 25 fold-level AUC values from repeated stratified five-fold cross-validation (five repeats; stratified by the binary model-win label; the structure was the cross-validation unit; Supplementary Methods S1.3). Error correlations in Figure 6 were computed from signed per-structure errors (ŷi - yi), rather than absolute errors.",
            "Figure 4a summarizes formation-energy MAE":
                "Figure 4a summarizes formation-energy MAE for all six model configurations on their finite-prediction subsets of the evaluation split (up to n = 46,314; Methods 2.2). SevenNet achieved the lowest MAE in triclinic structures (0.1036 eV/atom); MEGNet led in monoclinic (0.0890 eV/atom), trigonal (0.0928 eV/atom) and hexagonal (0.0753 eV/atom) structures; M3GNet led in tetragonal (0.1013 eV/atom) and cubic (0.0847 eV/atom) structures; and ALIGNN led in orthorhombic structures (0.1561 eV/atom).",
            "For band gap (Figure 4b)":
                "For band gap (Figure 4b), the performance ranking was more heterogeneous. CHGNet, represented by a frozen backbone and the supervised regression head described in Methods 2.2, achieved the lowest MAE in triclinic (0.431 eV), monoclinic (0.500 eV) and orthorhombic (0.365 eV) structures. ALIGNN led in tetragonal (0.289 eV), hexagonal (0.189 eV) and cubic (0.245 eV) structures, and the two models were statistically indistinguishable in trigonal structures (ALIGNN, 0.437 eV; CHGNet, 0.453 eV; paired bootstrap p = 0.20; Figure 5b). CGCNN had the highest band-gap MAE across all systems (0.69-0.85 eV). Because the evaluation-set loss was monitored during CHGNet-head training, these band-gap estimates were obtained on structures excluded from parameter fitting but do not constitute performance on an untouched independent test set.",
            "For band gap (Figure 7b)":
                "For band gap (Figure 7b), the router achieved an MAE of 0.3765 eV, compared with 0.4001 eV for the best aggregate single model and 0.4345 eV for the ensemble, corresponding to a 5.9% reduction relative to the single-model baseline. The router also outperformed the ensemble for this model pool. These values use the monitored evaluation split described in Methods 2.2 and therefore should not be interpreted as estimates from an untouched independent test set.",
            "This comparison is intentionally deployment-oriented.":
                "This comparison is intentionally deployment-oriented. We evaluated publicly released checkpoints without harmonizing their original training data or target definitions, except for the supervised CHGNet band-gap head, whose backbone was frozen and whose regression head was trained on the routing-construction split. SevenNet and MACE total energies were converted to formation energies using elemental reference values fitted exclusively on the routing-construction split. The results therefore quantify the performance of the evaluated model configurations on Materials Project structures rather than intrinsic architectural superiority under matched training conditions.",
            "Several limitations bound this interpretation.":
                "Several limitations bound this interpretation. The evaluation split was not used to construct or reselect the routing table, but individual structures may have appeared in the original pretraining corpora of released checkpoints. The CHGNet band-gap head was fitted on routing-construction structures, and its loss was monitored on the evaluation split through the fixed 100-epoch schedule; consequently, the band-gap results are not based on an untouched independent test set and may be optimistically biased. The model pool covers only formation energy and band gap, routing is conditioned only on crystal system, and the current routing matrix does not adapt online to new chemical spaces. The reported gains should therefore be interpreted within the evaluated model pool and Materials Project distribution rather than as evidence of strict de novo or out-of-distribution generalization.",
            "The frozen crystal-system router reduced formation-energy MAE":
                "The frozen crystal-system router reduced formation-energy MAE by 20.1% relative to the best aggregate single-model baseline on the independent evaluation split. In the monitored band-gap evaluation, the corresponding reduction was 5.9%. The formation-energy ensemble remained slightly more accurate than the hard router, whereas the band-gap router outperformed the ensemble, illustrating that the value of routing depends on the error structure of the available model pool."
        }

        for prefix, text in replacements.items():
            replace_text(find_paragraph(body, prefix), text)

        model_p = find_paragraph(body, "We evaluate six models for formation energy prediction")
        model_paragraphs = [
            "We evaluated six released model configurations for formation energy: M3GNet and MEGNet (MatGL implementations), ALIGNN (JARVIS pretrained weights), CGCNN (original pretrained weights), SevenNet and MACE-MP-0. Band-gap comparisons included MEGNet, ALIGNN, CGCNN and a CHGNet-based predictor. Model and checkpoint identifiers are listed in Supplementary Table S2. Except for the target harmonization and CHGNet head described below, checkpoints were used as released without retraining or fine-tuning. The experiment is therefore a deployment benchmark of directly usable configurations and not a comparison of architectures trained on matched data.",
            "SevenNet and MACE output total energies, which were converted to formation energies using fitted elemental reference energies. For structure i, the design row was the fractional composition vector xᵢ,el = nᵢ,el/nᵢ and the target was yᵢ = Eᵢ/nᵢ − Eform,iDFT. Elemental corrections were estimated by ridge regression, δ = (XᵀX + λI)⁻¹Xᵀy, with λ = 0.1, using the first 5,000 entries of the routing-construction split in dataset order; no random sampling or random seed was used. The corrected formation energy was Eform/N = Etot/N − Σel[(nel/N)μel], where μ = μcode + δ. The fit was performed separately for SevenNet and MACE, and the final shared table was the element-wise average of the two solutions. The table covers 85 elements; the maximum cross-model disagreement was 0.031 eV/atom (Supplementary Table S6e). This averaging accommodates small model-specific residual-energy conventions while retaining one shared reference table. No evaluation-set structure was used in either fit.",
            "For band-gap prediction, the released CHGNet backbone was frozen and its 64-dimensional structure features were passed to a four-layer multilayer perceptron with widths 64-128-64-32-1. Each of the three hidden layers comprised Linear, BatchNorm1d, SiLU and Dropout (rate 0.1), followed by a linear output layer. The head was trained on frozen features from 108,056 routing-construction structures for up to 100 epochs using Adam (learning rate 10⁻³, weight decay 10⁻⁵, batch size 256). Labels were normalized using the training-set mean (1.064 eV) and s.d. (1.515 eV). ReduceLROnPlateau used a factor of 0.5 and patience of 8 epochs, and early stopping used a patience of 15 epochs. Evaluation-set loss was monitored during training; it decreased through epoch 100, so early stopping was never triggered and the reported checkpoint is the fixed 100-epoch model. The evaluation split thus acted as a monitoring set for this head rather than an untouched test set. Its reported MAEs were nevertheless calculated on structures not used to fit the head parameters.",
            "Failed or non-finite predictions were handled prospectively by analysis type. During routing-table construction, the recorded model-specific exclusions were 324 structures for MEGNet, 3 for ALIGNN, 3 for CGCNN and 140 for the CHGNet head; M3GNet, SevenNet and MACE had no exclusions. Descriptive evaluation tables used each model's finite-prediction subset (formation energy, at most 46,314 structures; band gap, 46,313 labelled structures). Paired bootstrap inference used complete-case intersections so that every comparison retained identical structure indices: n = 46,187 for formation energy after excluding 127 non-finite MEGNet predictions, and n = 46,272 for band gap after excluding any structure with a non-finite prediction. SevenNet and MACE produced no failed or non-finite predictions on the evaluation split."
        ]
        replace_text(model_p, model_paragraphs[0])
        cursor = model_p
        for text_value in model_paragraphs[1:]:
            cursor = insert_after(body, cursor, new_paragraph_like(model_p, text_value))

        stats_p = find_paragraph(body, "The routing table and the per-system winner-runner-up pairs were predefined")
        stability_methods = (
            "To assess the stability of the frozen routing decisions, we performed B = 10,000 stratified bootstrap resamples within each crystal system on the evaluation set, sampling structures with replacement while retaining the original stratum sizes. The routing table, constructed exclusively from the routing-construction split, remained fixed throughout. For each resample, we recomputed the per-model MAEs, the rank of the preselected routing model, and the MAE differences between the router and either the best aggregate single model or the inverse-MAE-weighted ensemble. The evaluation data were not used to update or reselect a route. Routing agreement was defined as the percentage of bootstrap resamples in which the preselected routing model retained rank 1 within its crystal system."
        )
        insert_after(body, stats_p, new_paragraph_like(stats_p, stability_methods))

        bg_route_p = find_paragraph(body, "For band gap (Figure 7b)")
        stability_results = (
            "The frozen routes were stable under stratified resampling (Figure 7c,d). For formation energy, the preselected model retained rank 1 in 100% of bootstrap resamples in every crystal system. For band gap, agreement was 100% in five systems, 99% in hexagonal structures and 90% in trigonal structures, consistent with the small and non-significant trigonal difference in Figure 5b. At the aggregate level, the formation-energy router outperformed the best single model (mean MAE difference, router minus comparator = -0.0267 eV/atom; 95% percentile CI, -0.0296 to -0.0237) but was slightly less accurate than the ensemble (MAE difference = +0.0024 eV/atom; 95% CI, +0.0009 to +0.0039). The band-gap router outperformed both the best single model (MAE difference = -0.0235 eV; 95% CI, -0.0273 to -0.0198) and the ensemble (MAE difference = -0.0580 eV; 95% CI, -0.0620 to -0.0539). Negative values favour routing."
        )
        insert_after(body, bg_route_p, new_paragraph_like(bg_route_p, stability_results))

        caption = find_paragraph(body, "Figure 7. Performance comparison")
        replace_text(
            caption,
            "Figure 7. Performance and stability of the frozen crystal-system router. (a) Formation-energy MAE and (b) band-gap MAE for the router (blue), the best aggregate single model (red) and the inverse-MAE-weighted ensemble (orange dashed line). Overall descriptive values use the finite-prediction evaluation samples; paired comparisons use the complete-case samples defined in Methods 2.2 and Supplementary Table S6. (c) Bootstrap agreement between the route frozen on the routing-construction split and the rank-1 model recomputed in stratified evaluation-set resamples. Bars show the percentage of B = 10,000 resamples in which the preselected model retained rank 1 within each crystal system (blue, formation energy; orange, band gap); no route was updated or reselected from evaluation data. (d) Bootstrap distributions of the MAE difference (router minus comparator) for the best aggregate single model and inverse-MAE-weighted ensemble. Negative values favour the router. Solid vertical lines indicate the bootstrap mean, dashed vertical lines indicate zero, and reported intervals are 95% percentile confidence intervals. The resampling unit was the structure, sampled with replacement within crystal-system strata while retaining stratum sizes."
        )
        keep_lines(caption)
        body.remove(caption)

        panel_c_label = find_paragraph(body, "(c)")
        panel_d = find_paragraph(body, "(d)")
        panel_c_image = body[body.index(panel_c_label) + 1]
        drawings_c = panel_c_image.xpath(".//w:drawing", namespaces=NS)
        drawings_d = panel_d.xpath(".//w:drawing", namespaces=NS)
        if len(drawings_c) != 1 or len(drawings_d) != 1:
            raise RuntimeError("Could not identify Figure 7c/d drawings")
        resize_drawing(drawings_c[0], 3.05)
        resize_drawing(drawings_d[0], 3.05)
        insert_after(body, panel_d, caption)

        for p in body.findall(f"{WQ}p"):
            if paragraph_text(p).startswith("Figure "):
                keep_lines(p)

        output_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")

        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = output_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    print(dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: revise_crystalagent_docx.py INPUT.docx OUTPUT.docx")
    main(sys.argv[1], sys.argv[2])
