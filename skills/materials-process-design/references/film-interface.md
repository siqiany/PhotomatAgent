# Film formation and interface controls

Record substrate and surface treatment, precursor and solvent, atmosphere, thickness, deposition equipment, and thermal budget. Separate nucleation and phase evidence from final electronic alignment.

Register an original-substrate control and a proposed new substrate with `process.register_plan`. Record transferred conditions separately, state the substrate difference and keep the original process as a control. Add morphology and phase checkpoints before electrical interpretation. If equipment or thermal budget is missing, return a conditional plan.

When switching spin coating to blade coating, carry the ink and target film
identity forward. Reassess the blade gap, traverse speed, wetting and drying
conditions. Compare matched dry thickness and spatial uniformity before
transferring performance. Spin acceleration and rotation speed describe the
spin route. Treat any cross-route relationship as a testable assumption.
Carry over only settings the user supplied or supported records establish.
Choose an unknown drying setting by a same-ink control. Measure wet thickness
or wet mass directly for transfer calibration, or propagate uncertainty in
solids content and dry-density assumptions if estimating it from dry thickness.
If the target is outside the calibrated window, retain that finding. Test the
direction of any proposed change with the measured wet-to-dry relationship
before treating a new formulation or equipment setting as a viable branch.

A thickness-matched film on a different substrate can have different phase formation and interface states. A planar optical calculation can support an optical check under lateral homogeneity, with surface morphology and interface chemistry requiring separate measurements.

Reference system: [Substrate-dependent film formation of MAPbI₃](https://pmc.ncbi.nlm.nih.gov/articles/PMC5234022/).
