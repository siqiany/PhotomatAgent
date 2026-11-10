# Ligand exchange and carrier transport

Record the dot composition, size distribution, original and exchanged ligand, solvent, substrate, film thickness, atmosphere, bias and temperature. Use a stable sample ID for each combination. Distinguish a contact-limited measurement from bulk transport before interpreting mobility.

Use `method_search` to compare transport analysis with process controls. Use `literature.read_paper` and `literature.extract_evidence` to recover the original Methods and sample conditions. Use `process.register_plan` to record a conditional transfer and an original-ligand control on the same substrate. Pair each reported parameter with its own sample-bound evidence. Leave solvent or equipment unspecified when it is unknown.

Treat liquid and solid-phase exchange as separate operations with their own
solvent or carrier, dose basis, contact endpoint and specimen state. For a
conditional liquid route, distinguish precursor dissolution, the starting
dispersion and receiving exchange medium, the antisolvent, and the final
redispersion or coating solvent. Name both materials and the basis of every
ratio, such as antisolvent volume per dispersion volume or ligand moles per
estimated surface site. Keep an unknown basis explicit. On a matched small
aliquot, check dissolution, exchange or phase transfer, recovery and redispersion,
and whether a coated test coupon survives the next solvent without wash-off.
Carry forward only the ligand/medium combinations that pass. Use a
matched untreated or original-ligand control to test the tradeoff between
surface passivation, charge transport and film damage. A change in optical
response alone does not resolve that tradeoff.

If the outgoing dispersion and the next coating carrier differ in phase or
solubility, insert an explicit transfer step with a recovery and redispersion
check, or coat from the outgoing medium on a test coupon first. Do not call an
unresolved change of carrier part of a continuous baseline.

Compare thickness-matched transport curves under controlled atmosphere and matched contacts. Stop interpretation if oxidation changes the effective dot size, if contacts dominate, or if the fit omits temperature or field dependence. An improved photocurrent alone cannot establish increased mobility.

Reference system: [Ligand-dependent oxidation in PbS quantum-dot devices](https://arxiv.org/abs/2007.14678).
