# Electrodeposition metrology

Record the actual total exposed cathode area, including exposed edges and
multiple faces when present. Keep the user-specified area for the stated case.
When only exposed area is supplied, it is the total; face count and layout
remain separate unknowns. Recheck the assumption in the operating steps.
Changing the exposed area at fixed thickness and efficiency scales the total
charge, while the charge per area stays the same.

Use Faraday's relation for a specified deposition reaction and density assumption.
With current density `j = I/A` and charge per area `σ = Q/A`, report
`Q = A σ` and `t = Q/I = Q/(j A) = σ/j` for constant current. Include these
conversions at the operating step, using the calibrated efficiency to estimate
the target charge. Stop by integrated charge when current varies. Keep predicted
thickness and efficiency separate from measurements.

Use a same-batch zero-current blank and several charge levels to check the
mass-gain calibration. Fit corrected mass gain versus measured total charge
with a free intercept to diagnose a residual baseline offset or bias. Use a
through-origin slope only if the blank, free-intercept estimate and residuals
support the zero-mass-at-zero-charge assumption and the slope is stable over the
working range. If the assumptions fail, resolve the baseline or efficiency
variation before using a single conversion factor.

Keep current density, temperature, agitation and electrochemical limits within
the applicable approved bath and equipment windows. Express missing settings
as constrained variables or pretests. Compare a same-batch thickness map from
edge and center with the area-average estimate to separate geometry effects from
overall efficiency.
If an approved operating window is missing, make a screening sweep conditional
on confirming a permitted envelope first. It is a design option, not a method
for granting authorization to operate outside that envelope.
