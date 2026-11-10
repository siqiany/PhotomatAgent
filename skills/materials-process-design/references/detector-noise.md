# Detector noise and responsivity

Record one device ID, active area, bias, temperature, wavelength, optical power, measurement frequency and bandwidth. Preserve whether the noise quantity is an integrated RMS current or an amplitude spectral density.

Use the infrared photodetector design skill for the measurement definitions. Record the acquisition sequence and a matched dark/noise control with `process.register_plan`. Use independent arithmetic only after checking the units and conditions. Measure and report the instrument floor separately.

Keep measured noise and shot-noise estimates in separate evidence records. A calculated detectivity from an assumed noise model leaves experimental detectivity UNKNOWN. Changing bandwidth without measuring the noise spectrum can invalidate area- and bandwidth-normalized comparisons.

Keep required lower bounds, ideal limits, measured points and conditional
predictions distinct in the conclusion. If a scenario sweeps uncertain inputs,
label the result as a scenario envelope and state what was varied together.
Use wavelength, temperature, bias and bandwidth to keep comparisons scoped.

Reference protocol: [Guidelines for accurate evaluation of photodetectors](https://www.nature.com/articles/s41566-025-01759-1).
