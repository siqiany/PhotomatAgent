# VASP artifacts and scientific postprocessing

## Small vs large files

| Class | Files | Handling |
|---|---|---|
| SMALL (downloadable/readable) | OUTCAR, OSZICAR, EIGENVAL, vasprun.xml (iterparse if huge), POSCAR/CONTCAR, INCAR, KPOINTS, JSON results | downloaded to `results/<stage>/`; parsed into bounded summaries; never dumped raw into the model context |
| LARGE (remote-first) | WAVECAR, CHGCAR, LOCPOT (a 448³ LOCPOT is ~1.6 GB) | stay on the compute/login side by default; the agent only receives `sha256`, `size_bytes`, remote path reference; staged between remote job directories when needed |

Never read, print, log or commit POTCAR content; never copy private keys or
`.env` secrets into any output.

## LOCPOT (ESP / vacuum alignment)

- ESP requires the **LVHAR** LOCPOT (ionic + Hartree only).
- Large LOCPOT grids are analyzed with the streaming/chunked reader: header
  only (`grid`, lattice lengths, byte offset), then planar averages per axis
  in VASP x-fastest order — the full 3D grid is never materialized.
- Vacuum level = mean of the six boundary-layer faces for the declared
  thickness (0.5 / 1.0 / 1.5 / 2.0 Å supported); report the six-face
  mean/std/range and the thickness stability.
- If the six-face std exceeds the stability threshold, vacuum-aligned
  HOMO/LUMO are unreliable and must carry a warning.
- For very large remote LOCPOT files, analysis runs as an
  application-layer VASP postprocess or Slurm job on the compute node —
  there is NO generic remote shell.

## Orbital densities (PARCHG) vs ESP surfaces

- Orbital isosurface figures come from **PARCHG** (LPARD band density),
  including VASP 5.4.4 naming variants (`PARCHG.<band>`, `PARCHG.0001.<band>`).
- The ESP figure is a **geometry/dimension proxy** of the electrostatic
  potential from the LVHAR LOCPOT — it is NOT an electron-density isosurface.
- A proxy surface must be labeled as a proxy; it is never written, reported
  or plotted as a real CHGCAR isosurface.
- Binding/relative quantities must never mix "VASP-computed" values with
  defined-zero references (e.g. E(Li+) = 0) without the
  `explicit_reference_assumption` marker and high-risk note.

## Acceptable collection order

`prepare → preflight → submit → status/resume → collect → validate →
analyze/report`; results.json + evidence.json are written only after
validation; Slurm COMPLETED alone produces nothing scientific.

