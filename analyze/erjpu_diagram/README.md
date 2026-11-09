# Architecture diagrams of the final visual cortex model (erjpu)

Diagrams of the iWSP visual cortex model selected on LAION-fMRI: its V1-like
front end, its eight feature modules, its 9,996 output features and where its
nonlinear steps are.

- `arch.py` draws the text diagram `erjpu_architecture.{png,pdf}`. It reads the
  model's output layout (`atlas.json`, 36 blocks with widths, descriptions and
  per-feature families), which is exported with the visual cortex results and
  not stored here:
  `uv run python arch.py <atlas.json> erjpu_architecture.png`
- `panels.py` renders the picture panels in `panels/` used by the interactive
  diagram: `uv run python panels.py panels`
  (needs scikit-image for the sample photo).

The panels are illustrations: each stage is re-implemented in a few lines and
applied to one sample photo (scikit-image's `coffee`), so they show what each
kind of computation does, not the model's exact filters or parameters. Feature
counts come from the model's output layout. Nonlinearities are read from its
channel definitions and the paper appendix, not from the model code.

## How the eight modules map onto the six feature families

Families group features by what they encode; modules group them by the code
that computes them. Counts per module and family:

| Module | Edges | Color | Corners and contours | Luminance curvature | Texture statistics | Global layout | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 maps on foveated grids (5,190) | 3,114 | 2,076 | | | | | |
| Raw color layout (108) | 36 | 72 | | | | | |
| Texture boundaries (880) | | | | | 880 | | |
| Orientation and curvature (494) | 114 | | 380 | | | | |
| Edge phase (342) | 152 | | 190 | | | | |
| Surface geometry (1,022) | 146 | | 365 | 511 | | | |
| Co-occurrences (1,754) | | | | 111 | 1,400 | | 243 |
| Global layout (206) | | | | 48 | | 158 | |
| Total | 3,562 | 2,148 | 935 | 670 | 2,280 | 158 | 243 |
