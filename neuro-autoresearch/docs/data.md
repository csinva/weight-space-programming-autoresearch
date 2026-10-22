# Prepared data

Prepared data is reused in place; no dataset download or multi-gigabyte copy is
needed. Defaults are the existing shared files:

| Dataset | Response directory environment variable | Default response pickle |
| --- | --- | --- |
| LAION-fMRI | `LAION_DATA_DIR` | `/data/shared/datasets/laion_fmri/processed/laion_data.pkl` |
| Triple-N | `TRIPLEN_DATA_DIR` | `/data/shared/datasets/triplen/processed/triplen_data.pkl` |
| NSD | `NSD_DATA_DIR` | `/data/shared/datasets/nsd_processed/nsd_data.pkl` |
| TVSD | `TVSD_DATA_DIR` | `/data/shared/datasets/papale2025.tvsd/processed/tvsd_data.pkl` |

Directory overrides retain these filenames; `NSD_DATA_FILE` can select a different
NSD pickle within `NSD_DATA_DIR`.

Stimulus paths follow the source loaders: `LAION_FMRI_DIR`, `NSD_ROOT`,
`NSD_STIMULI_HDF5_PATH`, and `TVSD_DATA_DIR`. NSD's ncsnr volumes are also needed
for its final reliability filter. Paths can be set in the shell or local `.env`.
The old `scripts/prepare_data` baked formats are retired for these vision tasks.
