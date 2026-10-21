"""Retired baked-format preparer; the canonical harness reads looped source data."""

MESSAGE = 'The legacy baked data/nsd/ format is retired. The canonical NSD loader now\nreads NSD_DATA_DIR/nsd_data.pkl, the official NSD_STIMULI_HDF5_PATH (imgBrick),\nand the eight official ncsnr volumes under NSD_BETAS_DIR/ppdata/subjNN/.\nPrepared responses on this machine: /data/shared/datasets/nsd_processed.\nUse the looped-networks nsd_data.pkl export layout with stimulus IDs and voxel\nx/y/z coordinates; do not generate the former per-subject .pt blobs.\nNSD is reserved for final encoding evaluation after a LAION-fMRI search.\n'


def main():
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
