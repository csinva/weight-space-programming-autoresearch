"""Retired baked-format preparer; the canonical harness reads looped source data."""

MESSAGE = 'The legacy baked data/nnn/ format is retired. The canonical Triple-N loader now\nreads TRIPLEN_DATA_DIR/triplen_data.pkl and NSD_STIMULI_HDF5_PATH (imgBrick).\nPrepared responses on this machine: /data/shared/datasets/triplen/processed.\nFor raw-data preparation, use looped-networks/prepare_data/triplen.py at source\nrevision 3ee575baeceb1ad784102162a051b7444182fdaa, then set the paths in .env.\nValidate the result with: python -m search.preflight triplen\n'


def main():
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
