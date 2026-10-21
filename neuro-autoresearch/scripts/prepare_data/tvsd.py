"""Retired baked-format preparer; the canonical harness reads looped source data."""

MESSAGE = 'The legacy baked data/tvsd/ format is retired. The canonical TVSD loader now\nreads TVSD_DATA_DIR/tvsd_data.pkl and TVSD_DATA_DIR/tvsd_stimuli.h5.\nPrepared data on this machine: /data/shared/datasets/papale2025.tvsd/processed.\nFor raw-data preparation, use looped-networks/prepare_data/tvsd.py at source\nrevision 3ee575baeceb1ad784102162a051b7444182fdaa, then set TVSD_DATA_DIR in .env.\nTVSD is reserved for final encoding evaluation after a Triple-N search.\n'


def main():
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
