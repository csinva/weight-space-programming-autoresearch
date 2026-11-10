"""multisubj copy of the harness data module: evolve-wave/src/data.py is executed here unchanged,
then, when MS_SPLIT points to a subject's split.json, the subject's directories, electrode count
and frozen train/test lists replace the patient1 constants. get_chunk_names returns the frozen lists
(num_train takes a prefix of the training list, as the harness's own pool[:num_train] does)."""
import json as _json
import os as _os

_ORIG = _os.path.join(_os.path.dirname(_os.path.realpath(__file__)), "..", "..", "..", "evolve-wave",
                      "src", "data.py")
exec(compile(open(_ORIG).read(), _ORIG, "exec"), globals())  # noqa: S102

MS_CFG = _json.load(open(_os.environ["MS_SPLIT"])) if _os.environ.get("MS_SPLIT") else None
if MS_CFG is not None:
    WAV_DIR = MS_CFG["data_dir"]
    WAVE_DIR = MS_CFG["wave_dir"]
    N_ELECTRODES = MS_CFG["n_electrodes"]

    def get_chunk_names(num_train=None, num_test=20, seed=42):  # noqa: F811
        train = MS_CFG["train"] if num_train is None else MS_CFG["train"][:num_train]
        return list(train), list(MS_CFG["test"])
