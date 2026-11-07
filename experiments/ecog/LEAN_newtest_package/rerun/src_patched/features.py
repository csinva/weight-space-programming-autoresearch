"""newtest copy of the harness feature module: the original evolve-wave/src/features.py is executed
in this namespace unchanged, then get_features_per_chunk is wrapped.

Why: grid-phase (Jun 6+) encoders do not call eval.run_encoding. Their own __main__ computes
features for the training list and fits the ridge itself (on the 125 grid electrodes; ridge is
per-electrode, so this equals the all-166 fit restricted to 0-124). When RESCORE_OUT is set and
the wrapper sees the frozen training list (or a prefix of it, for the few
trials evaluated with fewer training chunks), it hands the encoder's own embedder (and its
ndelays/lag/trim) to rescore_stream.rescore, which runs the same streamed fit as rescore-grid and
scores the old and new test sets, then exits. Those encoders also monkey-patch this function with
a disk-backed memmap version to save host RAM; while RESCORE_OUT is set that assignment is
ignored, so the streamed path (memory only, same values) is kept.
"""
import json
import os
import sys
import types

_ORIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "evolve-wave",
                     "src", "features.py")
exec(compile(open(_ORIG).read(), _ORIG, "exec"), globals())  # noqa: S102

_orig_get_features_per_chunk = get_features_per_chunk  # noqa: F821  (defined by the exec above)
_STATE = {"in_rescore": False, "train": None}


def _frozen_train():
    if _STATE["train"] is None:
        _STATE["train"] = json.load(open(os.path.join(os.environ["NEWTEST_DIR"], "train_frozen.json")))
    return _STATE["train"]


def get_features_per_chunk(stim_dict, resp_dict, chunks, embedder, ndelays=4, lag=LAG_FRAMES,  # noqa: F821
                           trim=TRIM_FRAMES):  # noqa: F821
    out = os.environ.get("RESCORE_OUT")
    if (out and os.environ.get("NEWTEST_DIR") and not _STATE["in_rescore"]
            and len(chunks) >= 20 and list(chunks) == _frozen_train()[:len(chunks)]):
        from .eval import EncodingConfig
        from .rescore_stream import rescore
        _STATE["in_rescore"] = True
        print(f"[newtest] intercepted training features: ndelays={ndelays} lag={lag} trim={trim}",
              flush=True)
        rescore(embedder, EncodingConfig(num_train=len(chunks), num_test=100, ndelays=ndelays,
                                         lag=lag, trim=trim), out)
        raise SystemExit(0)
    return _orig_get_features_per_chunk(stim_dict, resp_dict, chunks, embedder, ndelays=ndelays,
                                        lag=lag, trim=trim)


class _Guarded(types.ModuleType):
    def __setattr__(self, k, v):
        if k == "get_features_per_chunk" and os.environ.get("RESCORE_OUT") and v is not get_features_per_chunk:
            print("[newtest] kept streamed get_features_per_chunk (ignored encoder's monkey-patch)",
                  flush=True)
            return
        super().__setattr__(k, v)


sys.modules[__name__].__class__ = _Guarded
