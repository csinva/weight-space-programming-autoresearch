"""multisubj copy of the harness feature module (as runs-wave/newtest/src_patched/features.py):
the original evolve-wave/src/features.py runs unchanged in this namespace, then
get_features_per_chunk is wrapped so that encoders whose own __main__ fits the ridge (the Jun 6+
grid missions) hand their embedder to rescore_stream when they request features for the frozen
training list (or a prefix of it). Their disk-memmap monkey-patch is ignored while RESCORE_OUT is
set (memory only, same values)."""
import json
import os
import sys
import types

_ORIG = "/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/src/features.py"
exec(compile(open(_ORIG).read(), _ORIG, "exec"), globals())  # noqa: S102

_orig_get_features_per_chunk = get_features_per_chunk  # noqa: F821
_STATE = {"in_rescore": False}


def _frozen_train():
    return json.load(open(os.environ["MS_SPLIT"]))["train"]


def get_features_per_chunk(stim_dict, resp_dict, chunks, embedder, ndelays=4, lag=LAG_FRAMES,  # noqa: F821
                           trim=TRIM_FRAMES):  # noqa: F821
    out = os.environ.get("RESCORE_OUT")
    if (out and os.environ.get("MS_SPLIT") and not _STATE["in_rescore"] and len(chunks) >= 20
            and list(chunks) == _frozen_train()[:len(chunks)]):
        from .eval import EncodingConfig
        from .rescore_stream import rescore
        _STATE["in_rescore"] = True
        print(f"[multisubj] intercepted training features: ndelays={ndelays} lag={lag} trim={trim}", flush=True)
        rescore(embedder, EncodingConfig(num_train=len(chunks), num_test=100, ndelays=ndelays, lag=lag,
                                         trim=trim), out)
        raise SystemExit(0)
    return _orig_get_features_per_chunk(stim_dict, resp_dict, chunks, embedder, ndelays=ndelays, lag=lag,
                                        trim=trim)


class _Guarded(types.ModuleType):
    def __setattr__(self, k, v):
        if k == "get_features_per_chunk" and os.environ.get("RESCORE_OUT") and v is not get_features_per_chunk:
            print("[multisubj] kept streamed get_features_per_chunk (ignored encoder's monkey-patch)", flush=True)
            return
        super().__setattr__(k, v)


sys.modules[__name__].__class__ = _Guarded
