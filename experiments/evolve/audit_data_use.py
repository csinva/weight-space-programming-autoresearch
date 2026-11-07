"""Counterfactual check: do a run's weights depend on task data, and on the labels in particular?

Each saved program is imported and build_model(task) is run four times: (A) normally, (A2) normally again (to confirm the build is
deterministic), (C) with the same inputs but random answers (same seeds; each answer replaced by a random string of the same length; for the
dataset tasks the same pool with random labels), and (B) with different inputs as well (a different seed with random answers; for the dataset tasks the second half of the training
split with random labels). The evaluation function returns a constant in C and B.
  data_independent : A == A2 == B == C, so any use of task data was only a check (e.g. a printed self-test)
  label_dependent  : C differs from A: the weights depend on the answers, i.e. they were fitted to labels or chosen by accuracy on them
  input_dependent  : C equals A but B differs: the weights depend on example inputs only (e.g. a LayerNorm gain calibrated on sample prompts)
  nondeterministic : A != A2 (cannot tell)
  error            : the program failed to build

  uv run experiments/evolve/audit_data_use.py runs-opus55      -> runs-opus55/audit_data_use.json  (WSP_LIB=reconstructed for rebuilt programs)"""
import contextlib, importlib.util, io, json, os, random, re, sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")


def weights(mod, task):
    with contextlib.redirect_stdout(io.StringIO()):
        m = mod.build_model(task)
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def same(a, b):
    import torch
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)


_ORIG = {}


def restore(T, E):
    """Undo every patch from earlier checks in this worker process (workers are reused across programs)."""
    if not _ORIG:
        _ORIG["evaluate"] = E.evaluate
        _ORIG["tasks"] = {c: (c.__dict__.get("generate_examples"), c.__dict__.get("_load_pool")) for c in T.TASK_REGISTRY.values()}
    E.evaluate = _ORIG["evaluate"]
    for c, (gen, lp) in _ORIG["tasks"].items():
        if gen is not None:
            c.generate_examples = gen
        if hasattr(c, "_pool"):
            c._pool = None


def check(args):
    run_dir, path = args
    sys.path.insert(0, run_dir)
    os.chdir(run_dir)
    import src.task as T
    import src.eval as E
    restore(T, E)
    try:
        return _check(run_dir, path, T, E)
    finally:
        restore(T, E)
        sys.path.remove(run_dir)


def _check(run_dir, path, T, E):
    task_name = re.sub(r"-(opus|sonnet|haiku)55-s\d+$", "", os.path.basename(run_dir.rstrip("/")))
    try:
        spec = importlib.util.spec_from_file_location("snap_" + os.path.basename(path)[:-3].replace("-", "_").replace(".", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        task = T.get_task(task_name)
        a = weights(mod, task)
        a2 = weights(mod, task)
        if not same(a, a2):
            return path, "nondeterministic"
        cls = type(task)
        orig_gen = cls.generate_examples
        const_eval = lambda *x, **k: (1.0, [])
        def shuffled(ex, seed):
            """Random relabelling: each answer becomes a random string of the same length over the answer characters seen
            (so label totals change too, unlike a shuffle)."""
            rng = random.Random(seed)
            chars = sorted({ch for e in ex for ch in e.answer}) or list("0")
            return [type(e)(prompt=e.prompt, answer="".join(rng.choice(chars) for _ in e.answer)) for e in ex]
        E.evaluate = const_eval
        if hasattr(mod, "evaluate"):
            mod.evaluate = const_eval
        # C: same inputs, answers shuffled
        cls.generate_examples = lambda self, n, seed=None: shuffled(orig_gen(self, n, seed=seed), 3)
        if hasattr(cls, "_load_pool"):
            pool0 = list(cls._load_pool())
            classes = sorted({e[-1] for e in pool0})
            rng = random.Random(4)
            cls._pool = [tuple(e[:-1]) + (rng.choice(classes),) for e in pool0]
        c = weights(mod, task)
        if not same(a, c):
            return path, "label_dependent"
        # B: different inputs as well
        cls.generate_examples = lambda self, n, seed=None: shuffled(orig_gen(self, n, seed=(seed or 0) + 7919), 1)
        if hasattr(cls, "_load_pool"):  # the second half of the training split, with labels shuffled
            import datasets
            orig_load = datasets.load_dataset
            def second_half(*x, **k):
                ds = orig_load(*x, **k)
                return ds.select(range(len(ds) // 2, len(ds)))
            datasets.load_dataset = second_half
            try:
                cls._pool = None
                pool = cls._load_pool()
            finally:
                datasets.load_dataset = orig_load
            classes = sorted({e[-1] for e in pool})
            rng = random.Random(2)
            cls._pool = [tuple(e[:-1]) + (rng.choice(classes),) for e in pool]
        b = weights(mod, task)
        return path, "data_independent" if same(a, b) else "input_dependent"
    except Exception as e:
        return path, "error: " + type(e).__name__ + ": " + str(e)[:120]


def main(root):
    jobs = []
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        lib = os.path.join(rd, os.environ.get("WSP_LIB", "interpretable_transformers_lib"))  # or "reconstructed" (from transcripts)
        if os.path.isdir(lib):
            jobs += [(rd, os.path.join(lib, f)) for f in sorted(os.listdir(lib)) if f.endswith(".py")]
    out = {}
    with ProcessPoolExecutor(32) as ex:
        for path, res in ex.map(check, jobs):
            out.setdefault(path.split(os.sep)[-3], {})[os.path.basename(path)[:-3]] = res
    kinds = ["data_independent", "label_dependent", "input_dependent", "nondeterministic", "error"]
    summary = {r: {k: sum(1 for v in d.values() if v.startswith(k)) for k in kinds} for r, d in out.items()}
    for r, s in summary.items():
        print(r, s)
    json.dump(dict(summary=summary, per_snapshot=out), open(os.path.join(root, "audit_data_use" + ("_reconstructed" if os.environ.get("WSP_LIB") == "reconstructed" else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv[1] if len(sys.argv) > 1 else "runs-opus55"))
