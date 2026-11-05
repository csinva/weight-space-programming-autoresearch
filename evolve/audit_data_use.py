"""Counterfactual check: do a run's weights depend on task data?

Each saved program is imported and build_model(task) is run three times: (A) normally, (A2) normally again (to confirm the build is
deterministic), and (B) with the task data altered: the example generator draws from a different seed with answers shuffled across
examples, and for the real-world tasks the dataset pool is rebuilt from the second half of the training split
with its labels shuffled; the evaluation function returns a constant.
  data_independent : A == A2 == B, so any use of task data was only a check (e.g. a printed self-test)
  data_dependent   : A == A2 but B differs, so the weights are computed from or chosen using task examples
  nondeterministic : A != A2 (cannot tell)
  error            : the program failed to build

  uv run evolve/audit_data_use.py runs-opus55      -> runs-opus55/audit_data_use.json"""
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


def check(args):
    run_dir, path = args
    sys.path.insert(0, run_dir)
    os.chdir(run_dir)
    import src.task as T
    import src.eval as E
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
        def altered(self, n, seed=None):
            ex = orig_gen(self, n, seed=(seed or 0) + 7919)
            ans = [e.answer for e in ex]
            random.Random(1).shuffle(ans)
            return [type(e)(prompt=e.prompt, answer=x) for e, x in zip(ex, ans)]
        cls.generate_examples = altered
        if hasattr(cls, "_load_pool"):  # real-world tasks: the second half of the training split, with labels shuffled
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
            labels = [e[-1] for e in pool]
            random.Random(2).shuffle(labels)
            cls._pool = [tuple(e[:-1]) + (l,) for e, l in zip(pool, labels)]
        E.evaluate = lambda *x, **k: (1.0, [])
        if hasattr(mod, "evaluate"):
            mod.evaluate = E.evaluate
        b = weights(mod, task)
        return path, "data_independent" if same(a, b) else "data_dependent"
    except Exception as e:
        return path, "error: " + type(e).__name__ + ": " + str(e)[:120]


def main(root):
    jobs = []
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        lib = os.path.join(rd, "interpretable_transformers_lib")
        if os.path.isdir(lib):
            jobs += [(rd, os.path.join(lib, f)) for f in sorted(os.listdir(lib)) if f.endswith(".py")]
    out = {}
    with ProcessPoolExecutor(32) as ex:
        for path, res in ex.map(check, jobs):
            out.setdefault(path.split(os.sep)[-3], {})[os.path.basename(path)[:-3]] = res
    kinds = ["data_independent", "data_dependent", "nondeterministic", "error"]
    summary = {r: {k: sum(1 for v in d.values() if v.startswith(k)) for k in kinds} for r, d in out.items()}
    for r, s in summary.items():
        print(r, s)
    json.dump(dict(summary=summary, per_snapshot=out), open(os.path.join(root, "audit_data_use.json"), "w"), indent=1)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv[1] if len(sys.argv) > 1 else "runs-opus55"))
