"""Write every tensor of Qwen3.5-0.8B (text part) as plain text for the "Read all weights" condition.
One file per tensor: line 1 is a header; then each row of the (2D-reshaped) tensor takes ceil(cols/16) lines of 16 values (4 decimals),
so row r starts at line 2 + r * ceil(cols/16). INDEX.md lists every tensor, its shape, file and lines per row.
  uv run python systematic-tests/dump_all_weights.py OUT_DIR"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def dump(args):
    name, arr, out = args
    a = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    cols = a.shape[1]
    per = 16 if cols % 16 == 0 else cols
    path = os.path.join(out, name + ".txt")
    with open(path, "w") as f:
        f.write(f"# {name} shape {tuple(arr.shape)} -> rows {a.shape[0]} x cols {cols}; {per} values per line, {cols // per} lines per row; row r starts at line 2 + r*{cols // per}\n")
        flat = a.reshape(-1, per)
        for s in range(0, flat.shape[0], 200000):
            np.savetxt(f, flat[s:s + 200000], fmt="%+.4f")
    return name, tuple(arr.shape), a.shape[0], cols // per


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    from common import load
    m, _ = load()
    jobs = [(n, p.detach().float().numpy(), out) for n, p in m.named_parameters()]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 32))) as ex:
        rows = list(ex.map(dump, jobs))
    with open(os.path.join(out, "INDEX.md"), "w") as f:
        f.write("# All weights of Qwen3.5-0.8B (text part), unedited. One file per tensor.\n"
                "Each file: line 1 header; then each row of the tensor (2D view: first dim = rows) takes the stated number of lines of 16 values.\n"
                "Row r of a tensor starts at line 2 + r * (lines per row). Use the Read tool with offset/limit to read specific rows.\n"
                "Note: model.embed_tokens.weight is also the output unembedding (tied). RMSNorm gains are stored as w (gain = 1 + w).\n\n"
                "| tensor | shape | rows | lines per row | file |\n|---|---|---|---|---|\n")
        for n, sh, r, lpr in rows:
            f.write(f"| {n} | {sh} | {r} | {lpr} | {n}.txt |\n")
    print("wrote", len(rows), "tensors to", out)
