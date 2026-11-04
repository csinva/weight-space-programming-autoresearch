"""
e01_probe_task_interface snapshot: stock interpretable_transformer.py (default architecture, random init)
with one change, a write_weights() that only logs task attributes and evaluator forward calls.
The unchanged stock architecture and evaluation harness are as in the e02 snapshot's main block.
"""

def write_weights(model, task) -> None:
    import torch
    torch.manual_seed(0)
    print("TASK TYPE:", type(task).__name__)
    for name in dir(task):
        if name.startswith("__"):
            continue
        val = getattr(task, name)
        print("TASK ATTR:", name, "=", repr(val)[:300])
    calls = {"n": 0}
    def probe(module, inputs):
        calls["n"] += 1
        if calls["n"] <= 40:
            ids = inputs[0]
            print("FORWARD CALL:", tuple(ids.shape), ids[0].tolist())
    model.register_forward_pre_hook(probe)
    return


model_shorthand_name = "e01_probe_task_interface"
model_description = "Random init baseline; logs public task attributes and evaluator forward calls to learn the task format."
