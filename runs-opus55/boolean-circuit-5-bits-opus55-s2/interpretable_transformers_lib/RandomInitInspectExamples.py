"""
Reconstructed snapshot of the second diagnostic run: identical to RandomInitBaseline.py (the
default random-init transformer) except for write_weights(), shown here, which printed sample
examples via the task's public generate_examples() to learn the I/O format.

    def write_weights(model, task):
        torch.manual_seed(0)
        print("TASK type:", type(task))
        print("TASK public attrs:", [a for a in dir(task) if not a.startswith("_")])
        print("TASK sizes:", task.name, task.prompt_len, task.answer_len, task.seq_len,
              task.vocab_size, task.vocab)
        import inspect
        print("generate_examples sig:", inspect.signature(task.generate_examples))
        print("is_correct sig:", inspect.signature(task.is_correct))
        print("encode sig:", inspect.signature(task.encode), "decode sig:", inspect.signature(task.decode))
        try:
            ex = task.generate_examples(40, seed=123)
        except TypeError:
            ex = task.generate_examples(40)
        print("example type:", type(ex))
        for e in list(ex)[:40]:
            print(repr(e))
        return

    model_shorthand_name = "RandomInitInspectExamples"
"""
