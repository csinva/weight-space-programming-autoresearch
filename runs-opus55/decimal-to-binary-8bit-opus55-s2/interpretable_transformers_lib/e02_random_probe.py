"""Snapshot of e01_random_baseline / e02_random_probe.

Both are the unmodified default architecture with random init. The only
differences from the template are the diagnostic prints in write_weights, which
report the task interface: prompt 'ddd=' (4 tokens), answer 8 bits MSB first,
vocab digits + '='. Neither attempt writes any weights; accuracy 0/200.
"""
