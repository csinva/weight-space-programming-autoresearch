# Task: describe an acoustic feature space in at most 200 words

An encoding-model study predicts human intracranial brain activity (ECoG high-gamma power, 30 Hz) from the sound a patient hears, using a fixed acoustic feature space and a linear readout. `/work/model.py` implements that feature space: it is the object returned by
`build_embedder(device, cond_mode='none', interact_set='v3e')`, whose `__call__` maps audio windows to
features. Code paths that this configuration does not use are not part of it. `/work/INTERFACE.md` explains
how the feature space is used.

Write a description of this feature space in **at most 200 words**, such that an expert given only your
description and INTERFACE.md could rebuild it. Save it as `/work/DESCRIPTION.md` and check its length with
`wc -w /work/DESCRIPTION.md` (at most 200). You may read and run the code; example audio is in
`/work/examples/` (no brain data). Work only inside `/work`.
