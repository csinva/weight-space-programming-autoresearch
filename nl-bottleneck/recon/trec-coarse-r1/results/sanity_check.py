"""Sanity check on hand-written questions (not dataset examples)."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import interpretable_transformer as it
import src.task as T

task = T.get_task("trec-coarse")
m = it.build_model(task)
print("patterns", len(it.PATTERNS), "params", sum(p.numel() for p in m.parameters()))
qs = ["what does nasa stand for", "who was the first president", "where is the eiffel tower",
      "when did the war end", "why is the sky blue", "what does ciao mean", "how did elvis die",
      "how many people live in china", "what is a cat", "what is the capital of france",
      "what color is the sky", "what is the fastest fish in the world", "name a famous painting",
      "what is the best selling album of all time", "what are the colors of the flag",
      "what is the tallest thing ever built by humans", "which is better"]
for q in qs:
    s = T._clean_text(q, set("abcdefghijklmnopqrstuvwxyz '"), 80) + "="
    ids = torch.tensor([task.encode(s)])
    print(task.decode(m(ids)[0, -1].argmax().view(1)), q)
