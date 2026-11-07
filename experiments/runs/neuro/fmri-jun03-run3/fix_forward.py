import sys
with open('interpretable_transformer.py', 'r') as f:
    text = f.read()

text = text.replace("""    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Returns the final-token hidden state for each sequence.
        Expected input_ids shape: (B, T).
        Output shape: (B, d_model).
        \"\"\"
        input_ids = input_ids.to(self.device)
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()""", """    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        \"\"\"
        Expected input_strings: list of strings (length B).
        Output shape: (B, d_model).
        \"\"\"
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):  # Take last T chars
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()""")

# Fix eval results format
text = text.replace("row = {", "row = make_result_row({")
text = text.replace("            \"corrs_test_median\": 0.0, \"corrs_test_p75\": 0.0, \"corrs_test_p90\": 0.0, \"corrs_test_p95\": 0.0, \"corrs_test_p99\": 0.0\n        }", "            \"corrs_test_median\": 0.0, \"corrs_test_p75\": 0.0, \"corrs_test_p90\": 0.0, \"corrs_test_p95\": 0.0, \"corrs_test_p99\": 0.0\n        }, model_shorthand_name, sum(p.numel() for p in embedder.model.parameters()), model_description, \"error\")")

with open('interpretable_transformer.py', 'w') as f:
    f.write(text)
