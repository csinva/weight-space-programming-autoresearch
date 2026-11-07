import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify InterpretableEmbedder
emb_old = """    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

emb_new = """    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        
        # Extract the final time step
        final_state = hidden_states[:, -1, :]
        
        # Lateral Inhibition (Winner-Take-All across the 15 timescales)
        # The 15 networks occupy dimensions 0 to 960 (64 dims each)
        networks = final_state[:, :960].view(B, 15, 64)
        
        # Compute magnitude of each network
        mags = networks.norm(dim=2) # (B, 15)
        
        # Find the threshold to keep only the Top 3 most active timescales per phrase
        # (Lateral inhibition: strongest timescales inhibit the weaker ones)
        k = 3
        topk_mags, _ = torch.topk(mags, k, dim=1)
        threshold = topk_mags[:, -1].unsqueeze(1)
        
        # Mask out networks below threshold
        mask = (mags >= threshold).unsqueeze(2).float()
        inhibited_networks = networks * mask
        
        # Reconstruct the final state
        new_final_state = final_state.clone()
        new_final_state[:, :960] = inhibited_networks.view(B, 960)
        
        return new_final_state.cpu()"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Lateral_Inhibition_WTA"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Lateral Inhibition Hypothesis (Winner-Take-All): Models inhibitory interneurons by enforcing strict sparsity across the 15 temporal networks. For any given input phrase, only the top 3 most resonant timescales are allowed to output features; the other 12 are completely inhibited."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Lateral Inhibition.")
