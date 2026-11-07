import os
import sys
import shutil

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

for seed in range(0, 10):
    if seed == 42:
        continue
        
    os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)
    
    with open(filepath, "r") as f:
        content = f.read()

    # Replace seed
    content = content.replace("torch.manual_seed(42)", f"torch.manual_seed({seed})")
    
    name = f"Deep_Ensemble_Seed_{seed}"
    content = content.replace("Deep_Ensemble_0421_Master", name)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"\n--- Running {name} ---")
    sys.stdout.flush()
    
    os.system("uv run python runs-neuro/fmri-jun03-run3/interpretable_transformer.py")

