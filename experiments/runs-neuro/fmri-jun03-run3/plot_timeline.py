import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "results/overall_results.csv"

def plot_timeline():
    df = pd.read_csv(RESULTS_FILE)
    df = df[df["subject"] == "UTS03"].reset_index(drop=True)
    
    corrs = df["test_corr"].values
    names = df["model_shorthand_name"].values
    
    running_max = -np.inf
    max_corrs = []
    max_names = []
    max_iters = []
    
    for i, c in enumerate(corrs):
        if c > running_max:
            running_max = c
            max_corrs.append(c)
            max_names.append(names[i])
            max_iters.append(i)
            
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(corrs)), corrs, alpha=0.3, color="gray", label="All Runs")
    plt.plot(max_iters, max_corrs, 'o-', color="red", label="SOTA Frontier")
    
    for x, y, name in zip(max_iters, max_corrs, max_names):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, rotation=45)
        
    plt.title("UTS03 Linear fMRI Encoding Optimization Timeline")
    plt.xlabel("Experiment Index")
    plt.ylabel("Test Correlation (Pearson r)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/timeline.pdf")
    print("Timeline plotted.")

if __name__ == "__main__":
    import numpy as np
    plot_timeline()
