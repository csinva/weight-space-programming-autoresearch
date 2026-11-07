import sys
with open('interpretable_transformer.py', 'r') as f:
    lines = f.readlines()

out = []
in_eval = False
for line in lines:
    if "if __name__ == \"__main__\":" in line:
        in_eval = True
        out.append(line)
        out.append("""    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = build_embedder(device=args.device)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    t0 = time.time()
    try:
        results = run_encoding(embedder, config)
        test_corr = results["test_corr"]
        print(f"Mean test correlation: {test_corr:.4f}")
        
        n_params = sum(p.numel() for p in embedder.model.parameters())
        row = make_result_row(results, model_shorthand_name, n_params, model_description, "success")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during evaluation: {e}")
        # dummy row on failure
        row = {
            "subject": args.subject,
            "test_corr": 0.0, "train_corr": 0.0, "frac_test_voxels_above_0.2": 0.0,
            "encoding_seconds": time.time() - t0,
            "status": "error", "model_shorthand_name": model_shorthand_name,
            "n_params": sum(p.numel() for p in embedder.model.parameters()),
            "description": model_description,
            "corrs_test_frac>0.1": 0.0, "corrs_test_frac>0.05": 0.0, "corrs_test_frac>0.0": 0.0,
            "corrs_test_median": 0.0, "corrs_test_p75": 0.0, "corrs_test_p90": 0.0, "corrs_test_p95": 0.0, "corrs_test_p99": 0.0
        }

    upsert_overall_results(RESULTS_DIR, row)
""")
        break
    else:
        out.append(line)

with open('interpretable_transformer.py', 'w') as f:
    f.writelines(out)
