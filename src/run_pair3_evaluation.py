import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the model functions from the separate models file
from models_pair3 import run_modelscope, run_animatediff

def plot_results(df: pd.DataFrame, results_dir: Path):
    """Generates simple bar plots comparing Latency, VRAM, and Model Size."""
    results_dir.mkdir(parents=True, exist_ok=True)
    avg_df = df.groupby("Model")[["Latency_s", "Peak_VRAM_GB", "Total_Params_M"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    # 1. Latency Plot
    bars0 = axes[0].bar(avg_df["Model"], avg_df["Latency_s"], color="#1f77b4", width=0.45)
    axes[0].set_title("Inference Latency (s)\nLower is Better", fontweight="bold")
    axes[0].set_ylabel("Seconds")
    
    # 2. Peak VRAM Plot
    bars1 = axes[1].bar(avg_df["Model"], avg_df["Peak_VRAM_GB"], color="#ff7f0e", width=0.45)
    axes[1].set_title("Peak VRAM Usage (GB)\nLower is Better", fontweight="bold")
    axes[1].set_ylabel("GB")
    
    # 3. Total Parameters Plot
    bars2 = axes[2].bar(avg_df["Model"], avg_df["Total_Params_M"], color="#2ca02c", width=0.45)
    axes[2].set_title("Total Parameters (M)\nModel Footprint", fontweight="bold")
    axes[2].set_ylabel("Million Parameters")

    # Add numeric labels to bars
    for ax, bars, fmt in zip(axes, [bars0, bars1, bars2], ["{:.1f}s", "{:.1f} GB", "{:.0f}M"]):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2, fmt.format(val), 
                    ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plot_path = results_dir / "pair3_system_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Plot Saved] {plot_path}")

def main():
    data_dir = Path("data")
    out_dir = Path("outputs/pair3_video_generation")
    results_dir = Path("results/pair3_video_generation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_file = data_dir / "prompts_pair3.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found at: {prompts_file}")

    with open(prompts_file, "r") as f:
        cases = json.load(f).get("pair3_video_generation", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Execution] Running Pair 3 Evaluation on device: {device.upper()}")
    
    # Execute Model Pipelines
    res_sys_a = run_modelscope(cases, out_dir, device)
    res_sys_b = run_animatediff(cases, out_dir, device)

    # 5. Collect System Metrics (Aligned with previous models)
    rows = []
    for case in cases:
        cid = case["id"]
        for model_name, res_dict in [
            ("ModelScopeT2V (System A)", res_sys_a),
            ("AnimateDiff (System B)", res_sys_b),
        ]:
            _, latency, vram, total_p = res_dict[cid]
            rows.append({
                "Test_Case": cid,
                "Image_File": case.get("filename", ""),
                "Model": model_name,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
                "Total_Params_M": total_p
            })

    df = pd.DataFrame(rows)
    
    # Print Summary Table
    print("\n" + "=" * 50)
    print("      PAIR 3 VIDEO GENERATION BENCHMARK SUMMARY")
    print("=" * 50)
    print(df.groupby("Model")[["Latency_s", "Peak_VRAM_GB", "Total_Params_M"]].mean().to_string())
    print("=" * 50)

    # Render Plot
    plot_results(df, results_dir)

if __name__ == "__main__":
    main()