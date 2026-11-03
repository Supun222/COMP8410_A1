import sys
import os 
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pair3_models import run_modelscope, run_animatediff

def load_video_prompts(prompts_file: Path) -> list:
  
    # incase json file is missing
    if not prompts_file.exists():
        print(f"[Warning] Prompts file not found at {prompts_file}")
        return []
        
    with open(prompts_file, "r", encoding="utf-8") as f:
        cases = json.load(f).get("pair3_video_generation", [])
            
    return cases

def plot_results(df: pd.DataFrame, results_dir: Path):

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cols = [str(c).lower() for c in df.columns]
    latency_col = df.columns[[i for i, c in enumerate(cols) if "latency" in c][0]] if any("latency" in c for c in cols) else "Latency_s"
    vram_col = df.columns[[i for i, c in enumerate(cols) if "vram" in c][0]] if any("vram" in c for c in cols) else "Peak_VRAM_GB"
    params_col = df.columns[[i for i, c in enumerate(cols) if "params" in c][0]] if any("params" in c for c in cols) else "Total_Params_M"

    available_cols = [c for c in [latency_col, vram_col, params_col] if c is not None and c in df.columns]
    avg_df = df.groupby("Model")[available_cols].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        (latency_col, "Inference Speed (Latency)\nLower is Better", "#1f77b4", "{:.2f}s"),
        (vram_col, "Memory (Peak VRAM)\nLower is Better", "#ff7f0e", "{:.2f} GB"),
        (params_col, "Total Parameters (M)\nModel Footprint", "#2ca02c", "{:.0f}M")
    ]

    # setting the plots structure
    for ax, (col, title, color, fmt) in zip(axes, metrics):
      if col not in avg_df.columns:
        ax.axis("off")
        continue

      bars = ax.bar(avg_df["Model"], avg_df[col], color=color, width=0.45)
      ax.set_title(title, fontweight="bold")
      for bar in bars:
        val = bar.get_height()
        ax.text(
          bar.get_x() + bar.get_width() / 2,
          val / 2,
          fmt.format(val),
          ha="center", va="center", color="white", fontweight="bold"
        )

    plt.tight_layout()
    plot_path = results_dir / "pair3_system_metrics.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {plot_path}")

def main():
    root = PROJECT_ROOT
    out_dir = root / "outputs" / "pair3_video_generation"
    results_dir = root / "results" / "pair3_video_generation"
    prompts_file = root / "data" / "prompts.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # load propmts from the json file
    cases = load_video_prompts(prompts_file)
    if not cases:
        print("[Error] No test cases found to evaluate. Exiting.")
        return

    # detecting device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # calling model execution function
    res_sys_a = run_modelscope(cases, out_dir, device=device)
    res_sys_b = run_animatediff(cases, out_dir, device=device)

    # collect System Metrics
    rows = []
    for case in cases:
      cid = case["id"]
      for model_name, res_dict in [
        ("ModelScopeT2V (System A)", res_sys_a),
        ("AnimateDiff (System B)", res_sys_b),
      ]:
        # getting values for metrics
        _, latency, vram, total_p = res_dict[cid]
            
        rows.append({
          "Test_Case": cid,
          "Prompt": case.get("prompt", "")[:30] + "...",  
          "Model": model_name,
          "Latency_s": latency,
          "Peak_VRAM_GB": vram,
          "Total_Params_M": total_p
        })

    # rendering charts
    df = pd.DataFrame(rows)
    plot_results(df, results_dir)

if __name__ == "__main__":
    main()