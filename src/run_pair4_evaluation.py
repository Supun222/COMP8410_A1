import sys
import os 
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup Project Paths
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from models.models_pair4 import run_ip2p_frame_by_frame, run_animatediff_v2v

def load_video_cases(raw_dir: Path, prompts_file: Path) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not prompts_file.exists(): return []
        
    with open(prompts_file, "r", encoding="utf-8") as f:
        cases = json.load(f).get("pair4_video_editing", [])
            
    return [c for c in cases if (raw_dir / c["filename"]).exists()]

def plot_results(df: pd.DataFrame, results_dir: Path):
    if df.empty: return
    results_dir.mkdir(parents=True, exist_ok=True)

    cols = [str(c).lower() for c in df.columns]
    latency_col = df.columns[[i for i, c in enumerate(cols) if "latency" in c][0]]
    vram_col = df.columns[[i for i, c in enumerate(cols) if "vram" in c][0]]
    params_col = df.columns[[i for i, c in enumerate(cols) if "params" in c][0]]

    avg_df = df.groupby("Model")[[latency_col, vram_col, params_col]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        (latency_col, "Inference Speed (Latency)\nLower is Better", "#1f77b4", "{:.2f}s"),
        (vram_col, "Memory (Peak VRAM)\nLower is Better", "#ff7f0e", "{:.2f} GB"),
        (params_col, "Total Parameters (M)\nModel Footprint", "#2ca02c", "{:.0f}M")
    ]

    for ax, (col, title, color, fmt) in zip(axes, metrics):
        bars = ax.bar(avg_df["Model"], avg_df[col], color=color, width=0.45)
        ax.set_title(title, fontweight="bold")
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2, fmt.format(val),
                    ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plot_path = results_dir / "pair4_system_metrics.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {plot_path}")

def main():
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "pair4_video_editing"
    out_dir = root / "outputs" / "pair4_video_editing"
    results_dir = root / "results" / "pair4_video_editing"
    prompts_file = root / "data" / "prompts.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_video_cases(raw_dir, prompts_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # calling model functions 
    res_sys_a = run_ip2p_frame_by_frame(cases, raw_dir, out_dir, device=device)
    res_sys_b = run_animatediff_v2v(cases, raw_dir, out_dir, device=device)

    rows = []
    for case in cases:
        cid = case["id"]
        for model_name, res_dict in [
            ("IP2P Baseline (System A)", res_sys_a),
            ("AnimateDiff V2V (System B)", res_sys_b),
        ]:
            if cid not in res_dict: continue
            _, latency, vram, total_p = res_dict[cid]
            rows.append({
                "Test_Case": cid,
                "Model": model_name,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
                "Total_Params_M": total_p
            })

    # calling to plot functions
    df = pd.DataFrame(rows)
    plot_results(df, results_dir)

if __name__ == "__main__":
    main()