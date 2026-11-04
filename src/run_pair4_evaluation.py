import sys
import os 
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers.utils import export_to_video

# Setup Project Paths
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pair4_models import run_ip2p_frame_by_frame, run_animatediff_v2v

def load_video_cases(raw_dir: Path, prompts_file: Path) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not prompts_file.exists():
        return []
        
    with open(prompts_file, "r", encoding="utf-8") as f:
        cases = json.load(f).get("pair4_video_editing", [])
            
    return [c for c in cases if (raw_dir / c["filename"]).exists()]

def plot_results(df: pd.DataFrame, results_dir: Path):
    if df.empty:
        return
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

def save_and_record_results(model_name, prefix, res_dict, cases, out_dir, fps=8):
    records = []
    for case in cases:
        cid = case["id"]
        if cid not in res_dict:
            continue
        
        frames, latency, vram, total_p = res_dict[cid]
        out_path = out_dir / f"{prefix}_{cid}.mp4"
        
        # Export video in the evaluation driver
        export_to_video(frames, str(out_path), fps=fps)
        print(f"[{model_name}] Saved output video: {out_path}")
        
        records.append({
            "Test_Case": cid,
            "Model": model_name,
            "Latency_s": latency,
            "Peak_VRAM_GB": vram,
            "Total_Params_M": total_p
        })
    return records

def main():
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "pair4_video_editing"
    out_dir = root / "outputs" / "pair4_video_editing"
    results_dir = root / "results" / "pair4_video_editing"
    prompts_file = root / "data" / "prompts.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_video_cases(raw_dir, prompts_file)
    if not cases:
        print(f"[Warning] No valid test video cases found matching {raw_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Run pure model inferences
    print("\n--- Running IP2P Baseline (System A) ---")
    res_sys_a = run_ip2p_frame_by_frame(cases, raw_dir, device=device)

    print("\n--- Running AnimateDiff V2V (System B) ---")
    res_sys_b = run_animatediff_v2v(cases, raw_dir, device=device)

    # 2. Export videos and record system metrics
    rows = []
    rows.extend(save_and_record_results("IP2P Baseline (System A)", "sys_a", res_sys_a, cases, out_dir))
    rows.extend(save_and_record_results("AnimateDiff V2V (System B)", "sys_b", res_sys_b, cases, out_dir))

    # 3. Aggregate metrics, save CSV, and plot
    df = pd.DataFrame(rows)
    metrics_csv = results_dir / "pair4_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[Metrics Saved] {metrics_csv}")

    plot_results(df, results_dir)

if __name__ == "__main__":
    main()