import sys
import os
import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Hugging Face environment optimizations
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Setup Project Paths
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import models and evaluation metrics
from models.pair1_models import run_instruct_pix2pix, run_ledits_pp
from evaluation.image_metrics import ImageEditingEvaluator

def load_existing_raw_images(raw_dir: Path, cases_json: list) -> list:
  """
  Scans raw_dir for image files and returns only:
  'id', 'filename', 'image_path', and 'prompt'.
  """
  raw_dir.mkdir(parents=True, exist_ok=True)
  valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
  existing_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in valid_extensions])

  # Map by filename directly from the input json
  case_lookup = {c["filename"]: c for c in cases_json if "filename" in c}
  active_cases = []

  for idx, img_file in enumerate(existing_files):
    filename = img_file.name
    meta = case_lookup.get(filename, {})

    active_cases.append({
      "id": meta.get("id", f"case_{idx+1:02d}"),
      "filename": filename,
      "image_path": str(img_file),
      "prompt": meta.get("prompt", "")
    })

  return active_cases

def plot_results(df: pd.DataFrame, results_dir: Path):
    if df.empty:
        print("[Error] DataFrame is empty. No metric rows were generated to plot.")
        return

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cols = [str(c).lower() for c in df.columns]
    
    clip_col = df.columns[[i for i, c in enumerate(cols) if "clip" in c][0]] if any("clip" in c for c in cols) else None
    lpips_col = df.columns[[i for i, c in enumerate(cols) if "lpips" in c][0]] if any("lpips" in c for c in cols) else None
    latency_col = df.columns[[i for i, c in enumerate(cols) if "latency" in c][0]] if any("latency" in c for c in cols) else "Latency_s"
    vram_col = df.columns[[i for i, c in enumerate(cols) if "vram" in c][0]] if any("vram" in c for c in cols) else "Peak_VRAM_GB"

    available_cols = [c for c in [clip_col, lpips_col, latency_col, vram_col] if c is not None and c in df.columns]
    
    if not available_cols:
        print("[Error] Could not find metric columns to plot. Available:", df.columns.tolist())
        return

    avg_df = df.groupby("Model")[available_cols].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    metrics = [
        (clip_col, "Edit Control (CLIP Score)\nHigher is Better", "#2ca02c", "{:.3f}"),
        (lpips_col, "Identity Preservation (LPIPS)\nLower is Better", "#d62728", "{:.3f}"),
        (latency_col, "Inference Speed (Latency)\nLower is Better", "#1f77b4", "{:.1f}s"),
        (vram_col, "Memory (Peak VRAM)\nLower is Better", "#ff7f0e", "{:.1f} GB")
    ]

    for ax, (col, title, color, fmt) in zip(axes, metrics):
        if col is None or col not in avg_df.columns:
            ax.axis('off')
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
    plot_path = results_dir / "pair1_4_core_metrics.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {plot_path}")

def main():
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "pair1_image_editing"
    out_dir = root / "outputs" / "pair1_image_editing"
    results_dir = root / "results" / "pair1_image_editing"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Prompts JSON if available
    prompts_file = root / "data" / "prompts.json"
    cases_from_json = []
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            cases_from_json = json.load(f).get("pair1_image_editing", [])

    # 2. Discover manually added raw images
    cases = load_existing_raw_images(raw_dir, cases_from_json)

    # 3. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Execution] Running evaluation on compute device: {device.upper()}")

    # 4. Run Model Executions
    res_sys_a = run_instruct_pix2pix(cases, raw_dir, out_dir, device=device)
    res_sys_b = run_ledits_pp(cases, raw_dir, out_dir, device=device)

    # 5. Evaluate Quantitative Metrics
    print("\n[Evaluation] Calculating CLIP text-image similarity, directional similarity, SSIM, and LPIPS...")
    evaluator = ImageEditingEvaluator(device=device)
    rows = []

    # --- RESTORED EVALUATION LOOP ---
    for case in cases:
        cid = case["id"]
        source_cap = case.get("source_caption", "")
        target_cap = case.get("prompt", case.get("instruction", case.get("target_caption", "")))

        for model_name, res_dict in [
            ("InstructPix2Pix (System A)", res_sys_a),
            ("LEDITS++ (System B)", res_sys_b),
        ]:
            orig_img, edit_img, latency, vram = res_dict[cid]
            metrics = evaluator.evaluate_pair(orig_img, edit_img, source_cap, target_cap)
            rows.append({
                "Test_Case": cid,
                "Image_File": case.get("filename", ""),
                "Model": model_name,
                **metrics,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
            })

    # 6. Build DataFrame and Render Results
    df = pd.DataFrame(rows)
    plot_results(df, results_dir)


if __name__ == "__main__":
    main()