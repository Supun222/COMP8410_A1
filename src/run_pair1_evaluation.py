import sys
import json
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Determine Project Root (works whether this file is in root or inside src/)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent

# 2. Add src/ to sys.path so 'models' and 'evaluation' packages resolve
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pair1_models import run_instruct_pix2pix, run_ledits_pp
from evaluation.image_metrics import ImageEditingEvaluator


def ensure_sample_images(raw_dir: Path, cases: list):
    """Creates synthetic product test images if real ones aren't placed in data/raw/pair1_image_editing yet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(cases):
        img_path = raw_dir / case["filename"]
        if not img_path.exists():
            print(f"[Setup] Generating placeholder product image at {img_path}...")
            bg_color = [(235, 235, 240), (210, 180, 140), (245, 230, 215)][idx % 3]
            img = Image.new("RGB", (512, 512), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([156, 156, 356, 356], fill=(255, 255, 255), outline=(40, 40, 40), width=4)
            img.save(img_path)


def plot_results(df: pd.DataFrame, results_dir: Path):
    summary = df.groupby("Model")[["CLIP_Text_Sim", "CLIP_Dir_Sim", "SSIM", "LPIPS", "Latency_s"]].mean()
    ax = summary[["CLIP_Text_Sim", "CLIP_Dir_Sim", "SSIM", "LPIPS"]].plot(
        kind="bar", figsize=(10, 5), rot=0, colormap="viridis"
    )
    plt.title("Pair 1 Quantitative Comparison: InstructPix2Pix vs. LEDITS++")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    chart_path = results_dir / "pair1_metrics_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Results] Saved plot to {chart_path}")


def main():
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "pair1_image_editing"
    out_dir = root / "outputs" / "pair1_image_editing"
    results_dir = root / "results" / "pair1_image_editing"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = root / "data" / "prompts.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Expected prompts file at: {prompts_file}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        cases = json.load(f)["pair1_image_editing"]

    ensure_sample_images(raw_dir, cases)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Run model executions
    res_sys_a = run_instruct_pix2pix(cases, raw_dir, out_dir, device=device)
    res_sys_b = run_ledits_pp(cases, raw_dir, out_dir, device=device)

    # 2. Run quantitative metrics
    evaluator = ImageEditingEvaluator(device=device)
    rows = []

    for case in cases:
        cid = case["id"]
        for model_name, res_dict in [
            ("InstructPix2Pix (System A)", res_sys_a),
            ("LEDITS++ (System B)", res_sys_b),
        ]:
            orig_img, edit_img, latency, vram = res_dict[cid]
            metrics = evaluator.evaluate_pair(
                orig_img, edit_img, case["source_caption"], case["target_caption"]
            )
            rows.append({
                "Test_Case": cid,
                "Model": model_name,
                **metrics,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
            })

    df = pd.DataFrame(rows)
    csv_path = results_dir / "pair1_quantitative_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n================ PAIR 1 SUMMARY TABLE ================")
    print(df.groupby("Model").mean(numeric_only=True).to_string())
    plot_results(df, results_dir)


if __name__ == "__main__":
    main()