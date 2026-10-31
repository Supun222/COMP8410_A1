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
    Scans data/raw/pair1_image_editing for actual .jpg/.png files.
    Matches found files with case metadata from prompts.json or applies culinary fallback prompts.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    existing_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in valid_extensions])
    
    if not existing_files:
        raise FileNotFoundError(
            f"\n[Error] No image files found in: {raw_dir}\n"
            f"Please drop your real .jpg or .png images into that directory before running."
        )

    print(f"\n[Input Verification] Detected {len(existing_files)} raw image(s) in {raw_dir}:")
    for f in existing_files:
        print(f"  • {f.name}")

    case_lookup = {c["filename"]: c for c in cases_json if "filename" in c}
    active_cases = []

    for idx, img_file in enumerate(existing_files):
        filename = img_file.name
        if filename in case_lookup:
            active_cases.append(case_lookup[filename])
        else:
            # Domain-aligned fallback configuration for manually uploaded dishes
            dish_name = img_file.stem.replace("_", " ").title()
            active_cases.append({
                "id": f"case_{idx+1:02d}",
                "filename": filename,
                "source_caption": f"a culinary prep photo of {dish_name}",
                "target_caption": f"gourmet plated {dish_name} on a dark rustic oak dining table in fine dining restaurant setting",
                "instruction": f"Place the plated {dish_name} onto a dark rustic oak dining table, fine dining restaurant setting, soft warm bokeh lighting",
                "concepts": ["dark rustic oak dining table", "fine dining restaurant setting"],
                "seed": 42 + idx
            })

    return active_cases


def plot_results(df: pd.DataFrame, results_dir: Path):
    """Generates and saves comparative quantitative metric charts."""
    summary = df.groupby("Model")[["CLIP_Text_Sim", "CLIP_Dir_Sim", "SSIM", "LPIPS"]].mean()
    
    ax = summary.plot(
        kind="bar", 
        figsize=(10, 5), 
        rot=0, 
        colormap="viridis"
    )
    plt.title("Pair 1 Quantitative Comparison: InstructPix2Pix vs. LEDITS++")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    chart_path = results_dir / "pair1_metrics_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Results] Saved comparative plot to: {chart_path}")


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

    # 2. Discover manually added raw images (no synthetic placeholder boxes drawn)
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
                "Image_File": case["filename"],
                "Model": model_name,
                **metrics,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
            })

    # 6. Save and Display Tabular Results
    df = pd.DataFrame(rows)
    csv_path = results_dir / "pair1_quantitative_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Results] Exported full metric records to: {csv_path}")

    print("\n" + "=" * 50)
    print("           PAIR 1 SUMMARY BENCHMARK TABLE")
    print("=" * 50)
    print(df.groupby("Model").mean(numeric_only=True).to_string())
    print("=" * 50)

    # 7. Render Charts
    plot_results(df, results_dir)


if __name__ == "__main__":
    main()