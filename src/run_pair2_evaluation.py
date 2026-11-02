import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pair2_model import run_triposr, run_shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_existing_raw_images(raw_dir: Path, cases_json: list) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    existing_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in valid_extensions])

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
    latency_col = df.columns[[i for i, c in enumerate(cols) if "latency" in c][0]] if any("latency" in c for c in cols) else "Latency_s"
    vram_col = df.columns[[i for i, c in enumerate(cols) if "vram" in c][0]] if any("vram" in c for c in cols) else "Peak_VRAM_GB"

    available_cols = [c for c in [latency_col, vram_col] if c is not None and c in df.columns]
    avg_df = df.groupby("Model")[available_cols].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    metrics = [
        (latency_col, "Inference Speed (Latency)\nLower is Better", "#1f77b4", "{:.2f}s"),
        (vram_col, "Memory (Peak VRAM)\nLower is Better", "#ff7f0e", "{:.2f} GB")
    ]

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
    plot_path = results_dir / "pair2_system_metrics.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {plot_path}")

def main():
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "pair2_3d_generation"
    out_dir = root / "outputs" / "pair2_3d_generation"
    results_dir = root / "results" / "pair2_3d_generation"

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Prompts JSON if available
    prompts_file = root / "data" / "prompts.json"
    cases_from_json = []
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            cases_from_json = json.load(f).get("pair2_3d_generation", [])

    # 2. Discover raw images
    cases = load_existing_raw_images(raw_dir, cases_from_json)

    # 3. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Execution] Running Pair 2 3D evaluation on: {device.upper()}")

    # 4. Run Model Executions
    res_sys_a = run_triposr(cases, raw_dir, out_dir, device=device)
    res_sys_b = run_shape(cases, raw_dir, out_dir, device=device)

    # 5. Collect System Metrics
    rows = []
    for case in cases:
        cid = case["id"]
        for model_name, res_dict in [
            ("TripoSR (System A)", res_sys_a),
            ("Shap-E (System B)", res_sys_b),
        ]:
            _, latency, vram = res_dict[cid]
            rows.append({
                "Test_Case": cid,
                "Image_File": case.get("filename", ""),
                "Model": model_name,
                "Latency_s": latency,
                "Peak_VRAM_GB": vram,
            })

    # Render Charts
    df = pd.DataFrame(rows)
    plot_results(df, results_dir)

if __name__ == "__main__":
    main()