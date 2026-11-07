import sys
import os
import time
import types
from pathlib import Path
from PIL import Image
import torch
import mcubes

mock_torchmcubes = types.ModuleType("torchmcubes")

def custom_marching_cubes(volume, threshold=0.0):
    """Drop-in replacement for torchmcubes using PyMCubes."""
    if isinstance(volume, torch.Tensor):
        vol_np = volume.detach().cpu().numpy()
        device = volume.device
    else:
        vol_np = volume
        device = "cpu"
        
    vertices, triangles = mcubes.marching_cubes(vol_np, float(threshold))
    
    verts_tensor = torch.from_numpy(vertices.astype("float32")).to(device)
    faces_tensor = torch.from_numpy(triangles.astype("int64")).to(device)
    return verts_tensor, faces_tensor

mock_torchmcubes.marching_cubes = custom_marching_cubes
sys.modules["torchmcubes"] = mock_torchmcubes

CURRENT_FILE = Path(__file__).resolve()

# Navigates from src/models/pair2_models.py -> COMP8410_A1/TripoSR
TRIPOSR_DIR = CURRENT_FILE.parents[2] / "TripoSR"

if TRIPOSR_DIR.exists() and str(TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(TRIPOSR_DIR))

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
except ImportError:
    # Fallback if installed via pip / standard package
    from tsr.system import TSR
    remove_background = None
    resize_foreground = None

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


def run_triposr(cases, raw_dir, out_dir, device="cuda"):
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Model A: TripoSR] Initializing model on {device}...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", 
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    model.eval()

    results = {}

    for idx, case in enumerate(cases):
        cid = case["id"]
        img_name = case.get("filename", case.get("image_name"))
        input_path = raw_dir / img_name
        output_mesh_path = out_dir / f"sys_a_{Path(img_name).stem}.obj"

        print(f"  -> [TripoSR] Processing Case {idx+1}/{len(cases)}: {img_name}")
        image = Image.open(input_path).convert("RGB")

        # TripoSR expects a segmented foreground object (RGBA)
        # Background segmentation + 3-channel solid backdrop
        if remove_background is not None:
            image = remove_background(image, rembg_session=None)
            image = resize_foreground(image, ratio=0.85)
            if image.mode == "RGBA":
                bg = Image.new("RGBA", image.size, (127, 127, 127, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg.convert("RGB")
        else:
            image = image.convert("RGB")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        # Autocast & Optimization 3: Optimal Marching Cubes grid (160)
        with torch.no_grad():
            # Generate scene codes (cast to float32 after inference)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                scene_codes = model([image], device=device)
            
            # Match data types for F.grid_sample
            if isinstance(scene_codes, torch.Tensor):
                scene_codes = scene_codes.float()
            elif isinstance(scene_codes, (list, tuple)):
                scene_codes = [sc.float() if isinstance(sc, torch.Tensor) else sc for sc in scene_codes]

            # Extract polygonal mesh using resolution=160
            meshes = model.extract_mesh(
                scene_codes, 
                has_vertex_color=True, 
                resolution=160, 
                threshold=25.0
            ) 

        if device == "cuda":
            torch.cuda.synchronize()
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        # Export reconstructed 3D mesh
        meshes[0].export(str(output_mesh_path))
        results[cid] = (str(output_mesh_path), latency, vram)
        print(f"     Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

def run_shape(cases, raw_dir, out_dir, device="cuda"):
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Model B: Shap-E] Initializing model on {device}...")
    xm = load_model("transmitter", device=device)
    model = load_model("image300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    results = {}

    for idx, case in enumerate(cases):
        cid = case["id"]
        img_name = case.get("filename", case.get("image_name"))
        input_path = raw_dir / img_name
        output_mesh_path = out_dir / f"sys_b_{Path(img_name).stem}.ply"

        print(f"  -> [Shap-E] Processing Case {idx+1}/{len(cases)}: {img_name}")
        image = Image.open(input_path).convert("RGB").resize((256, 256))

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=3.0,
                model_kwargs=dict(images=[image]),
                progress=False,
                clip_denoised=True,
                use_fp16=(device == "cuda"),
                use_karras=True,
                karras_steps=36,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()

        if device == "cuda":
            torch.cuda.synchronize()
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        with open(output_mesh_path, "wb") as f:
            mesh.write_ply(f)

        results[cid] = (str(output_mesh_path), latency, vram)
        print(f"     Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del xm, model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results