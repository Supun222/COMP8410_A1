import sys
import os
import time
import torch
from pathlib import Path
from PIL import Image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import types
import mcubes


# Add TripoSR directory to sys.path so 'tsr' is importable
triposr_path = Path("/content/TripoSR")
if not triposr_path.exists():
    triposr_path = Path("./TripoSR")  # Fallback for local path

if str(triposr_path) not in sys.path:
    sys.path.insert(0, str(triposr_path))

# Create a mock torchmcubes module dynamically in memory
mock_torchmcubes = types.ModuleType("torchmcubes")

def custom_marching_cubes(volume, threshold=0.0):
    """Drop-in replacement for torchmcubes using PyMCubes."""
    if isinstance(volume, torch.Tensor):
        vol_np = volume.detach().cpu().numpy()
    else:
        vol_np = volume
        
    vertices, triangles = mcubes.marching_cubes(vol_np, threshold)
    
    verts_tensor = torch.from_numpy(vertices.astype("float32")).to(volume.device if isinstance(volume, torch.Tensor) else "cpu")
    faces_tensor = torch.from_numpy(triangles.astype("int64")).to(volume.device if isinstance(volume, torch.Tensor) else "cpu")
    return verts_tensor, faces_tensor

mock_torchmcubes.marching_cubes = custom_marching_cubes
sys.modules["torchmcubes"] = mock_torchmcubes

print("torchmcubes mock registered successfully! TripoSR can now run without compilation errors.")


# Now this import will succeed
from tsr.system import TSR


def run_triposr(cases, raw_dir, out_dir, device="cuda"):

    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Model A: TripoSR] Initializing model on {device}...")
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)

    results = {}

    for idx, case in enumerate(cases):
        cid = case["id"]
        img_name = case.get("filename", case.get("image_name"))
        input_path = raw_dir / img_name
        output_mesh_path = out_dir / f"sys_a_{Path(img_name).stem}.obj"

        print(f"  -> [TripoSR] Processing Case {idx+1}/{len(cases)}: {img_name}")
        image = Image.open(input_path).convert("RGB")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            scene_codes = model([image], device=device) 
            meshes = model.extract_mesh(scene_codes, True, resolution=256, threshold=25.0)

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
                karras_steps=64,
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