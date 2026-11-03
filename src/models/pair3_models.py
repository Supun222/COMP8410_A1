import time
import torch
from diffusers import (
    DiffusionPipeline, 
    AnimateDiffPipeline, 
    MotionAdapter, 
    EulerAncestralDiscreteScheduler
)
from diffusers.utils import export_to_video

def count_params(model):
    """Count parameters in millions."""
    if model is None:
        return 0.0
    return sum(p.numel() for p in model.parameters()) / 1e6

def run_modelscope(cases, out_dir, device="cuda"):
    """
    Model A: ModelScopeT2V (Inflated 3D U-Net)
    Returns: {cid: (out_path, latency, vram, total_p)}
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model A] Loading ModelScopeT2V on {device}...")
    
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", 
        torch_dtype=dtype
    ).to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae)
    results = {}

    for case in cases:
        cid = case["id"]
        out_path = str(out_dir / f"sys_a_{cid}.mp4")
        print(f" -> Generating {cid}...")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(42)
        frames = pipe(
            case["prompt"], 
            negative_prompt=case.get("negative_prompt", ""),
            num_inference_steps=25, 
            generator=generator
        ).frames[0]

        if device == "cuda":
            torch.cuda.synchronize()
            
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        export_to_video(frames, out_path, fps=8)
        
        # Consistent tuple return format
        results[cid] = (out_path, latency, vram, total_p)
        print(f"    Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del pipe
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        
    return results

def run_animatediff(cases, out_dir, device="cuda"):
    """
    Model B: AnimateDiff (Plug-and-play temporal module)
    Returns: {cid: (out_path, latency, vram, total_p)}
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model B] Loading AnimateDiff on {device}...")
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism", 
        motion_adapter=adapter, 
        torch_dtype=dtype
    ).to(device)
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear"
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae) + count_params(adapter)
    results = {}

    for case in cases:
        cid = case["id"]
        out_path = str(out_dir / f"sys_b_{cid}.mp4")
        print(f" -> Generating {cid}...")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(42)
        frames = pipe(
            prompt=case["prompt"],
            negative_prompt=case.get("negative_prompt", ""),
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator
        ).frames[0]

        if device == "cuda":
            torch.cuda.synchronize()
            
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        export_to_video(frames, out_path, fps=8)
        
        # Consistent tuple return format
        results[cid] = (out_path, latency, vram, total_p)
        print(f"    Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del pipe
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        
    return results