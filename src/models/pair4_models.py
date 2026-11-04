import time
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    AnimateDiffVideoToVideoPipeline,
    MotionAdapter,
    EulerAncestralDiscreteScheduler
)
from diffusers.utils import export_to_video

def count_params(model):
    if model is None: return 0.0
    return sum(p.numel() for p in model.parameters()) / 1e6

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).resize((512, 512)))
    cap.release()
    return frames

def run_ip2p_frame_by_frame(cases, raw_dir, out_dir, device="cuda"):
    """Model A: Frame-by-Frame InstructPix2Pix (Baseline)"""
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model A] Loading InstructPix2Pix on {device}...")
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=dtype
    ).to(device)

    if device == "cuda": pipe.enable_model_cpu_offload()
    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae)
    results = {}

    for case in cases:
        cid = case["id"]
        in_path = str(raw_dir / case["filename"])
        out_path = str(out_dir / f"sys_a_{cid}.mp4")
        
        frames = extract_frames(in_path)
        if not frames: continue
        print(f" -> Generating {cid} (Frame-by-Frame)...")

        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(42)
        edited_frames = []
        
        for frame in frames:
            res = pipe(
                case["prompt"], 
                image=frame, 
                num_inference_steps=20, 
                image_guidance_scale=1.5,
                generator=generator
            ).images[0]
            edited_frames.append(np.array(res))

        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        export_to_video(edited_frames, out_path, fps=8)
        results[cid] = (out_path, latency, vram, total_p)

    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results

def run_animatediff_v2v(cases, raw_dir, out_dir, device="cuda"):
    """Model B: AnimateDiff Video-to-Video (Temporally Coherent)"""
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model B] Loading AnimateDiff V2V on {device}...")
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        motion_adapter=adapter, 
        torch_dtype=dtype
    ).to(device)
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    if device == "cuda": pipe.enable_model_cpu_offload()

    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae) + count_params(adapter)
    results = {}

    for case in cases:
        cid = case["id"]
        in_path = str(raw_dir / case["filename"])
        out_path = str(out_dir / f"sys_b_{cid}.mp4")
        
        frames = extract_frames(in_path)
        if not frames: continue
        print(f" -> Generating {cid} (AnimateDiff V2V)...")

        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(42)
        edited_frames = pipe(
            video=frames,
            prompt=case["prompt"],
            negative_prompt=case.get("negative_prompt", ""),
            num_inference_steps=25,
            strength=0.8,
            generator=generator
        ).frames[0]

        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        export_to_video(edited_frames, out_path, fps=8)
        results[cid] = (out_path, latency, vram, total_p)

    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results