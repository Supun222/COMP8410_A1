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

def count_params(model):
    if model is None:
        return 0.0
    return sum(p.numel() for p in model.parameters()) / 1e6

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).resize((512, 512)))
    cap.release()
    return frames

def run_ip2p_frame_by_frame(cases, raw_dir, device="cuda"): 
    dtype = torch.float16 if device == "cuda" else torch.float32
        
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=dtype
    ).to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    results = {}

    for case in cases:
        cid = case["id"]
        in_path = str(raw_dir / case["filename"])
        
        frames = extract_frames(in_path)
        if not frames:
            continue

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
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

        if device == "cuda":
            torch.cuda.synchronize()
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        # Return raw generated frames and performance metrics
        results[cid] = (edited_frames, latency, vram)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results

def run_animatediff_v2v(cases, raw_dir, device="cuda"): 
    dtype = torch.float16 if device == "cuda" else torch.float32 
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        motion_adapter=adapter, 
        torch_dtype=dtype
    ).to(device)
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    if device == "cuda":
        pipe.enable_model_cpu_offload()

    results = {}

    for case in cases:
        cid = case["id"]
        in_path = str(raw_dir / case["filename"])
        
        frames = extract_frames(in_path)
        if not frames:
            continue 

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(
            video=frames,
            prompt=case["prompt"], 
            num_inference_steps=25, 
            strength=0.8,
            generator=generator
        )
        edited_frames = output.frames[0]

        if device == "cuda":
            torch.cuda.synchronize()
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        # Return raw generated frames and performance metrics
        results[cid] = (edited_frames, latency, vram)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results