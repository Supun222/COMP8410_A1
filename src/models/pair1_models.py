import os
import time
import torch
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    LEditsPPPipelineStableDiffusion,
    DPMSolverMultistepScheduler
)

def run_instruct_pix2pix(cases, raw_dir, out_dir, device="cuda"):
    """
    Model A: InstructPix2Pix batch runner.
    Takes cases, executes inference, measures latency/VRAM, and returns {cid: (orig, edit, latency, vram)}.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model A: InstructPix2Pix] Initializing on {device}...")
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.enable_attention_slicing()

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for idx, case in enumerate(cases):
        cid = case["id"]
        # Correctly prioritize 'filename' over 'image_name'
        img_name = case.get("filename", case.get("image_name"))
        input_path = os.path.join(raw_dir, img_name)
        
        prompt = case.get("instruction", case.get("prompt", case.get("target_caption", "enhance the setting")))
        out_name = f"sys_a_{os.path.splitext(img_name)[0]}.png"
        output_path = os.path.join(out_dir, out_name)

        print(f"  -> [InstructPix2Pix] Processing Case {idx+1}/{len(cases)}: {img_name}")
        orig_img = Image.open(input_path).convert("RGB").resize((512, 512))

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(case.get("seed", 42))

        result = pipe(
            prompt=prompt,
            image=orig_img,
            num_inference_steps=35,
            image_guidance_scale=1.6,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        result.save(output_path)
        results[cid] = (orig_img, result, latency, vram)
        print(f"     Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

def run_ledits_pp(cases, raw_dir, out_dir, device="cuda"):
    """
    Model B: LEDITS++ batch runner.
    Performs DPM-Solver++ inversion followed by semantic guidance and returns {cid: (orig, edit, latency, vram)}.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n[Model B: LEDITS++] Initializing on {device}...")

    pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.enable_attention_slicing()

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for idx, case in enumerate(cases):
        cid = case["id"]
        # Correctly prioritize 'filename' over 'image_name'
        img_name = case.get("filename", case.get("image_name"))
        input_path = os.path.join(raw_dir, img_name)

        concepts = case.get("concepts", case.get("concept", case.get("target_caption", "studio table")))
        if isinstance(concepts, str):
            concepts = [concepts]

        out_name = f"sys_b_{os.path.splitext(img_name)[0]}.png"
        output_path = os.path.join(out_dir, out_name)

        print(f"  -> [LEDITS++] Processing Case {idx+1}/{len(cases)}: {img_name}")
        orig_img = Image.open(input_path).convert("RGB").resize((512, 512))

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(case.get("seed", 42))

        pipe.invert(
            image=orig_img,
            num_inversion_steps=35,
            skip=0.15,
            generator=generator
        )

        result = pipe(
            editing_prompt=concepts,
            reverse_editing_direction=[False] * len(concepts),
            edit_guidance_scale=[7.0] * len(concepts),
            edit_threshold=[0.75] * len(concepts),
            num_inference_steps=35
        ).images[0]

        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        result.save(output_path)
        results[cid] = (orig_img, result, latency, vram)
        print(f"     Finished in {latency:.2f}s (Peak VRAM: {vram:.2f} GB)")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

# Aliases for backwards compatibility
run_instructpix2pix = run_instruct_pix2pix
run_ledits = run_ledits_pp