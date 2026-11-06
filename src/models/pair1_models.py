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

def count_params(model): 
    if model is None:
        return 0.0
    return sum(p.numel() for p in model.parameters()) / 1e6

def run_instruct_pix2pix(cases, raw_dir, out_dir, device="cuda"):

     # device type
    dtype = torch.float16 if device == "cuda" else torch.float32 
    
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

    # total count params
    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae)

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
            num_inference_steps=40,
            image_guidance_scale=1.2,  
            guidance_scale=9.0,  
            generator=generator
        ).images[0]

        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        result.save(output_path)
        results[cid] = (orig_img, result, latency, vram, total_p) 

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

def run_ledits_pp(cases, raw_dir, out_dir, device="cuda"):

    dtype = torch.float16 if device == "cuda" else torch.float32 

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

    # total count params
    total_p = count_params(pipe.unet) + count_params(pipe.text_encoder) + count_params(pipe.vae)

    for idx, case in enumerate(cases):
        cid = case["id"]
        img_name = case.get("filename", case.get("image_name"))
        input_path = os.path.join(raw_dir, img_name)

        # Harmonized with Model A: retrieve the exact same prompt string
        prompt = case.get("prompt", case.get("instruction", case.get("target_caption", "studio table")))
        # LEDITS++ requires concepts as a list
        concepts = [prompt]

        out_name = f"sys_b_{os.path.splitext(img_name)[0]}.png"
        output_path = os.path.join(out_dir, out_name)

        print(f"  -> [LEDITS++] Processing Case {idx+1}/{len(cases)}: {img_name}")
        orig_img = Image.open(input_path).convert("RGB").resize((512, 512))

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(case.get("seed", 42))

        # iInvert image into latent trajectory
        pipe.invert(
          image=orig_img,
          num_inversion_steps=40,
          skip=0.05,  
          generator=generator
        )

        # apply editing concept
        result = pipe(
          editing_prompt=concepts,
          reverse_editing_direction=[False],
          edit_guidance_scale=[8.0],
          edit_threshold=[0.35],  
          num_inference_steps=40
        ).images[0]
        
        # Metric Collection (Identical to Model A)
        latency = time.time() - t0
        vram = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

        result.save(output_path)
        results[cid] = (orig_img, result, latency, vram, total_p) 

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

# aliases for backwards compatibility
run_instructpix2pix = run_instruct_pix2pix
run_ledits = run_ledits_pp