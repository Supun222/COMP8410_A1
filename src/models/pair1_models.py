import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    LEditsPPPipelineStableDiffusion,
    DPMSolverMultistepScheduler
)

DEVICE = "cpu"
DTYPE = torch.float32

def run_instructpix2pix(image_path, instruction, output_path, seed=42):
    """
    Model A: InstructPix2Pix
    Learns conditional forward diffusion from paired synthetic instructions.
    Best for broader stylistic contextual shifts.
    """
    print(f"\n[Model A: InstructPix2Pix] Processing dish: {image_path}")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    input_img = Image.open(image_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # image_guidance_scale balances food identity preservation vs. instruction adherence
    result = pipe(
        prompt=instruction,
        image=input_img,
        num_inference_steps=35,
        image_guidance_scale=1.6,   # >1.5 prevents the food itself from mutating
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    result.save(output_path)
    print(f"-> Saved InstructPix2Pix output to: {output_path}")

    # Explicit memory cleanup for Colab / single GPU setups
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_ledits_pp(image_path, edit_concepts, output_path, seed=42):
    """
    Model B: LEDITS++
    Uses DPM-Solver++ inversion with cross-attention semantic grounding.
    Best for precise concept addition/swapping with strict localized boundaries.
    """
    print(f"\n[Model B: LEDITS++] Processing dish: {image_path}")
    pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    input_img = Image.open(image_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Invert image into latent trajectory without text condition
    print("-> Computing DPM-Solver++ inversion trajectory...")
    pipe.invert(
        image=input_img,
        num_inversion_steps=35,
        skip=0.15,
        generator=generator
    )

    # Perform targeted semantic editing
    print(f"-> Applying culinary concept guidance: {edit_concepts}")
    result = pipe(
        editing_prompt=edit_concepts,
        reverse_editing_direction=[False] * len(edit_concepts),
        edit_guidance_scale=[7.0] * len(edit_concepts),
        edit_threshold=[0.75] * len(edit_concepts), # Masks out areas that don't match the concept
        num_inference_steps=35
    ).images[0]

    result.save(output_path)
    print(f"-> Saved LEDITS++ output to: {output_path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Example input: A raw prep photo of pasta, roasted steak, or a plated dessert
    sample_dish_path = "data/chef_dish_raw.jpg"

    if not os.path.exists(sample_dish_path):
        # Create a placeholder dummy image if testing before providing a real photo
        dummy = Image.new("RGB", (512, 512), color=(180, 120, 80))
        dummy.save(sample_dish_path)
        print(f"Generated sample placeholder image at {sample_dish_path}. Replace with real food photo.")

    # 1. InstructPix2Pix: Action-oriented instruction
    chef_instruction_a = "Place the plated dish on a dark rustic oak dining table, fine dining restaurant setting, soft warm bokeh lighting"

    # 2. LEDITS++: Exact visual concepts to ground semantically
    chef_concepts_b = ["dark rustic oak table", "luxury restaurant ambiance"]

    run_instructpix2pix(
        image_path=sample_dish_path,
        instruction=chef_instruction_a,
        output_path="results/dish_edited_instructpix2pix.png",
        seed=101
    )

    run_ledits_pp(
        image_path=sample_dish_path,
        edit_concepts=chef_concepts_b,
        output_path="results/dish_edited_ledits.png",
        seed=101
    )