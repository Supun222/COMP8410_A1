# src/evaluation/image_metrics.py
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import open_clip


class ImageEditingEvaluator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Evaluator] Initializing OpenCLIP (ViT-B-32) and LPIPS on {self.device}...")
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()

    def evaluate_pair(self, orig_pil: Image.Image, edit_pil: Image.Image, source_caption: str, target_caption: str) -> dict:
        orig_pil = orig_pil.convert("RGB").resize((512, 512))
        edit_pil = edit_pil.convert("RGB").resize((512, 512))

        orig_np = np.array(orig_pil)
        edit_np = np.array(edit_pil)

        # 1. Structural & Pixel Preservation Metrics (SSIM & PSNR)
        ssim_val = ssim(orig_np, edit_np, channel_axis=2, data_range=255)
        psnr_val = psnr(orig_np, edit_np, data_range=255)

        # 2. Perceptual Drift Metric (LPIPS - lower = better background preservation)
        t_orig = (torch.tensor(orig_np).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(self.device)
        t_edit = (torch.tensor(edit_np).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lpips_val = self.lpips_fn(t_orig, t_edit).item()

        # 3. Semantic Alignment Metrics (CLIP-T & CLIP-Directional)
        img_orig_clip = self.clip_preprocess(orig_pil).unsqueeze(0).to(self.device)
        img_edit_clip = self.clip_preprocess(edit_pil).unsqueeze(0).to(self.device)
        txt_tokens = self.tokenizer([source_caption, target_caption]).to(self.device)

        with torch.no_grad():
            f_img_orig = self.clip_model.encode_image(img_orig_clip)
            f_img_edit = self.clip_model.encode_image(img_edit_clip)
            f_txt = self.clip_model.encode_text(txt_tokens)

            f_img_orig /= f_img_orig.norm(dim=-1, keepdim=True)
            f_img_edit /= f_img_edit.norm(dim=-1, keepdim=True)
            f_txt /= f_txt.norm(dim=-1, keepdim=True)

            clip_t = (f_img_edit @ f_txt[1].unsqueeze(-1)).item()

            delta_img = f_img_edit - f_img_orig
            delta_txt = f_txt[1] - f_txt[0]
            delta_img /= (delta_img.norm(dim=-1, keepdim=True) + 1e-8)
            delta_txt /= (delta_txt.norm(dim=-1, keepdim=True) + 1e-8)
            clip_dir = (delta_img @ delta_txt.unsqueeze(-1)).item()

        return {
            "CLIP_Text_Sim": round(clip_t, 4),
            "CLIP_Dir_Sim": round(clip_dir, 4),
            "SSIM": round(float(ssim_val), 4),
            "PSNR_dB": round(float(psnr_val), 2),
            "LPIPS": round(float(lpips_val), 4),
        }