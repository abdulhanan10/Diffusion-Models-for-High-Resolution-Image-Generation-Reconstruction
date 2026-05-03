# ═══════════════════════════════════════════════════════════════════
# DDPM Image Generator — Standalone Gradio App
# Run locally:  python app.py
# Deploy:       gradio deploy  (from folder containing app.py + best_model.pth)
# ═══════════════════════════════════════════════════════════════════
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import gradio as gr

# ── Config ────────────────────────────────────────────────────────
IMG_SIZE = 128
CHANNELS = 3
T        = 400
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Helpers ───────────────────────────────────────────────────────
def unnormalize(t):
    return (t.clamp(-1, 1) + 1) / 2

# ── Model classes (must be here for torch.load to work) ──────────
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.norm1    = nn.GroupNorm(num_groups, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(num_groups, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.softmax(q.transpose(-1, -2) @ k * (C ** -0.5), dim=-1)
        out  = self.proj((attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W))
        return out + x

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    def forward(self, x, t_emb):
        x = self.res1(x, t_emb); x = self.res2(x, t_emb)
        x = self.attn(x) if isinstance(self.attn, SelfAttention) else x
        return self.down(x), x

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
    def forward(self, x, skip, t_emb):
        x = torch.cat([self.up(x), skip], dim=1)
        x = self.res1(x, t_emb); x = self.res2(x, t_emb)
        return self.attn(x) if isinstance(self.attn, SelfAttention) else x

class UNet(nn.Module):
    def __init__(self, img_channels=3, base_dim=64, dim_mults=(1,2,4), T=400):
        super().__init__()
        time_dim = base_dim * 4
        dims = [base_dim * m for m in dim_mults]
        self.time_mlp  = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.init_conv = nn.Conv2d(img_channels, dims[0], 3, padding=1)
        self.down1     = DownBlock(dims[0], dims[0], time_dim, False)
        self.down2     = DownBlock(dims[0], dims[1], time_dim, False)
        self.down3     = DownBlock(dims[1], dims[2], time_dim, True)
        self.mid_res1  = ResBlock(dims[2], dims[2], time_dim)
        self.mid_attn  = SelfAttention(dims[2])
        self.mid_res2  = ResBlock(dims[2], dims[2], time_dim)
        self.up3       = UpBlock(dims[2], dims[2], dims[1], time_dim, True)
        self.up2       = UpBlock(dims[1], dims[1], dims[0], time_dim, False)
        self.up1       = UpBlock(dims[0], dims[0], dims[0], time_dim, False)
        self.out_norm  = nn.GroupNorm(8, dims[0])
        self.out_conv  = nn.Conv2d(dims[0], img_channels, 1)
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        x1, s1 = self.down1(x,  t_emb)
        x2, s2 = self.down2(x1, t_emb)
        x3, s3 = self.down3(x2, t_emb)
        x3 = self.mid_res2(self.mid_attn(self.mid_res1(x3, t_emb)), t_emb)
        x  = self.up3(x3, s3, t_emb)
        x  = self.up2(x,  s2, t_emb)
        x  = self.up1(x,  s1, t_emb)
        return self.out_conv(F.silu(self.out_norm(x)))

# ── Scheduler ─────────────────────────────────────────────────────
class DDPMScheduler:
    def __init__(self, T=400, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T      = T
        self.device = device
        self.betas  = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_alphas_cumprod           = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas = (1.0 / self.alphas).sqrt()
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    @staticmethod
    def _gather(values, t, x_ref):
        return values.gather(0, t).view(-1, *([1] * (x_ref.ndim - 1)))
    @torch.no_grad()
    def p_sample(self, model, x_t, t_scalar):
        t_batch     = torch.full((x_t.shape[0],), t_scalar, dtype=torch.long, device=self.device)
        pred_noise  = model(x_t, t_batch)
        beta_t      = self._gather(self.betas, t_batch, x_t)
        sqrt_ra     = self._gather(self.sqrt_recip_alphas, t_batch, x_t)
        sqrt_1m_acp = self._gather(self.sqrt_one_minus_alphas_cumprod, t_batch, x_t)
        mean = sqrt_ra * (x_t - beta_t / sqrt_1m_acp * pred_noise)
        if t_scalar == 0:
            return mean
        post_var = self._gather(self.posterior_variance, t_batch, x_t)
        return mean + post_var.sqrt() * torch.randn_like(x_t)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, return_intermediates=False, every_n=50):
        model.eval()
        x = torch.randn(shape, device=self.device)
        intermediates = []
        for t in tqdm(reversed(range(self.T)), desc='Sampling', total=self.T, leave=False):
            x = self.p_sample(model, x, t)
            if return_intermediates and (t % every_n == 0 or t == self.T - 1):
                intermediates.append(x.clone().cpu())
        return (x.cpu(), intermediates) if return_intermediates else x.cpu()

# ── Load model ────────────────────────────────────────────────────
print('Loading model...')
inference_model = UNet(img_channels=CHANNELS, base_dim=64, dim_mults=(1,2,4), T=T)
state_dict = torch.load('best_model.pth', map_location=device)
inference_model.load_state_dict(state_dict)
inference_model = inference_model.to(device).eval()
scheduler = DDPMScheduler(T=T, device=device)
print('Model loaded on', device)

# ── Gradio callbacks ──────────────────────────────────────────────
def to_pil(t):
    arr = (unnormalize(t).permute(1,2,0).clamp(0,1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_images(num_images, seed):
    torch.manual_seed(int(seed))
    num_images = max(1, min(int(num_images), 5))
    shape = (num_images, CHANNELS, IMG_SIZE, IMG_SIZE)
    final = scheduler.p_sample_loop(inference_model, shape)
    return [to_pil(final[i]) for i in range(num_images)]

def generate_with_steps(num_images, seed):
    torch.manual_seed(int(seed))
    num_images = max(1, min(int(num_images), 4))
    shape = (num_images, CHANNELS, IMG_SIZE, IMG_SIZE)
    final, intermediates = scheduler.p_sample_loop(
        inference_model, shape, return_intermediates=True, every_n=80
    )
    final_pils = [to_pil(final[i]) for i in range(num_images)]
    step_pils  = [to_pil(f[0]) for f in intermediates]
    return final_pils, step_pils

# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(title='DDPM Face Generator') as demo:
    gr.Markdown('# DDPM Face Generator\nGenerates faces via learned reverse diffusion (DDPM trained on CelebA-HQ).')

    with gr.Tab('Generate Images'):
        with gr.Row():
            num_slider = gr.Slider(1, 5, value=3, step=1, label='Number of images')
            seed_input = gr.Number(value=42, label='Random seed')
        gen_btn = gr.Button('Generate', variant='primary')
        output_gallery = gr.Gallery(label='Generated Images', columns=5, height='auto')
        gen_btn.click(generate_images, inputs=[num_slider, seed_input], outputs=output_gallery)

    with gr.Tab('Generate + Show Denoising Steps'):
        with gr.Row():
            num_slider2 = gr.Slider(1, 4, value=2, step=1, label='Number of images')
            seed_input2 = gr.Number(value=42, label='Random seed')
        step_btn = gr.Button('Generate with steps', variant='primary')
        with gr.Row():
            final_gallery = gr.Gallery(label='Final images', columns=4)
            steps_gallery = gr.Gallery(label='Denoising steps (first image)', columns=6)
        step_btn.click(generate_with_steps, inputs=[num_slider2, seed_input2], outputs=[final_gallery, steps_gallery])

    with gr.Tab('Model Info'):
        gr.Markdown("""
        ## Architecture
        | Component | Detail |
        |---|---|
        | Model | U-Net (64→128→256 channels) |
        | Time embedding | Sinusoidal + MLP |
        | Attention | Self-attention at bottleneck |
        | Timesteps T | 400 |
        | Noise schedule | Linear β: 1e-4 → 0.02 |
        | Training loss | MSE on predicted noise |
        | Dataset | CelebA-HQ 128×128 |
        """)

demo.launch()
