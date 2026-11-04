import os
import time
import re
import xml.etree.ElementTree as ET
import hashlib
import pickle
import threading
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Global model state lock for thread safety
jax_lock = threading.Lock()

app = FastAPI(title="SVG Flash-Flow Diffusion Daemon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CMD_DIM = 8

# Normalized command types
CMD_PAD = 0.0
CMD_MOVE = 1.0 / 3.0   # M x y (~0.333)
CMD_LINE = 2.0 / 3.0   # L x y (~0.666)
CMD_CUBIC = 3.0 / 3.0  # C cx1 cy1 cx2 cy2 x y (1.000)

MASTER_CKPT_PATH = Path("/app/checkpoints/master_model_params.pkl")
REPLAY_BUFFER_LIMIT = 32

# Shared Global Telemetry State
training_metrics = {
    "status": "Initializing",
    "current_file": "None",
    "epoch": 0,
    "max_epochs": 1000,
    "mse_loss": 0.0,
    "ssim_loss": 0.0,
    "combined_loss": 0.0,
    "high_t_mse": 0.0,
    "low_t_mse": 0.0,
    "velocity_cosine_sim": 0.0,
    "spatial_coord_var": 0.0,
    "effective_rank": 0.0,
    "kernel_alignment": 0.0,
    "attention_entropy": 0.0,
    "dataset_count": 0,
    "buffer_size": 0,
    "sequence_length": 0,
    "mode": "Standby",
    "active_svg": ""
}


# -----------------------------------------------------------------------------
# 1. Dynamic SVG Path Parsing & Serialization (Blank Canvas Fix)
# -----------------------------------------------------------------------------

def parse_svg_path_d(d_str, stroke_w=2.0):
    commands = []
    tokens = re.findall(r'([A-Za-z])|(-?\d*\.?\d+(?:e[-+]?\d+)?)', d_str)
    
    current_cmd = None
    args = []
    
    def process_command(cmd, coords):
        c_type = CMD_PAD
        x = y = cx1 = cy1 = cx2 = cy2 = 0.5
        
        if cmd in ('M', 'm'):
            c_type = CMD_MOVE
            if len(coords) >= 2: x, y = coords[0] / 100.0, coords[1] / 100.0
        elif cmd in ('L', 'l'):
            c_type = CMD_LINE
            if len(coords) >= 2: x, y = coords[0] / 100.0, coords[1] / 100.0
        elif cmd in ('C', 'c'):
            c_type = CMD_CUBIC
            if len(coords) >= 6:
                cx1, cy1 = coords[0] / 100.0, coords[1] / 100.0
                cx2, cy2 = coords[2] / 100.0, coords[3] / 100.0
                x, y = coords[4] / 100.0, coords[5] / 100.0
        else:
            return None
            
        sw = stroke_w / 10.0
        return [c_type, x, y, cx1, cy1, cx2, cy2, sw]

    for t_cmd, t_num in tokens:
        if t_cmd:
            if current_cmd and args:
                res = process_command(current_cmd, args)
                if res: commands.append(res)
                args = []
            current_cmd = t_cmd
        elif t_num:
            args.append(float(t_num))

    if current_cmd and args:
        res = process_command(current_cmd, args)
        if res: commands.append(res)

    return commands


def svg_to_vector_sequence(svg_text):
    vectors = []
    try:
        clean_text = re.sub(r'xmlns="[^"]+"', '', svg_text)
        root = ET.fromstring(clean_text)
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            stroke_w = float(elem.get('stroke-width', 2.0))
            
            if tag == 'path':
                d = elem.get('d', '')
                parsed = parse_svg_path_d(d, stroke_w=stroke_w)
                vectors.extend(parsed)
            elif tag == 'circle':
                cx, cy = float(elem.get('cx', 50)), float(elem.get('cy', 50))
                r = float(elem.get('r', 10))
                vectors.append([CMD_MOVE, (cx + r)/100.0, cy/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0])
                vectors.append([CMD_CUBIC, (cx - r)/100.0, cy/100.0, (cx + r)/100.0, (cy + r)/100.0, (cx - r)/100.0, (cy + r)/100.0, stroke_w/10.0])
            elif tag == 'rect':
                x, y = float(elem.get('x', 10)), float(elem.get('y', 10))
                w, h = float(elem.get('width', 20)), float(elem.get('height', 20))
                vectors.append([CMD_MOVE, x/100.0, y/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0])
                vectors.append([CMD_LINE, (x+w)/100.0, y/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0])
                vectors.append([CMD_LINE, (x+w)/100.0, (y+h)/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0])
                vectors.append([CMD_LINE, x/100.0, (y+h)/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0])
    except Exception as e:
        print(f"[!] XML Path Parsing error: {e}")

    if not vectors:
        vectors = [[CMD_MOVE, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.2],
                   [CMD_LINE, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.2]]

    return np.array(vectors, dtype=np.float32)


def vector_sequence_to_svg(cmd_matrix):
    """Robust conversion ensuring non-blank SVG rendering on intermediate passes."""
    path_d_list = []
    
    if len(cmd_matrix) == 0:
        cmd_matrix = np.array([[CMD_MOVE, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2],
                               [CMD_LINE, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.2]], dtype=np.float32)

    avg_sw = np.mean(cmd_matrix[:, 7]) * 10.0 if len(cmd_matrix) > 0 else 2.0
    stroke_width = np.clip(avg_sw, 1.0, 8.0)

    for i, vec in enumerate(cmd_matrix):
        c_type, x, y, cx1, cy1, cx2, cy2, sw = vec
        
        # Enforce canvas boundary constraints (5 to 95)
        px, py = np.clip(x * 100.0, 5.0, 95.0), np.clip(y * 100.0, 5.0, 95.0)
        pcx1, pcy1 = np.clip(cx1 * 100.0, 5.0, 95.0), np.clip(cy1 * 100.0, 5.0, 95.0)
        pcx2, pcy2 = np.clip(cx2 * 100.0, 5.0, 95.0), np.clip(cy2 * 100.0, 5.0, 95.0)
        
        if i == 0 or (c_type <= 0.45):
            path_d_list.append(f"M {px:.1f} {py:.1f}")
        elif 0.45 < c_type <= 0.78:
            path_d_list.append(f"L {px:.1f} {py:.1f}")
        else:
            path_d_list.append(f"C {pcx1:.1f} {pcy1:.1f}, {pcx2:.1f} {pcy2:.1f}, {px:.1f} {py:.1f}")

    d_string = " ".join(path_d_list) if path_d_list else "M 15 15 L 85 85"
    
    return f'''<svg viewBox="0 0 100 100" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
  <rect width="100" height="100" fill="#ffffff"/>
  <path d="{d_string}" fill="none" stroke="#0f172a" stroke-width="{stroke_width:.1f}" stroke-linecap="round" stroke-linejoin="round" />
</svg>'''.strip()


def collate_svg_batch(vector_list):
    B = len(vector_list)
    seq_lens = [len(v) for v in vector_list]
    N_max = max(seq_lens)
    
    batch_tensor = np.zeros((B, N_max, CMD_DIM), dtype=np.float32)
    mask = np.zeros((B, N_max), dtype=bool)
    
    for i, vec in enumerate(vector_list):
        length = len(vec)
        batch_tensor[i, :length, :] = vec
        mask[i, :length] = True
        
    return batch_tensor, mask


# -----------------------------------------------------------------------------
# 2. FlashAttention Block-Tiled Module
# -----------------------------------------------------------------------------

class FlashSelfAttention(nn.Module):
    num_heads: int = 32
    d_model: int = 2048
    block_size: int = 64

    @nn.compact
    def __call__(self, x, mask=None):
        B, N, D = x.shape
        head_dim = D // self.num_heads
        scale = 1.0 / jnp.sqrt(head_dim)

        q = nn.Dense(D, use_bias=False)(x).reshape((B, N, self.num_heads, head_dim))
        k = nn.Dense(D, use_bias=False)(x).reshape((B, N, self.num_heads, head_dim))
        v = nn.Dense(D, use_bias=False)(x).reshape((B, N, self.num_heads, head_dim))

        q = jnp.transpose(q, (0, 2, 1, 3)) * scale
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        num_blocks = (N + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - N
        
        if pad_len > 0:
            k_padded = jnp.pad(k, ((0,0), (0,0), (0, pad_len), (0,0)))
            v_padded = jnp.pad(v, ((0,0), (0,0), (0, pad_len), (0,0)))
            if mask is not None:
                mask_padded = jnp.pad(mask, ((0,0), (0, pad_len)), constant_values=False)
            else:
                mask_padded = None
        else:
            k_padded, v_padded = k, v
            mask_padded = mask

        k_tiles = jnp.reshape(k_padded, (B, self.num_heads, num_blocks, self.block_size, head_dim))
        v_tiles = jnp.reshape(v_padded, (B, self.num_heads, num_blocks, self.block_size, head_dim))
        k_tiles = jnp.transpose(k_tiles, (2, 0, 1, 3, 4))
        v_tiles = jnp.transpose(v_tiles, (2, 0, 1, 3, 4))

        if mask_padded is not None:
            mask_tiles = jnp.reshape(mask_padded, (B, num_blocks, self.block_size))
            mask_tiles = jnp.transpose(mask_tiles, (1, 0, 2))[:, :, None, None, :]
        else:
            mask_tiles = None

        def tile_step(carry, xs):
            out, m_prev, l_prev = carry
            
            if mask_tiles is not None:
                k_block, v_block, mask_block = xs
            else:
                k_block, v_block = xs
                mask_block = None

            S_tile = jnp.matmul(q, jnp.transpose(k_block, (0, 1, 3, 2)))

            if mask_block is not None:
                S_tile = jnp.where(mask_block, S_tile, -1e9)

            m_curr = jnp.maximum(m_prev, jnp.max(S_tile, axis=-1, keepdims=True))
            p_tile = jnp.exp(S_tile - m_curr)
            
            l_curr = jnp.exp(m_prev - m_curr) * l_prev + jnp.sum(p_tile, axis=-1, keepdims=True)

            v_out = jnp.matmul(p_tile, v_block)
            out_curr = (jnp.exp(m_prev - m_curr) * l_prev * out + v_out) / jnp.maximum(1e-6, l_curr)

            return (out_curr, m_curr, l_curr), None

        init_out = jnp.zeros_like(q)
        init_m = jnp.full((B, self.num_heads, N, 1), -1e9)
        init_l = jnp.zeros((B, self.num_heads, N, 1))

        scan_xs = (k_tiles, v_tiles, mask_tiles) if mask_tiles is not None else (k_tiles, v_tiles)

        (attn_out, _, _), _ = jax.lax.scan(tile_step, (init_out, init_m, init_l), scan_xs)

        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape((B, N, D))
        return nn.Dense(D)(attn_out)


# -----------------------------------------------------------------------------
# 3. High-Capacity Infinite-Width Architecture (d_model=2048)
# -----------------------------------------------------------------------------

class InfiniteWidthDense(nn.Module):
    features: int
    std_w: float = 1.0
    std_b: float = 0.1

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        w_scale = self.std_w / jnp.sqrt(in_dim)
        b_scale = self.std_b

        w_init = nn.initializers.normal(stddev=1.0)
        b_init = nn.initializers.normal(stddev=1.0)

        w = self.param('kernel', w_init, (in_dim, self.features)) * w_scale
        b = self.param('bias', b_init, (self.features,)) * b_scale

        return jnp.dot(x, w) + b


class RandomFourierKernelProjection(nn.Module):
    num_fourier_features: int = 4096
    gamma: float = 10.0

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        omega = self.variable(
            'fourier', 'omega',
            lambda: jax.random.normal(jax.random.PRNGKey(0), (in_dim, self.num_fourier_features)) * self.gamma
        ).value
        
        proj = jnp.matmul(x, omega)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1) * jnp.sqrt(2.0 / self.num_fourier_features)


class SinusoidalPosEmb(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class InfiniteWidthSVGDiffusionTransformer(nn.Module):
    d_model: int = 2048
    num_heads: int = 32
    num_layers: int = 4
    cmd_dim: int = CMD_DIM

    @nn.compact
    def __call__(self, x_t, timesteps, mask=None):
        x_kernel = RandomFourierKernelProjection(num_fourier_features=self.d_model)(x_t)
        
        t_emb = SinusoidalPosEmb(self.d_model)(timesteps)
        t_emb = InfiniteWidthDense(self.d_model)(t_emb)
        t_emb = nn.relu(t_emb)
        t_emb = InfiniteWidthDense(self.d_model)(t_emb)[:, None, :]

        h = InfiniteWidthDense(self.d_model)(x_kernel) + t_emb

        for _ in range(self.num_layers):
            norm_h = nn.LayerNorm()(h)
            attn = FlashSelfAttention(
                num_heads=self.num_heads,
                d_model=self.d_model,
                block_size=64
            )(norm_h, mask=mask)
            h = h + attn
            
            norm_h2 = nn.LayerNorm()(h)
            mlp_h = InfiniteWidthDense(self.d_model * 4)(norm_h2)
            mlp_h = nn.relu(mlp_h)
            mlp_out = InfiniteWidthDense(self.d_model)(mlp_h)
            h = h + mlp_out
            
        pred_velocity = InfiniteWidthDense(self.cmd_dim)(nn.LayerNorm()(h))
        return pred_velocity


model_instance = InfiniteWidthSVGDiffusionTransformer()
optimizer = optax.adam(learning_rate=2e-4)


# -----------------------------------------------------------------------------
# 4. Anti-Rank Collapse Loss & Trigonometric Non-Linear Trajectory
# -----------------------------------------------------------------------------

def compute_ssim_loss(x_true, x_pred, mask, c1=0.01**2, c2=0.03**2):
    m = mask[:, :, None]
    x = x_true * m
    y = x_pred * m

    mu_x = jnp.mean(x, axis=1, keepdims=True)
    mu_y = jnp.mean(y, axis=1, keepdims=True)

    sigma_x = jnp.var(x, axis=1, keepdims=True)
    sigma_y = jnp.var(y, axis=1, keepdims=True)
    sigma_xy = jnp.mean((x - mu_x) * (y - mu_y), axis=1, keepdims=True)

    ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    
    ssim_map = ssim_num / jnp.maximum(1e-5, ssim_den)
    ssim_val = jnp.mean(ssim_map)
    
    return jnp.clip((1.0 - ssim_val) / 2.0, 0.0, 1.0)


def compute_inductive_bias_losses(pred_v, x_interp, mask):
    m = mask[:, :, None]
    x_pred_1 = x_interp + pred_v
    
    xy = x_pred_1[:, :, 1:3]
    curr_pts = xy[:, :-1, :]
    next_pts = xy[:, 1:, :]
    continuity_err = jnp.mean((curr_pts - next_pts)**2 * m[:, 1:, :])

    cmd_types = x_pred_1[:, :, 0]
    valid_cmds = jnp.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    cmd_distances = jnp.abs(cmd_types[:, :, None] - valid_cmds[None, None, :])
    cmd_quant_loss = jnp.mean(jnp.min(cmd_distances, axis=-1) * mask)

    ctrl_pts = x_pred_1[:, :, 3:7]
    ctrl_out_of_bounds = jnp.maximum(0.0, 0.05 - ctrl_pts) + jnp.maximum(0.0, ctrl_pts - 0.95)
    ctrl_loss = jnp.mean(ctrl_out_of_bounds**2 * m)

    return 0.05 * continuity_err + 0.02 * cmd_quant_loss + 0.01 * ctrl_loss


def inter_item_flow_loss_fn(p, x_start, x_target, t_float, m):
    t_expanded = t_float[:, None, None]
    
    # 1. Trigonometric Non-Linear Interpolation Trajectory (Prevents Linear Vector Collapse)
    angle = (np.pi / 2.0) * t_expanded
    x_interpolated = jnp.sin(np.pi / 2.0 - angle) * x_start + jnp.sin(angle) * x_target
    target_velocity = (np.pi / 2.0) * (jnp.cos(angle) * x_target - jnp.sin(angle) * x_start)
    
    pred_velocity = model_instance.apply(p, x_interpolated, t_float, mask=m)
    
    sq_err = (target_velocity - pred_velocity) ** 2
    m_expanded = m[:, :, None]
    total_valid = jnp.maximum(1.0, jnp.sum(m_expanded) * CMD_DIM)
    mse_loss = jnp.sum(sq_err * m_expanded) / total_valid

    ssim_loss = compute_ssim_loss(target_velocity, pred_velocity, m)
    inductive_loss = compute_inductive_bias_losses(pred_velocity, x_interpolated, m)

    # 2. Covariance De-correlation Loss (Anti-Rank Collapse Penalty)
    flat_pred = jnp.reshape(pred_velocity * m_expanded, (-1, CMD_DIM))
    pred_centered = flat_pred - jnp.mean(flat_pred, axis=0, keepdims=True)
    cov_matrix = jnp.dot(pred_centered.T, pred_centered) / jnp.maximum(1.0, flat_pred.shape[0] - 1.0)
    off_diag_penalty = jnp.sum((cov_matrix * (1.0 - jnp.eye(CMD_DIM))) ** 2)

    total_loss = mse_loss + 0.1 * ssim_loss + inductive_loss + 0.05 * off_diag_penalty

    # Telemetry Breakdown
    high_t_mask = (t_float >= 0.5)[:, None, None] * m_expanded
    low_t_mask = (t_float < 0.5)[:, None, None] * m_expanded
    
    high_t_mse = jnp.sum(sq_err * high_t_mask) / jnp.maximum(1.0, jnp.sum(high_t_mask) * CMD_DIM)
    low_t_mse = jnp.sum(sq_err * low_t_mask) / jnp.maximum(1.0, jnp.sum(low_t_mask) * CMD_DIM)

    dot_prod = jnp.sum(target_velocity * pred_velocity * m_expanded, axis=-1)
    norm_target = jnp.sqrt(jnp.sum(target_velocity**2 * m_expanded, axis=-1) + 1e-6)
    norm_pred = jnp.sqrt(jnp.sum(pred_velocity**2 * m_expanded, axis=-1) + 1e-6)
    cos_sim = jnp.mean(dot_prod / (norm_target * norm_pred + 1e-6))

    spatial_coords = x_interpolated[:, :, 1:3]
    coord_var = jnp.mean(jnp.var(spatial_coords, axis=1))

    # Effective Rank SVD Calculation
    s_vals = jnp.linalg.svd(cov_matrix, compute_uv=False)
    p_s = s_vals / (jnp.sum(s_vals) + 1e-6)
    effective_rank = jnp.exp(-jnp.sum(p_s * jnp.log(p_s + 1e-6)))
    attn_entropy = -jnp.sum(p_s * jnp.log(p_s + 1e-6))

    gram_target = jnp.dot(jnp.reshape(target_velocity, (target_velocity.shape[0], -1)), 
                          jnp.reshape(target_velocity, (target_velocity.shape[0], -1)).T)
    gram_pred = jnp.dot(jnp.reshape(pred_velocity, (pred_velocity.shape[0], -1)), 
                        jnp.reshape(pred_velocity, (pred_velocity.shape[0], -1)).T)
    
    tka_num = jnp.sum(gram_target * gram_pred)
    tka_den = jnp.sqrt(jnp.sum(gram_target**2) * jnp.sum(gram_pred**2) + 1e-6)
    kernel_alignment = tka_num / tka_den

    aux_metrics = (mse_loss, ssim_loss, high_t_mse, low_t_mse, cos_sim, coord_var, effective_rank, kernel_alignment, attn_entropy)
    return total_loss, aux_metrics


@jax.jit
def train_step(p, opt_s, x_start, x_target, t_float, m):
    (loss, aux), grads = jax.value_and_grad(inter_item_flow_loss_fn, has_aux=True)(p, x_start, x_target, t_float, m)
    updates, new_opt_s = optimizer.update(grads, opt_s, p)
    new_p = optax.apply_updates(p, updates)
    return new_p, new_opt_s, loss, aux


def sample_inter_item_trajectory(params, x_source, num_steps=20):
    x_t = jnp.array(x_source)
    dt = 1.0 / num_steps
    mask = jnp.ones((1, x_source.shape[1]), dtype=bool)

    for i in range(num_steps):
        t_val = jnp.array([i * dt], dtype=jnp.float32)
        pred_velocity = model_instance.apply(params, x_t, t_val, mask=mask)
        x_t = x_t + pred_velocity * dt
        x_t = x_t.at[:, :, 1:7].set(jnp.clip(x_t[:, :, 1:7], 0.05, 0.95))

    return np.array(x_t[0])


# -----------------------------------------------------------------------------
# 5. Telemetry & Execution Endpoints
# -----------------------------------------------------------------------------

@app.get("/metrics")
@app.get("/metrics/")
def get_metrics():
    return JSONResponse(content=training_metrics)


@app.get("/generate", response_class=PlainTextResponse)
@app.get("/generate/", response_class=PlainTextResponse)
def generate_svg_endpoint(response: Response):
    try:
        input_dir = "/app/output"
        _, _, training_vectors = compute_dataset_statistics(input_dir)
        
        with jax_lock:
            if not MASTER_CKPT_PATH.exists() or len(training_vectors) == 0:
                response.headers["X-Generator-Mode"] = "random-noise"
                key = jax.random.PRNGKey(int(time.time()))
                seq_len = 32
                params = model_instance.init(key, jnp.zeros((1, seq_len, CMD_DIM)), jnp.zeros((1,), dtype=jnp.float32), jnp.ones((1, seq_len), dtype=bool))
                source_batch = jnp.full((1, seq_len, CMD_DIM), 0.5, dtype=jnp.float32)
            else:
                response.headers["X-Generator-Mode"] = "flash-inter-item-flow"
                with open(MASTER_CKPT_PATH, "rb") as f:
                    payload = pickle.load(f)
                    params = payload["params"]
                
                batch_tensor, _ = collate_svg_batch(training_vectors)
                source_batch = jnp.array(batch_tensor[:1])

            generated_matrix = sample_inter_item_trajectory(params, source_batch, num_steps=50)
            svg_output = vector_sequence_to_svg(generated_matrix)
            
        return svg_output
    except Exception as e:
        response.status_code = 500
        return f"<div class='text-red-400'>Internal server error: {str(e)}</div>"


def compute_dataset_statistics(input_dir):
    svg_files = list(Path(input_dir).glob("*.[sS][vV][gG]")) + list(Path(input_dir).glob("*.svg"))
    unique_svgs = list({f.resolve(): f for f in svg_files}.values())
    
    all_vectors = []
    for svg_file in unique_svgs:
        try:
            svg_text = svg_file.read_text(encoding="utf-8")
            vecs = svg_to_vector_sequence(svg_text)
            if len(vecs) > 0:
                all_vectors.append(vecs)
        except Exception:
            pass

    if not all_vectors:
        mean_vec = np.array([[CMD_MOVE, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2]], dtype=np.float32)
        cov_matrix = np.eye(CMD_DIM, dtype=np.float32) * 0.05
        return mean_vec, cov_matrix, [mean_vec[0]]

    concat_cmds = np.concatenate(all_vectors, axis=0)
    cov_matrix = np.cov(concat_cmds, rowvar=False) + 1e-4 * np.eye(CMD_DIM)
    
    return None, cov_matrix, all_vectors


def compute_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_processed_hashes(checkpoint_dir):
    hash_file = Path(checkpoint_dir) / "processed_hashes.pkl"
    if hash_file.exists():
        try:
            with open(hash_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return set()


def save_processed_hashes(checkpoint_dir, hashes):
    hash_file = Path(checkpoint_dir) / "processed_hashes.pkl"
    tmp_file = hash_file.with_suffix(".tmp")
    with open(tmp_file, "wb") as f:
        pickle.dump(hashes, f)
    tmp_file.replace(hash_file)


def train_on_svg(svg_path, checkpoint_dir, input_dir):
    global training_metrics
    filename = Path(svg_path).name
    print(f"[*] Extracting vector features for Anti-Rank-Collapse Flow Diffusion: {filename}")
    
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    new_vec_data = svg_to_vector_sequence(svg_text)
    
    _, _, all_dataset_vectors = compute_dataset_statistics(input_dir)
    buffer_vectors = all_dataset_vectors[-REPLAY_BUFFER_LIMIT:]
    if not any(np.array_equal(new_vec_data, v) for v in buffer_vectors):
        buffer_vectors.append(new_vec_data)

    x_batch, mask_batch = collate_svg_batch(buffer_vectors)
    B_size = x_batch.shape[0]
    seq_len = new_vec_data.shape[0]

    key = jax.random.PRNGKey(42)

    with jax_lock:
        if MASTER_CKPT_PATH.exists():
            with open(MASTER_CKPT_PATH, "rb") as f:
                payload = pickle.load(f)
                params = payload["params"]
                opt_state = payload.get("opt_state", optimizer.init(params))
        else:
            dummy_t = jnp.zeros((B_size,), dtype=jnp.float32)
            params = model_instance.init(key, jnp.array(x_batch), dummy_t, jnp.array(mask_batch))
            opt_state = optimizer.init(params)

    training_metrics.update({
        "status": "Training (Flash JIT Compiling)",
        "current_file": filename,
        "dataset_count": len(all_dataset_vectors),
        "buffer_size": B_size,
        "sequence_length": seq_len,
        "mode": "Anti-Rank-Collapse Flow Diffusion"
    })

    x_batch_jnp = jnp.array(x_batch)
    mask_jnp = jnp.array(mask_batch)

    # Warmup Step Outside Lock
    print(f"[*] Warming up FlashAttention XLA JIT tracing for N={seq_len}...")
    key, subkey1 = jax.random.split(key)
    idx_start = np.random.choice(B_size, size=B_size)
    idx_target = np.random.choice(B_size, size=B_size)
    t_rand = jax.random.uniform(subkey1, shape=(B_size,), minval=0.0, maxval=1.0)
    
    params, opt_state, loss_val, aux = train_step(
        params, opt_state, x_batch_jnp[idx_start], x_batch_jnp[idx_target], t_rand, mask_jnp
    )
    print(f"[+] Flash JIT Tracing complete. Starting active fine-tuning loop...")

    training_metrics["status"] = "Training"
    epoch = 1
    
    while float(loss_val) > 0.005 and epoch < 1000:
        key, subkey1 = jax.random.split(key)
        
        idx_start = np.random.choice(B_size, size=B_size)
        idx_target = np.random.choice(B_size, size=B_size)
        t_rand = jax.random.uniform(subkey1, shape=(B_size,), minval=0.0, maxval=1.0)

        with jax_lock:
            params, opt_state, loss_val, aux = train_step(
                params, opt_state, x_batch_jnp[idx_start], x_batch_jnp[idx_target], t_rand, mask_jnp
            )
            
        epoch += 1

        if epoch % 20 == 0:
            (mse_val, ssim_val, high_t_mse, low_t_mse, cos_sim, coord_var, rank_val, tka_val, entropy_val) = aux
            
            sample_matrix = sample_inter_item_trajectory(params, x_batch_jnp[:1], num_steps=20)
            active_svg_str = vector_sequence_to_svg(sample_matrix)

            training_metrics.update({
                "epoch": epoch,
                "mse_loss": round(float(mse_val), 6),
                "ssim_loss": round(float(ssim_val), 6),
                "combined_loss": round(float(loss_val), 6),
                "high_t_mse": round(float(high_t_mse), 6),
                "low_t_mse": round(float(low_t_mse), 6),
                "velocity_cosine_sim": round(float(cos_sim), 6),
                "spatial_coord_var": round(float(coord_var), 6),
                "effective_rank": round(float(rank_val), 4),
                "kernel_alignment": round(float(tka_val), 6),
                "attention_entropy": round(float(entropy_val), 4),
                "active_svg": active_svg_str
            })

    ckpt_payload = {"params": params, "opt_state": opt_state, "seq_len": seq_len}
    tmp_master_path = MASTER_CKPT_PATH.with_suffix(".tmp")
    with jax_lock:
        with open(tmp_master_path, "wb") as f:
            pickle.dump(ckpt_payload, f)
        tmp_master_path.replace(MASTER_CKPT_PATH)

    training_metrics.update({
        "status": "Idle",
        "current_file": "None",
        "epoch": epoch,
        "mode": "Master Saved"
    })


def main(input_dir, checkpoint_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processed_hashes = load_processed_hashes(checkpoint_dir)
    print(f"[*] Anti-Rank-Collapse Flow Daemon watching: {input_dir}")

    while True:
        svg_files = list(Path(input_dir).glob("*.[sS][vV][gG]")) + list(Path(input_dir).glob("*.svg"))
        unique_svgs = list({f.resolve(): f for f in svg_files}.values())
        
        for svg_file in unique_svgs:
            if svg_file.is_file():
                file_hash = compute_hash(svg_file)
                if file_hash not in processed_hashes:
                    try:
                        train_on_svg(svg_file, checkpoint_dir, input_dir)
                        processed_hashes.add(file_hash)
                        save_processed_hashes(checkpoint_dir, processed_hashes)
                    except Exception as e:
                        print(f"[!] Error processing {svg_file.name}: {e}")
        
        time.sleep(3)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/app/output")
    parser.add_argument("--checkpoint_dir", default="/app/checkpoints")
    args = parser.parse_args()
    
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"), 
        daemon=True
    )
    api_thread.start()

    main(args.input_dir, args.checkpoint_dir)