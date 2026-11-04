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
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Global model state lock for thread safety
jax_lock = threading.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CMD_DIM = 10  # Features: [cmd_type, x, y, cx1, cy1, cx2, cy2, stroke_width, red, green]

# Command Types
CMD_PAD = 0.0
CMD_MOVE = 1.0   # M x y
CMD_LINE = 2.0   # L x y
CMD_CUBIC = 3.0  # C cx1 cy1 cx2 cy2 x y


# -----------------------------------------------------------------------------
# 1. Fully Dynamic SVG Path Parsing & Serialization
# -----------------------------------------------------------------------------

def parse_color(color_str):
    """Parses color strings or defaults to black [0, 0, 0]."""
    if not color_str or color_str == "none":
        return [0.0, 0.0, 0.0]
    color_str = color_str.strip().lower()
    if color_str.startswith("#"):
        hex_str = color_str.lstrip("#")
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        if len(hex_str) == 6:
            return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return [0.0, 0.0, 0.0]


def parse_svg_path_d(d_str, stroke_w=2.0, color=(0.0, 0.0, 0.0)):
    """Parses arbitrary SVG path string 'd' attribute into continuous command vectors."""
    commands = []
    tokens = re.findall(r'([A-Za-z])|(-?\d*\.?\d+(?:e[-+]?\d+)?)', d_str)
    
    current_cmd = None
    args = []
    
    def process_command(cmd, coords):
        c_type = CMD_PAD
        x = y = cx1 = cy1 = cx2 = cy2 = 0.5
        
        if cmd in ('M', 'm'):
            c_type = CMD_MOVE
            if len(coords) >= 2:
                x, y = coords[0] / 100.0, coords[1] / 100.0
        elif cmd in ('L', 'l'):
            c_type = CMD_LINE
            if len(coords) >= 2:
                x, y = coords[0] / 100.0, coords[1] / 100.0
        elif cmd in ('C', 'c'):
            c_type = CMD_CUBIC
            if len(coords) >= 6:
                cx1, cy1 = coords[0] / 100.0, coords[1] / 100.0
                cx2, cy2 = coords[2] / 100.0, coords[3] / 100.0
                x, y = coords[4] / 100.0, coords[5] / 100.0
        else:
            return None
            
        r, g = color[0], color[1]
        sw = stroke_w / 10.0
        return [c_type, x, y, cx1, cy1, cx2, cy2, sw, r, g]

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
    """Encodes arbitrary SVG XML into an un-truncated dynamic length sequence of path commands."""
    vectors = []
    try:
        clean_text = re.sub(r'xmlns="[^"]+"', '', svg_text)
        root = ET.fromstring(clean_text)
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            stroke_w = float(elem.get('stroke-width', 2.0))
            rgb = parse_color(elem.get('stroke', elem.get('fill', '#000000')))
            
            if tag == 'path':
                d = elem.get('d', '')
                parsed = parse_svg_path_d(d, stroke_w=stroke_w, color=rgb)
                vectors.extend(parsed)
            elif tag == 'circle':
                cx, cy = float(elem.get('cx', 50)), float(elem.get('cy', 50))
                r = float(elem.get('r', 10))
                vectors.append([CMD_MOVE, (cx + r)/100.0, cy/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0, rgb[0], rgb[1]])
                vectors.append([CMD_CUBIC, (cx - r)/100.0, cy/100.0, (cx + r)/100.0, (cy + r)/100.0, (cx - r)/100.0, (cy + r)/100.0, stroke_w/10.0, rgb[0], rgb[1]])
            elif tag == 'rect':
                x, y = float(elem.get('x', 10)), float(elem.get('y', 10))
                w, h = float(elem.get('width', 20)), float(elem.get('height', 20))
                vectors.append([CMD_MOVE, x/100.0, y/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0, rgb[0], rgb[1]])
                vectors.append([CMD_LINE, (x+w)/100.0, y/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0, rgb[0], rgb[1]])
                vectors.append([CMD_LINE, (x+w)/100.0, (y+h)/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0, rgb[0], rgb[1]])
                vectors.append([CMD_LINE, x/100.0, (y+h)/100.0, 0.5, 0.5, 0.5, 0.5, stroke_w/10.0, rgb[0], rgb[1]])
    except Exception as e:
        print(f"[!] XML Path Parsing error: {e}")

    if not vectors:
        vectors = [[CMD_MOVE, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0],
                   [CMD_LINE, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0]]

    return np.array(vectors, dtype=np.float32)


def vector_sequence_to_svg(cmd_matrix):
    """Reconstructs scalable SVG markup using dynamically predicted attributes."""
    path_d_list = []
    
    avg_sw = np.mean(cmd_matrix[:, 7]) * 10.0 if len(cmd_matrix) > 0 else 2.0
    stroke_width = np.clip(avg_sw, 0.5, 8.0)
    
    avg_r = int(np.clip(np.mean(cmd_matrix[:, 8]) * 255, 0, 255)) if len(cmd_matrix) > 0 else 0
    avg_g = int(np.clip(np.mean(cmd_matrix[:, 9]) * 255, 0, 255)) if len(cmd_matrix) > 0 else 0
    color_hex = f"#{avg_r:02x}{avg_g:02x}00" if (avg_r > 10 or avg_g > 10) else "#000000"

    for vec in cmd_matrix:
        c_type, x, y, cx1, cy1, cx2, cy2, sw, r, g = vec
        
        px, py = np.clip(x * 100.0, 0, 100), np.clip(y * 100.0, 0, 100)
        pcx1, pcy1 = np.clip(cx1 * 100.0, 0, 100), np.clip(cy1 * 100.0, 0, 100)
        pcx2, pcy2 = np.clip(cx2 * 100.0, 0, 100), np.clip(cy2 * 100.0, 0, 100)
        
        if 0.5 < c_type <= 1.5:  # MoveTo
            path_d_list.append(f"M {px:.1f} {py:.1f}")
        elif 1.5 < c_type <= 2.5:  # LineTo
            path_d_list.append(f"L {px:.1f} {py:.1f}")
        elif c_type > 2.5:  # CubicTo
            path_d_list.append(f"C {pcx1:.1f} {pcy1:.1f}, {pcx2:.1f} {pcy2:.1f}, {px:.1f} {py:.1f}")

    d_string = " ".join(path_d_list) if path_d_list else "M 10 10 L 90 90"
    
    return f'''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="{d_string}" fill="none" stroke="{color_hex}" stroke-width="{stroke_width:.1f}" stroke-linecap="round" stroke-linejoin="round" />
</svg>'''.strip()


# -----------------------------------------------------------------------------
# 2. Sequence-Length Agnostic Diffusion Transformer
# -----------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class FlexibleSVGDiffusionTransformer(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    cmd_dim: int = CMD_DIM

    @nn.compact
    def __call__(self, x_t, timesteps):
        t_emb = SinusoidalPosEmb(self.d_model)(timesteps)
        t_emb = nn.Dense(self.d_model)(t_emb)
        t_emb = nn.relu(t_emb)
        t_emb = nn.Dense(self.d_model)(t_emb)[:, None, :]

        h = nn.Dense(self.d_model)(x_t) + t_emb
        
        for _ in range(self.num_layers):
            norm_h = nn.LayerNorm()(h)
            attn = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                use_bias=False
            )(norm_h)
            h = h + attn
            
            norm_h2 = nn.LayerNorm()(h)
            mlp = nn.Sequential([
                nn.Dense(self.d_model * 4),
                nn.relu,
                nn.Dense(self.d_model)
            ])(norm_h2)
            h = h + mlp
            
        pred_eps = nn.Dense(self.cmd_dim)(nn.LayerNorm()(h))
        return pred_eps


class LinearNoiseSchedule:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.jnp_sqrt_alphas_cumprod = jnp.array(self.sqrt_alphas_cumprod)
        self.jnp_sqrt_one_minus_alphas_cumprod = jnp.array(self.sqrt_one_minus_alphas_cumprod)

schedule = LinearNoiseSchedule()


# -----------------------------------------------------------------------------
# 3. Dynamic Training, Parameter Averaging & Reverse Denoising
# -----------------------------------------------------------------------------

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


def load_and_average_checkpoints(checkpoint_files):
    """Loads all model checkpoints, computes mean sequence length, and averages parameter trees."""
    all_params = []
    seq_lens = []

    for ckpt_path in checkpoint_files:
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
            all_params.append(payload["params"])
            seq_lens.append(payload["seq_len"])

    avg_seq_len = int(np.round(np.mean(seq_lens)))

    # Compute unweighted element-wise arithmetic mean across parameter trees
    avg_params = jax.tree.map(
        lambda *args: jnp.mean(jnp.stack(args, axis=0), axis=0),
        *all_params
    )

    return avg_params, avg_seq_len


def train_on_svg(svg_path, checkpoint_dir):
    print(f"[*] Extracting dynamic vector features for diffusion training: {svg_path}")
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    vec_data = svg_to_vector_sequence(svg_text)
    x_0 = vec_data[None, ...]  # Shape: (1, N_strokes, CMD_DIM)
    seq_len = x_0.shape[1]
    print(f"[*] File loaded with dynamic sequence length N = {seq_len} commands.")

    model = FlexibleSVGDiffusionTransformer()
    key = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros_like(x_0)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    
    with jax_lock:
        params = model.init(key, dummy_x, dummy_t)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def diffusion_loss_fn(p, x_start, t, noise):
        sqrt_a = schedule.jnp_sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_a = schedule.jnp_sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        x_noisy = sqrt_a * x_start + sqrt_one_minus_a * noise
        pred_noise = model.apply(p, x_noisy, t)
        loss = jnp.mean((noise - pred_noise) ** 2)
        return loss

    @jax.jit
    def step(p, opt_s, x_start, t, noise):
        loss, grads = jax.value_and_grad(diffusion_loss_fn)(p, x_start, t, noise)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss

    epoch = 0
    loss_val = float('inf')
    
    while loss_val > 1e-4 and epoch < 500:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        t_rand = jax.random.randint(subkey1, (1,), 0, schedule.timesteps)
        noise_rand = jax.random.normal(subkey2, x_0.shape)

        with jax_lock:
            params, opt_state, loss_val = step(params, opt_state, x_0, t_rand, noise_rand)
            
        loss_val_py = float(loss_val)
        if np.isnan(loss_val_py):
            print(f"[!] Loss diverged to NaN at epoch {epoch}.")
            return
        epoch += 1

    ckpt_path = Path(checkpoint_dir) / f"{svg_path.stem}_params.pkl"
    tmp_ckpt_path = ckpt_path.with_suffix(".tmp")
    
    ckpt_payload = {
        "params": params,
        "seq_len": seq_len
    }
    
    with jax_lock:
        with open(tmp_ckpt_path, "wb") as f:
            pickle.dump(ckpt_payload, f)
        tmp_ckpt_path.replace(ckpt_path)
        
    print(f"[*] Diffusion checkpoint saved: {ckpt_path} (N={seq_len}, Final MSE: {loss_val_py:.6f})")


def sample_diffusion_from_noise(params, seq_len, rng_key):
    """Runs reverse diffusion for dynamically sized stroke sequences using aggregated weights."""
    model = FlexibleSVGDiffusionTransformer()
    
    rng_key, subkey = jax.random.split(rng_key)
    x_t = jax.random.normal(subkey, (1, seq_len, CMD_DIM))

    for t in reversed(range(schedule.timesteps)):
        t_array = jnp.array([t], dtype=jnp.int32)
        pred_eps = model.apply(params, x_t, t_array)
        
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alphas_cumprod[t]
        beta_t = schedule.betas[t]
        
        mean = (1.0 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * pred_eps)
        
        if t > 0:
            rng_key, subkey = jax.random.split(rng_key)
            z = jax.random.normal(subkey, x_t.shape)
            x_t = mean + np.sqrt(beta_t) * z
        else:
            x_t = mean

    return np.array(x_t[0])


# -----------------------------------------------------------------------------
# 4. FastAPI Service Endpoint & Execution Loop
# -----------------------------------------------------------------------------

@app.get("/generate", response_class=PlainTextResponse)
def generate_svg_endpoint(response: Response):
    try:
        print("[*] FastAPI received GET /generate request.")
        checkpoint_dir = Path("/app/checkpoints")
        ckpt_files = list(checkpoint_dir.glob("*_params.pkl"))
        
        with jax_lock:
            if not ckpt_files:
                print("[!] No trained checkpoints found. Initializing fallback noise model...")
                response.headers["X-Generator-Mode"] = "random-noise"
                model = FlexibleSVGDiffusionTransformer()
                key = jax.random.PRNGKey(int(time.time()))
                seq_len = 32
                dummy_x = jnp.zeros((1, seq_len, CMD_DIM))
                dummy_t = jnp.zeros((1,), dtype=jnp.int32)
                params = model.init(key, dummy_x, dummy_t)
            else:
                print(f"[*] Aggregating and averaging weights across {len(ckpt_files)} checkpoints...")
                response.headers["X-Generator-Mode"] = "diffusion-aggregated"
                params, seq_len = load_and_average_checkpoints(ckpt_files)

            seed = int(time.time() * 1000) % 100000
            generated_matrix = sample_diffusion_from_noise(params, seq_len, jax.random.PRNGKey(seed))
            svg_output = vector_sequence_to_svg(generated_matrix)
            
        print(f"[+] Successfully generated aggregated SVG ({len(svg_output)} chars, N={seq_len}, Checkpoints={len(ckpt_files)}).")
        return svg_output
    except Exception as e:
        print(f"[!] Error in /generate endpoint: {e}")
        response.status_code = 500
        return f"<div class='text-red-400'>Internal server error: {str(e)}</div>"


def main(input_dir, checkpoint_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processed_hashes = load_processed_hashes(checkpoint_dir)
    print(f"[*] Flexible SVG Diffusion Daemon started. Watching directory: {input_dir}")

    while True:
        svg_files = list(Path(input_dir).glob("*.[sS][vV][gG]")) + list(Path(input_dir).glob("*.svg"))
        unique_svgs = list({f.resolve(): f for f in svg_files}.values())
        
        for svg_file in unique_svgs:
            if svg_file.is_file():
                file_hash = compute_hash(svg_file)
                if file_hash not in processed_hashes:
                    print(f"[*] New SVG detected for vector diffusion: {svg_file.name}")
                    try:
                        train_on_svg(svg_file, checkpoint_dir)
                        processed_hashes.add(file_hash)
                        save_processed_hashes(checkpoint_dir, processed_hashes)
                        print(f"[+] Successfully processed and indexed: {svg_file.name}")
                    except Exception as e:
                        print(f"[!] Error processing {svg_file.name}: {e}")
        
        time.sleep(5)


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
    print("[*] FastAPI server started on port 8000")

    main(args.input_dir, args.checkpoint_dir)