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

MAX_PATHS = 16       # Fixed maximum number of SVG path primitives per file
ATTR_DIM = 8         # Features: [is_circle, x, y, r, stroke_width, red, green, blue]


# -----------------------------------------------------------------------------
# 1. SVG Parsing & Vector Serialization
# -----------------------------------------------------------------------------

def parse_color(color_str):
    """Converts hex or basic string color into normalized [R, G, B] floats (0-1)."""
    if not color_str or color_str == "none":
        return [0.0, 0.0, 0.0]
    color_str = color_str.strip().lower()
    if color_str.startswith("#"):
        hex_str = color_str.lstrip("#")
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        if len(hex_str) == 6:
            return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return [0.2, 0.5, 0.8]  # Fallback default tint


def svg_to_vector(svg_text, max_paths=MAX_PATHS):
    """Parses SVG XML tags into a normalized numpy array of continuous vector features."""
    vectors = []
    try:
        clean_text = re.sub(r'xmlns="[^"]+"', '', svg_text)
        root = ET.fromstring(clean_text)
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            if tag == 'circle':
                cx = float(elem.get('cx', 50)) / 100.0
                cy = float(elem.get('cy', 50)) / 100.0
                r = float(elem.get('r', 10)) / 100.0
                sw = float(elem.get('stroke-width', 2)) / 10.0
                rgb = parse_color(elem.get('stroke', elem.get('fill', '#000000')))
                vectors.append([1.0, cx, cy, r, sw, rgb[0], rgb[1], rgb[2]])
            elif tag == 'rect':
                x = (float(elem.get('x', 0)) + float(elem.get('width', 20)) / 2) / 100.0
                y = (float(elem.get('y', 0)) + float(elem.get('height', 20)) / 2) / 100.0
                r = float(elem.get('width', 20)) / 200.0
                sw = float(elem.get('stroke-width', 2)) / 10.0
                rgb = parse_color(elem.get('stroke', elem.get('fill', '#000000')))
                vectors.append([0.0, x, y, r, sw, rgb[0], rgb[1], rgb[2]])
    except Exception as e:
        print(f"[!] XML Parsing error: {e}")

    if not vectors:
        vectors = [[1.0, 0.5, 0.5, 0.2, 0.2, 0.1, 0.5, 0.9]]

    vec_arr = np.array(vectors, dtype=np.float32)
    if len(vec_arr) < max_paths:
        pad_len = max_paths - len(vec_arr)
        padding = np.zeros((pad_len, ATTR_DIM), dtype=np.float32)
        vec_arr = np.vstack([vec_arr, padding])
    else:
        vec_arr = vec_arr[:max_paths]

    return vec_arr


def vector_to_svg(vector_arr):
    """Reconstructs valid SVG XML string from continuous vector array predictions."""
    svg_elements = ['<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">']
    
    for vec in vector_arr:
        is_circle, x, y, r, sw, red, green, blue = vec
        
        if r < 0.01:
            continue
            
        cx, cy = np.clip(x * 100.0, 0, 100), np.clip(y * 100.0, 0, 100)
        radius = np.clip(r * 100.0, 1, 50)
        stroke_w = np.clip(sw * 10.0, 0.5, 10)
        
        r_byte = int(np.clip(red * 255, 0, 255))
        g_byte = int(np.clip(green * 255, 0, 255))
        b_byte = int(np.clip(blue * 255, 0, 255))
        color_hex = f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}"
        
        if is_circle > 0.5:
            elem = f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="none" stroke="{color_hex}" stroke-width="{stroke_w:.1f}" />'
        else:
            w, h = radius * 1.5, radius * 1.5
            elem = f'  <rect x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{color_hex}" stroke-width="{stroke_w:.1f}" />'
        svg_elements.append(elem)
        
    svg_elements.append('</svg>')
    return "\n".join(svg_elements).strip()


# -----------------------------------------------------------------------------
# 2. Diffusion Transformer Architecture & Noise Schedule
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


class VectorDiffusionTransformer(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    attr_dim: int = ATTR_DIM

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
            
        pred_eps = nn.Dense(self.attr_dim)(nn.LayerNorm()(h))
        return pred_eps


class LinearNoiseSchedule:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Convert schedule arrays to JAX DeviceArrays to support JIT dynamic indexing
        self.jnp_sqrt_alphas_cumprod = jnp.array(self.sqrt_alphas_cumprod)
        self.jnp_sqrt_one_minus_alphas_cumprod = jnp.array(self.sqrt_one_minus_alphas_cumprod)

schedule = LinearNoiseSchedule()


# -----------------------------------------------------------------------------
# 3. Training & Sampling Functions
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


def train_on_svg(svg_path, checkpoint_dir):
    print(f"[*] Processing SVG vector features for diffusion training: {svg_path}")
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    vec_data = svg_to_vector(svg_text)
    x_0 = vec_data[None, ...]

    model = VectorDiffusionTransformer()
    key = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros_like(x_0)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    
    with jax_lock:
        params = model.init(key, dummy_x, dummy_t)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def diffusion_loss_fn(p, x_start, t, noise):
        # Dynamically index JAX device arrays
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
    
    with jax_lock:
        with open(tmp_ckpt_path, "wb") as f:
            pickle.dump(params, f)
        tmp_ckpt_path.replace(ckpt_path)
        
    print(f"[*] Diffusion checkpoint saved: {ckpt_path} (Final MSE: {loss_val_py:.6f})")


def sample_diffusion_from_noise(params, rng_key):
    """Runs reverse diffusion steps starting from pure Gaussian noise to synthesize blended SVGs."""
    model = VectorDiffusionTransformer()
    
    rng_key, subkey = jax.random.split(rng_key)
    x_t = jax.random.normal(subkey, (1, MAX_PATHS, ATTR_DIM))

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
# 4. FastAPI Service & Daemon Loop
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
                model = VectorDiffusionTransformer()
                key = jax.random.PRNGKey(int(time.time()))
                dummy_x = jnp.zeros((1, MAX_PATHS, ATTR_DIM))
                dummy_t = jnp.zeros((1,), dtype=jnp.int32)
                params = model.init(key, dummy_x, dummy_t)
            else:
                latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                response.headers["X-Generator-Mode"] = "diffusion-blended"
                with open(latest_ckpt, "rb") as f:
                    params = pickle.load(f)

            seed = int(time.time() * 1000) % 100000
            generated_vectors = sample_diffusion_from_noise(params, jax.random.PRNGKey(seed))
            svg_output = vector_to_svg(generated_vectors)
            
        print(f"[+] Successfully generated synthesized SVG ({len(svg_output)} chars).")
        return svg_output
    except Exception as e:
        print(f"[!] Error in /generate endpoint: {e}")
        response.status_code = 500
        return f"<div class='text-red-400'>Internal server error: {str(e)}</div>"


def main(input_dir, checkpoint_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processed_hashes = load_processed_hashes(checkpoint_dir)
    print(f"[*] Vector Diffusion Daemon started. Watching directory: {input_dir}")

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