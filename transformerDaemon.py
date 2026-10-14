import os
import time
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CausalTransformer(nn.Module):
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    max_len: int = 512

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        embeddings = nn.Embed(self.vocab_size, self.d_model)(x)
        pos = jnp.arange(seq_len)[None, :]
        pos_emb = nn.Embed(self.max_len, self.d_model)(pos)
        h = embeddings + pos_emb
        
        mask = nn.make_causal_mask(jnp.ones((seq_len,), dtype=bool))
        
        for _ in range(self.num_layers):
            norm_h = nn.LayerNorm()(h)
            attn = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                use_bias=False
            )(norm_h, mask=mask)
            h = h + attn
            
            norm_h2 = nn.LayerNorm()(h)
            mlp = nn.Sequential([
                nn.Dense(self.d_model * 4),
                nn.relu,
                nn.Dense(self.d_model)
            ])(norm_h2)
            h = h + mlp
            
        logits = nn.Dense(self.vocab_size)(nn.LayerNorm()(h))
        return logits

def compute_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_processed_hashes(checkpoint_dir):
    hash_file = Path(checkpoint_dir) / "processed_hashes.pkl"
    if hash_file.exists():
        with open(hash_file, "rb") as f:
            return pickle.load(f)
    return set()

def save_processed_hashes(checkpoint_dir, hashes):
    hash_file = Path(checkpoint_dir) / "processed_hashes.pkl"
    with open(hash_file, "wb") as f:
        pickle.dump(hashes, f)

def train_on_svg(svg_path, checkpoint_dir):
    print(f"[*] Loading SVG for autoregressive training: {svg_path}")
    text = Path(svg_path).read_text(encoding="utf-8")
    tokens = np.array([ord(c) for c in text if 0 <= ord(c) < 256], dtype=np.int32)
    
    if len(tokens) < 2:
        print(f"[!] SVG file too short or invalid: {svg_path}")
        return

    tokens = tokens[:512]
    inputs = tokens[:-1][None, :]
    targets = tokens[1:][None, :]

    model = CausalTransformer()
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, len(tokens) - 1), dtype=np.int32)
    params = model.init(key, dummy_input)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.01),
        optax.adam(learning_rate=1e-5)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(p, x, y):
        logits = model.apply(p, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    @jax.jit
    def step(p, opt_s, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss

    print(f"[*] Training transformer safely for {svg_path.name}...")
    epoch = 0
    loss_val = float('inf')
    
    while loss_val > 1e-3 and epoch < 300:
        params, opt_state, loss_val = step(params, opt_state, inputs, targets)
        if jnp.isnan(loss_val):
            print(f"[!] Loss diverged to NaN at epoch {epoch}. Aborting training.")
            return
        epoch += 1

    ckpt_path = Path(checkpoint_dir) / f"{svg_path.stem}_params.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(params, f)
    print(f"[*] Checkpoint saved to {ckpt_path}")

def generate_svg_fast(params, prompt_tokens, max_len=512):
    model = CausalTransformer()
    pad_len = max_len - len(prompt_tokens)
    initial_tokens = list(prompt_tokens) + [0] * pad_len
    initial_tokens = np.array(initial_tokens, dtype=np.int32)
    
    init_carry = (
        jnp.array(initial_tokens), 
        jnp.array(len(prompt_tokens), dtype=jnp.int32), 
        jax.random.PRNGKey(int(time.time() * 1000))
    )

    def scan_fn(carry, _):
        tokens, cur_len, rng = carry
        
        start_idx = jnp.maximum(0, cur_len - model.max_len)
        x = jax.lax.dynamic_slice(tokens, (start_idx,), (model.max_len,))
        x = x[None, :]
            
        logits = model.apply(params, x)
        # Lower temperature (0.3) to restrict output to valid structural characters
        next_logits = logits[0, cur_len - 1, :] / 0.3
        
        rng, subkey = jax.random.split(rng)
        next_token = jax.random.categorical(subkey, next_logits)
        
        updated_tokens = tokens.at[cur_len].set(next_token.astype(tokens.dtype))
        new_len = jnp.minimum(cur_len + 1, max_len)
        
        return (updated_tokens, new_len, rng), next_token

    print("[*] Executing compiled lax.scan generation graph...")
    (_, final_len, _), sampled_tokens = jax.lax.scan(scan_fn, init_carry, None, length=300)
    
    all_tokens = np.concatenate([prompt_tokens, np.array(sampled_tokens)])
    full_text = "".join([chr(int(t)) for t in all_tokens if 32 <= t < 127 or t in (10, 13, 9)])
    
    start_idx = full_text.find("<svg")
    end_idx = full_text.find("</svg>")
    
    if start_idx != -1 and end_idx != -1:
        return full_text[start_idx:end_idx + 6]
    return full_text

@app.get("/generate", response_class=PlainTextResponse)
def generate_svg_endpoint(response: Response):
    try:
        print("[*] FastAPI received GET /generate request.")
        checkpoint_dir = Path("/app/checkpoints")
        ckpt_files = list(checkpoint_dir.glob("*_params.pkl"))
        
        if not ckpt_files:
            print("[!] No checkpoint files found. Initializing fallback random noise model...")
            response.headers["X-Generator-Mode"] = "random-noise"
            model = CausalTransformer()
            key = jax.random.PRNGKey(int(time.time()))
            dummy_input = jnp.zeros((1, model.max_len), dtype=jnp.int32)
            params = model.init(key, dummy_input)
        else:
            latest_ckpt = max(ckpt_files, key=os.path.getmtime)
            print(f"[*] Loading active model checkpoint: {latest_ckpt.name}")
            response.headers["X-Generator-Mode"] = "trained-checkpoint"
            with open(latest_ckpt, "rb") as f:
                params = pickle.load(f)

        prompt = [ord(c) for c in "<svg viewBox=\"0 0 100 100\">"]
        generated_svg = generate_svg_fast(params, prompt)
        print(f"[+] Successfully generated SVG string ({len(generated_svg)} characters).")
        return generated_svg
    except Exception as e:
        print(f"[!] Error in /generate endpoint: {e}")
        response.status_code = 500
        return f"<div class='text-red-400'>Internal server error: {str(e)}</div>"

def main(input_dir, checkpoint_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processed_hashes = load_processed_hashes(checkpoint_dir)
    print(f"[*] JAX Transformer Daemon started. Watching directory: {input_dir}")

    while True:
        svg_files = list(Path(input_dir).glob("*.[sS][vV][gG]")) + list(Path(input_dir).glob("*.svg"))
        unique_svgs = {f.resolve(): f for f in svg_files}.values()
        
        for svg_file in unique_svgs:
            if svg_file.is_file():
                file_hash = compute_hash(svg_file)
                if file_hash not in processed_hashes:
                    print(f"[*] New unmemorized SVG detected: {svg_file.name}")
                    try:
                        train_on_svg(svg_file, checkpoint_dir)
                        processed_hashes.add(file_hash)
                        save_processed_hashes(checkpoint_dir, processed_hashes)
                        print(f"[+] Successfully finished and recorded hash for: {svg_file.name}")
                    except Exception as e:
                        print(f"[!] Error processing {svg_file.name}: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/app/output")
    parser.add_argument("--checkpoint_dir", default="/app/checkpoints")
    args = parser.parse_args()
    
    api_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"), daemon=True)
    api_thread.start()
    print("[*] FastAPI server started on port 8000")

    main(args.input_dir, args.checkpoint_dir)