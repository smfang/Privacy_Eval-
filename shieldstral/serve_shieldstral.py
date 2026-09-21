"""
serve_shieldstral.py — Modal + vLLM serving recipe for mistralai/Shieldstral-1.0-3B.

VERIFIED against the model card on 2026-09-02:
  https://huggingface.co/mistralai/Shieldstral-1.0/raw/main/README.md

The card's own recommended recipe:
    pip install vllm --upgrade   # needs vllm>=0.26.0, pulls in mistral_common>=1.11.5
    vllm serve mistralai/Shieldstral-1.0-3B --max-model-len 32768

"Shieldstral can be deployed locally, as it fits in 16GB of VRAM in BF16" — a
meaningfully smaller footprint than GuardReasoner-Omni-3B (which needed
L40S/24576 max-len for its vision/audio towers plus long <think> traces).
Shieldstral emits a single yes/no token with no reasoning chain, so both the
weights and the per-call generation are cheaper here. L4 (24GB) has comfortable
headroom; go bigger only if you raise --max-num-seqs a lot.

Commands (new app/secret name, distinct from the guardreasoner-omni deploy):
    modal secret create shieldstral-secrets VLLM_API_KEY=sk-pick-something
    modal run serve_shieldstral.py::download_model    # optional pre-cache
    modal deploy serve_shieldstral.py                 # prints the public URL
    modal app stop shieldstral                        # when done

Shieldstral is openly released (Apache 2.0), so no HF token is required.
"""
import os
import subprocess

import modal

MODEL_NAME = "mistralai/Shieldstral-1.0-3B"
GPU_CONFIG = "L4:1"        # 16GB bf16 weights fit comfortably per the model card
MAX_MODEL_LEN = 32768      # the model card's own recommended --max-model-len
APP_NAME = "shieldstral"
SECRET_NAME = "shieldstral-secrets"
# vllm>=0.26.0 required per the model card (pulls in mistral_common>=1.11.5 automatically)
VLLM_VERSION = "vllm>=0.26.0"

# Same base image family as serve_guardreasoner.py / serve_cope.py — proven to
# deploy on Modal, so a new model is the only unknown, not the image too.
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(VLLM_VERSION, "hf-transfer", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=30 * 60,
)
def download_model():
    """Optional one-time pre-cache of the weights onto the persistent volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    hf_cache.commit()
    print(f"cached {MODEL_NAME}")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=10 * 60,   # spin down after 10 min idle
    timeout=60 * 60,
)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", MODEL_NAME,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
