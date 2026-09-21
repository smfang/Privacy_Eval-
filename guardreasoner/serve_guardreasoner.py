"""
serve_guardreasoner.py — Modal + vLLM serving recipe for GuardReasoner-Omni,
adapted from the cope-evaluation guide's serve_cope.py.

This targets the Omni line (zhu-thu-22/GuardReasoner-Omni-3B/7B, Qwen2.5-Omni
based, multimodal) — NOT the original text-only GuardReasoner (yueliu1999/
GuardReasoner-3B/8B/1B, Llama-3.2 based). See guardreasoner_adapter.py for why
the prompt template differs between the two lines.

vLLM support: confirmed in vllm's model registry (vllm/model_executor/models/
registry.py) that a checkpoint whose config.json declares
`architectures: ["Qwen2_5OmniForConditionalGeneration"]` (which is what
GuardReasoner-Omni-3B/7B ship) is auto-mapped to vLLM's supported
`Qwen2_5OmniThinkerForConditionalGeneration` implementation — no --hf-overrides
needed. You do need a vLLM release new enough to have that registry entry; if
`vllm serve` fails to recognize the architecture, upgrade vLLM first.

This eval only sends TEXT (no image/video/audio), which is Omni's default,
most-tested path — see generate.py's text_benchmark_list loop, which builds
messages with no multimodal content block at all.

Per the guide, adapting to a different model = change MODEL_NAME, GPU_CONFIG,
MAX_MODEL_LEN, the app name, and the secret name. Everything else mirrors it.

Commands (new app/secret name — distinct from any base-GuardReasoner deploy):
    uv tool install modal                                # one-time
    modal token new                                       # one-time
    modal secret create guardreasoner-omni-secrets HF_TOKEN=hf_xxx VLLM_API_KEY=sk-pick-something
    modal run serve_guardreasoner.py::download_model       # optional pre-cache (guide Step 6)
    modal deploy serve_guardreasoner.py                    # prints the public URL (Step 7)
    #   ... point your eval script at <URL>/v1/chat/completions ...
    modal app stop guardreasoner-omni                      # when done (Step 9)

NOTE: GuardReasoner-Omni is openly released, so the guide's Step 1 ("Agree and
access" for the gated cope model) does NOT apply. An HF token isn't strictly
required, but keeping it in the secret is harmless.
"""
import os
import subprocess

import modal

# ---- the only bits you change per model (per the guide) -------------------
MODEL_NAME = "zhu-thu-22/GuardReasoner-Omni-3B"   # Qwen2.5-Omni-3B based. Alternative:
#   "zhu-thu-22/GuardReasoner-Omni-7B"   # best accuracy, needs more GPU memory
MODEL_REVISION = None      # set to a commit SHA to PIN for reproducibility
GPU_CONFIG = "L40S:1"      # 3B Omni (LM + vision/audio towers) fits L40S/A100 for text-only use.
                            # Bump to A100/H100 for -7B or if you turn on image/video/audio.
MAX_MODEL_LEN = 24576      # matches the model card's own usage example (room for long <think> traces)
APP_NAME = "guardreasoner-omni"
SECRET_NAME = "guardreasoner-omni-secrets"
VLLM_VERSION = "vllm"      # unpinned: needs a release with Qwen2_5OmniThinkerForConditionalGeneration
                            # support (confirmed present on vllm-project/vllm main as of 2026-08-12).
                            # Pin to a specific version once you've confirmed it serves this model.
# ---------------------------------------------------------------------------

# Same base image as serve_cope.py's cope-b-a4b recipe (proven to deploy on Modal) —
# reusing it means the model swap is the only unknown, not the image too.
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
    """Optional one-time pre-cache of the weights onto the persistent volume (guide Step 6)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    hf_cache.commit()
    print(f"cached {MODEL_NAME}")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=10 * 60,   # spin down after 10 min idle (guide's behavior)
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
        "--trust-remote-code",                              # Qwen2.5-Omni custom processor/config
        "--limit-mm-per-prompt", '{"image":1,"video":1,"audio":1}',  # matches model card; harmless for text-only
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    subprocess.Popen(cmd)
