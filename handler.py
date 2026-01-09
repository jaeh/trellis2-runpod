"""
TRELLIS.2 RunPod Serverless Handler
Following RunPod best practices:
- Model initialized OUTSIDE handler (loaded once at startup)
- Uses RunPod's built-in model caching
- Progress updates via runpod.serverless.progress_update()

Usage:
    POST /run with JSON body:
    {
        "input": {
            "input_image": "https://example.com/image.png" or "base64...",
            "seed": 42,
            "pipeline_type": "512" | "1024_cascade" | "1536_cascade",
            "texture_size": 1024,
            "decimation_target": 20000
        }
    }
"""

import runpod
import os
import sys
import base64
import tempfile
import time
import torch
import requests
from io import BytesIO
from PIL import Image

# Add TRELLIS.2 to path
TRELLIS_PATH = os.environ.get("TRELLIS_PATH", "/app/TRELLIS.2")
sys.path.insert(0, TRELLIS_PATH)

# RunPod model cache configuration
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
HF_MODEL_ID = "microsoft/TRELLIS.2-4B"

# Global pipeline - initialized once at worker startup
pipeline = None


def find_cached_model_path(model_name: str) -> str | None:
    """
    Find model path in RunPod's cache directory.
    RunPod caches HuggingFace models at /runpod-volume/huggingface-cache/hub/
    """
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0])
    return None


def load_model():
    """
    Load TRELLIS.2 model into GPU memory.
    Uses RunPod's model caching - model should be pre-cached by RunPod.
    """
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading TRELLIS.2 model...")
    start_time = time.time()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Check RunPod's cache first
    cached_path = find_cached_model_path(HF_MODEL_ID)

    if cached_path:
        print(f"Loading from RunPod cache: {cached_path}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cached_path)
    else:
        # Fallback: download from HuggingFace (RunPod will cache it)
        print(f"Model not in cache, downloading: {HF_MODEL_ID}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(HF_MODEL_ID)

    pipeline.cuda()

    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f}s")

    return pipeline


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 image string."""
    if "," in data:
        data = data.split(",", 1)[1]
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data)).convert("RGB")


def handler(job):
    """
    RunPod serverless handler for TRELLIS.2 image-to-3D.

    Input parameters:
    - input_image: str - URL or base64 encoded image (required)
    - seed: int - Random seed (default: random)
    - pipeline_type: str - "512", "1024_cascade", or "1536_cascade" (default: "512")
    - texture_size: int - Output texture resolution (default: 1024)
    - decimation_target: int - Target triangle count (default: 20000)

    Returns:
    - glb: str - Base64 encoded GLB file
    - format: str - "glb"
    - inference_time: float - Time in seconds
    """
    job_input = job["input"]

    # Validate required input
    input_image = job_input.get("input_image")
    if not input_image:
        return {"error": "input_image is required"}

    # Load model (cached after first call)
    runpod.serverless.progress_update(job, "Loading model...")
    pipe = load_model()

    # Load image
    runpod.serverless.progress_update(job, "Processing image...")
    try:
        if input_image.startswith(("http://", "https://")):
            image = download_image(input_image)
        else:
            image = decode_base64_image(input_image)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    print(f"Image size: {image.size}")

    # Extract parameters with defaults
    seed = job_input.get("seed")
    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)

    pipeline_type = job_input.get("pipeline_type", "512")
    texture_size = job_input.get("texture_size", 1024)
    decimation_target = job_input.get("decimation_target", 20000)

    # Sampler parameters
    ss_sampling_steps = job_input.get("ss_sampling_steps", 20)
    ss_guidance_strength = job_input.get("ss_guidance_strength", 10.0)
    ss_guidance_rescale = job_input.get("ss_guidance_rescale", 0.5)
    ss_rescale_t = job_input.get("ss_rescale_t", 0.35)

    shape_slat_sampling_steps = job_input.get("shape_slat_sampling_steps", 20)
    shape_slat_guidance_strength = job_input.get("shape_slat_guidance_strength", 5.0)
    shape_slat_guidance_rescale = job_input.get("shape_slat_guidance_rescale", 0.5)
    shape_slat_rescale_t = job_input.get("shape_slat_rescale_t", 0.35)

    tex_slat_sampling_steps = job_input.get("tex_slat_sampling_steps", 20)
    tex_slat_guidance_strength = job_input.get("tex_slat_guidance_strength", 3.0)
    tex_slat_guidance_rescale = job_input.get("tex_slat_guidance_rescale", 0.5)
    tex_slat_rescale_t = job_input.get("tex_slat_rescale_t", 0.35)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Run inference
    runpod.serverless.progress_update(job, f"Generating 3D model ({pipeline_type})...")
    print(f"Running inference: pipeline_type={pipeline_type}, seed={seed}")
    start_time = time.time()

    try:
        outputs, latents = pipe.run(
            image,
            seed=seed,
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type=pipeline_type,
            return_latent=True,
        )
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f}s")

    # Export GLB mesh
    runpod.serverless.progress_update(job, "Exporting GLB...")
    try:
        from trellis2.representations import o_voxel

        shape_slat, tex_slat, res = outputs
        mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipe.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            glb_path = f.name

        glb.export(glb_path, extension_webp=True)

        with open(glb_path, "rb") as f:
            glb_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(glb_path)

    except Exception as e:
        return {"error": f"GLB export failed: {str(e)}"}

    glb_size_mb = len(glb_base64) * 3 / 4 / 1024 / 1024
    print(f"GLB exported: ~{glb_size_mb:.2f}MB")

    return {
        "glb": glb_base64,
        "format": "glb",
        "seed": seed,
        "pipeline_type": pipeline_type,
        "texture_size": texture_size,
        "decimation_target": decimation_target,
        "inference_time": round(inference_time, 2),
        "glb_size_mb": round(glb_size_mb, 2),
    }


# Initialize model at worker startup (RunPod best practice)
# This runs ONCE when the container starts, not on every request
print("=" * 50)
print("TRELLIS.2 RunPod Worker Starting...")
print("=" * 50)

try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed: {e}")
    print("Model will be loaded on first request")

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
