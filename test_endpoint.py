#!/usr/bin/env python3
"""
Test script for TRELLIS.2 RunPod endpoint.

Usage:
    export RUNPOD_API_KEY="your-api-key"
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"
    python test_endpoint.py --image path/to/image.png --output output.glb
"""

import os
import sys
import time
import json
import base64
import argparse
import requests
from pathlib import Path


def submit_job(endpoint_id: str, api_key: str, input_data: dict) -> str:
    """Submit a job to RunPod and return job ID."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json={"input": input_data})
    response.raise_for_status()

    result = response.json()
    return result["id"]


def check_status(endpoint_id: str, api_key: str, job_id: str) -> dict:
    """Check job status."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


def wait_for_completion(endpoint_id: str, api_key: str, job_id: str, timeout: int = 300) -> dict:
    """Poll for job completion."""
    start = time.time()

    while time.time() - start < timeout:
        status = check_status(endpoint_id, api_key, job_id)
        state = status.get("status")

        print(f"  Status: {state}")

        if state == "COMPLETED":
            return status.get("output", {})
        elif state == "FAILED":
            error = status.get("error", "Unknown error")
            raise Exception(f"Job failed: {error}")
        elif state in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        else:
            time.sleep(2)

    raise Exception(f"Timeout after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Test TRELLIS.2 RunPod endpoint")
    parser.add_argument("--image", required=True, help="Path to input image or URL")
    parser.add_argument("--output", default="output.glb", help="Output GLB path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pipeline", default="512", choices=["512", "1024_cascade", "1536_cascade"])
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    # Get credentials from environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")

    if not api_key or not endpoint_id:
        print("Error: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
        sys.exit(1)

    # Prepare input
    if args.image.startswith("http://") or args.image.startswith("https://"):
        input_image = args.image
    else:
        # Read and encode local file
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        with open(image_path, "rb") as f:
            input_image = base64.b64encode(f.read()).decode("utf-8")
        print(f"Encoded image: {len(input_image)} bytes")

    input_data = {
        "input_image": input_image,
        "seed": args.seed,
        "pipeline_type": args.pipeline,
        "texture_size": args.texture_size,
    }

    print(f"Submitting job to endpoint {endpoint_id}...")
    job_id = submit_job(endpoint_id, api_key, input_data)
    print(f"Job ID: {job_id}")

    print("Waiting for completion...")
    result = wait_for_completion(endpoint_id, api_key, job_id, args.timeout)

    if result.get("status") == "success":
        # Decode and save GLB
        glb_data = base64.b64decode(result["glb"])
        with open(args.output, "wb") as f:
            f.write(glb_data)

        print(f"\nSuccess!")
        print(f"  Inference time: {result.get('inference_time', '?')}s")
        print(f"  GLB size: {result.get('glb_size_mb', '?')} MB")
        print(f"  Saved to: {args.output}")
    else:
        print(f"\nFailed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
