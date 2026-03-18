"""Modal app for Rediscover cloud GPU training."""

import modal

app = modal.App("rediscover")

# Persistent volume for data shards + tokenizer (~500MB, downloaded once)
vol = modal.Volume.from_name("rediscover-data", create_if_missing=True)

# Container image with PyTorch + dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "tiktoken>=0.6.0",
        "pyarrow>=15.0.0",
        "requests>=2.31.0",
        "rustbpe>=0.1.0",
    )
)

CACHE_DIR = "/data/autoresearch"
TRAINING_TIMEOUT = 900  # match local timeout


@app.function(
    gpu="A10G",
    timeout=TRAINING_TIMEOUT + 120,  # extra buffer for container startup
    volumes={"/data": vol},
    image=image,
    memory=16384,  # 16GB RAM
)
def run_experiment(train_code: str, prepare_code: str) -> dict:
    """Run a single training experiment on cloud GPU.

    Args:
        train_code: Contents of train.py (modified by council)
        prepare_code: Contents of prepare.py (fixed, never modified)

    Returns:
        dict with keys: val_bpb (float|None), output (str), success (bool)
    """
    import os
    import re
    import subprocess
    import sys
    import tempfile

    # Create experiment directory
    exp_dir = tempfile.mkdtemp(prefix="rediscover_")
    train_path = os.path.join(exp_dir, "train.py")
    prepare_path = os.path.join(exp_dir, "prepare.py")

    # Write experiment files
    with open(train_path, "w") as f:
        f.write(train_code)
    with open(prepare_path, "w") as f:
        # Patch CACHE_DIR to use Modal volume path
        patched = prepare_code.replace(
            'os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")',
            f'"{CACHE_DIR}"',
        )
        f.write(patched)

    # Ensure data exists on volume (first run downloads, subsequent runs use cache)
    data_dir = os.path.join(CACHE_DIR, "data")
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        # Run prepare.py to download data + train tokenizer
        result = subprocess.run(
            [sys.executable, prepare_path, "--num-shards", "4"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=exp_dir,
            env={**os.environ, "HOME": "/data"},
        )
        if result.returncode != 0:
            return {
                "val_bpb": None,
                "output": f"PREPARE FAILED:\n{result.stdout}\n{result.stderr}",
                "success": False,
            }
        vol.commit()  # persist data to volume

    # Run training
    try:
        result = subprocess.run(
            [sys.executable, train_path],
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT,
            cwd=exp_dir,
        )
        output = result.stdout + result.stderr

        if result.returncode != 0:
            return {"val_bpb": None, "output": output, "success": False}

        # Parse val_bpb
        match = re.search(r"val_bpb:\s+([\d.]+)", output)
        val_bpb = float(match.group(1)) if match else None

        return {"val_bpb": val_bpb, "output": output, "success": val_bpb is not None}

    except subprocess.TimeoutExpired:
        return {
            "val_bpb": None,
            "output": "TIMEOUT: Training exceeded time limit on Modal",
            "success": False,
        }
