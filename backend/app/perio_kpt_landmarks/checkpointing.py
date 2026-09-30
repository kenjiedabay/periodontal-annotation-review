"""Versioned exact-resume checkpoints for future Perio-KPT training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

CHECKPOINT_SCHEMA = "perio-kpt-exact-resume-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_worker(worker_id: int) -> None:
    """Deterministically seed Python and NumPy from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed); np.random.seed(worker_seed)


def worker_seed_config(num_workers: int, persistent_workers: bool, base_seed: int) -> dict:
    return {"strategy": "torch.initial_seed modulo 2**32 seeds Python and NumPy",
            "worker_init_fn": "app.perio_kpt_landmarks.checkpointing.seed_worker",
            "num_workers": num_workers, "persistent_workers": persistent_workers,
            "base_seed": base_seed}


def save_exact_checkpoint(path: Path, *, model, optimizer, scaler, scheduler,
                          dataloader_generator: torch.Generator, completed_epoch: int,
                          config: dict, source_fold_manifest: dict,
                          worker_seeding: dict, training_state: dict | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "grad_scaler_state_dict": scaler.state_dict(),
               "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
               "scheduler_class": scheduler.__class__.__qualname__ if scheduler is not None else None,
               "rng_state": capture_rng_state(),
               "dataloader_generator_state": dataloader_generator.get_state(),
               "worker_seeding": worker_seeding,
               "completed_epoch": completed_epoch, "config": config,
               "source_fold_manifest": source_fold_manifest,
               "training_state": training_state or {},
               "checkpoint_hash_algorithm": "sha256",
               "checkpoint_hash_sidecar": path.name + ".sha256"}
    torch.save(payload, path)
    digest = file_sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return {"path": str(path), "sha256": digest, "sidecar": str(sidecar)}


def load_exact_checkpoint(path: Path, *, model, optimizer, scaler, scheduler,
                          dataloader_generator: torch.Generator,
                          expected_config: dict | None = None,
                          expected_manifest_hash: str | None = None) -> dict:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError("exact-resume checkpoint SHA-256 sidecar is missing")
    expected_digest = sidecar.read_text(encoding="ascii").split()[0]
    actual_digest = file_sha256(path)
    if actual_digest != expected_digest:
        raise ValueError("checkpoint SHA-256 verification failed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint does not contain exact-resume state")
    if expected_config is not None:
        # A continuation may raise only the stopping ceiling; all training
        # behavior and data identity must remain identical.
        mutable = {"maximum_epochs"}
        saved_fixed = {key: value for key, value in payload["config"].items() if key not in mutable}
        expected_fixed = {key: value for key, value in expected_config.items() if key not in mutable}
        if saved_fixed != expected_fixed:
            raise ValueError("resume configuration differs from checkpoint configuration")
    manifest_hash = payload["source_fold_manifest"].get("manifest_sha256")
    if expected_manifest_hash is not None and manifest_hash != expected_manifest_hash:
        raise ValueError("source/fold manifest differs from checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["grad_scaler_state_dict"])
    if payload["scheduler_state_dict"] is not None:
        if scheduler is None: raise ValueError("checkpoint requires a scheduler")
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    elif scheduler is not None:
        raise ValueError("runtime scheduler is absent from checkpoint")
    dataloader_generator.set_state(payload["dataloader_generator_state"])
    restore_rng_state(payload["rng_state"])
    payload["verified_sha256"] = actual_digest
    return payload
