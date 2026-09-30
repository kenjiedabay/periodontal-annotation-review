"""CPU-fast tests for the isolated Gate 2 training components."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.perio_kpt_landmarks.checkpointing import (load_exact_checkpoint,
                                                   save_exact_checkpoint,
                                                   worker_seed_config)
from app.perio_kpt_landmarks.evaluation import decode_heatmaps, summarize_predictions
from app.perio_kpt_landmarks.model import (PerioLandmarkNet,
                                           balanced_masked_heatmap_loss,
                                           masked_heatmap_loss)
from app.perio_kpt_landmarks.training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/perio-kpt-landmarks/gate1/five_fold_manifest.json"


def test_model_produces_eleven_full_resolution_channels() -> None:
    model = PerioLandmarkNet(11).eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 64, 64)))
    assert output.shape == (1, 11, 64, 64)
    assert torch.isfinite(output).all()


def test_masked_loss_ignores_unavailable_channel() -> None:
    logits = torch.zeros((1, 2, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    availability = torch.tensor([[1.0, 0.0]])
    baseline = masked_heatmap_loss(logits, target, availability)
    changed = target.detach().clone(); changed[:, 1] = 1.0
    assert masked_heatmap_loss(logits, changed, availability) == baseline
    baseline.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_decoder_argmax_and_metric_summary() -> None:
    heatmaps = torch.zeros((1, 11, 16, 16)); heatmaps[0, 0, 4, 3] = 1
    points, confidence = decode_heatmaps(heatmaps)
    assert points[0, 0].tolist() == [3.0, 4.0]
    assert confidence[0, 0] == 1
    row = {"class_id": 0, "prediction": points[0].tolist(),
           "target": points[0].tolist(), "availability": [True] + [False] * 10}
    metrics = summarize_predictions([row], 16)
    assert metrics["per_landmark"]["CEJ-m"]["mean_pixel_error"] == 0


def test_balanced_loss_and_decoder_exclude_padding() -> None:
    logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits); target[0, 0, 3, 3] = 1
    available = torch.tensor([[1.0, 0.0]])
    content = torch.zeros((1, 8, 8)); content[:, 2:6, 2:6] = 1
    loss, parts = balanced_masked_heatmap_loss(logits, target, available, content)
    assert loss.item() == pytest.approx(0.25)
    assert parts["foreground"][0, 0].item() == pytest.approx(0.25)
    assert parts["background"][0, 0].item() == pytest.approx(0.25)
    probabilities = torch.zeros((1, 2, 8, 8)); probabilities[:, :, 0, 0] = 1
    probabilities[:, :, 4, 5] = 0.8
    decoded, confidence = decode_heatmaps(probabilities, content)
    assert decoded[0, 0].tolist() == [5.0, 4.0]
    assert confidence[0, 0] == pytest.approx(0.8)


def test_fold_zero_dataset_uses_manifest_and_skips_quarantined_rows() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fold = manifest["folds"][0]
    training = PerioCropDataset(MANIFEST, fold["train_image_ids"], 256)
    validation = PerioCropDataset(MANIFEST, fold["validation_image_ids"], 256)
    record_ids = {annotation.record_id for _, annotation in training.records + validation.records}
    assert "Image190:object-2" not in record_ids
    assert "Image24:object-3" not in record_ids
    assert training and validation
    sample = training[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["target"].shape == (11, 256, 256)
    assert sample["availability"].sum() > 0
    assert sample["valid_content"].shape == (256, 256)

    # Every validation crop must be representable without clipping a landmark.
    for index in range(len(validation)):
        item = validation[index]
        assert torch.isfinite(item["target"]).all()


def test_loss_rejects_empty_landmark_batch() -> None:
    with pytest.raises(ValueError, match="no available landmarks"):
        masked_heatmap_loss(torch.zeros((1, 11, 4, 4)), torch.zeros((1, 11, 4, 4)),
                            torch.zeros((1, 11)))


def test_tooth_only_dataset_excludes_arr_and_pls() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fold = manifest["folds"][0]
    dataset = PerioCropDataset(MANIFEST, fold["validation_image_ids"], 256, {0, 1, 2})
    assert dataset.records
    assert {annotation.class_id for _, annotation in dataset.records} <= {0, 1, 2}


def _optimizer_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from _optimizer_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value: yield from _optimizer_tensors(child)


def test_exact_resume_matches_continuous_two_epoch_training() -> None:
    checkpoint = Path(__file__).parent / ".test_exact_resume.pt"
    sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
    features = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 48
    targets = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
    dataset = TensorDataset(features, targets)

    def setup():
        random.seed(77); torch.manual_seed(77); np.random.seed(77)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.2), nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = torch.Generator().manual_seed(42)
        return model, optimizer, scheduler, scaler, generator

    def epoch(model, optimizer, scheduler, generator):
        loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=generator, num_workers=0)
        model.train()
        for inputs, truth in loader:
            optimizer.zero_grad(set_to_none=True)
            jitter = 1.0 + .01 * (random.random() + float(np.random.rand()))
            loss = (model(inputs * jitter) - truth).square().mean(); loss.backward(); optimizer.step()
        scheduler.step()

    try:
        continuous = setup(); epoch(continuous[0], continuous[1], continuous[2], continuous[4]); epoch(continuous[0], continuous[1], continuous[2], continuous[4])
        split = setup(); epoch(split[0], split[1], split[2], split[4])
        manifest = {"manifest_sha256": "synthetic-fixed-manifest", "folds": [{"fold": 0}]}
        save_exact_checkpoint(checkpoint, model=split[0], optimizer=split[1], scaler=split[3],
                              scheduler=split[2], dataloader_generator=split[4], completed_epoch=1,
                              config={"seed": 42, "maximum_epochs": 1}, source_fold_manifest=manifest,
                              worker_seeding=worker_seed_config(0, False, 42))
        torch.manual_seed(999); np.random.seed(999)
        resumed = setup()
        payload = load_exact_checkpoint(checkpoint, model=resumed[0], optimizer=resumed[1],
                                        scaler=resumed[3], scheduler=resumed[2],
                                        dataloader_generator=resumed[4],
                                        expected_config={"seed": 42, "maximum_epochs": 2},
                                        expected_manifest_hash="synthetic-fixed-manifest")
        assert payload["completed_epoch"] == 1
        epoch(resumed[0], resumed[1], resumed[2], resumed[4])
        for expected, actual in zip(continuous[0].parameters(), resumed[0].parameters()):
            assert torch.allclose(expected, actual, atol=1e-8, rtol=1e-7)
        expected_tensors = list(_optimizer_tensors(continuous[1].state_dict()))
        actual_tensors = list(_optimizer_tensors(resumed[1].state_dict()))
        assert len(expected_tensors) == len(actual_tensors)
        assert all(torch.allclose(a, b, atol=1e-8, rtol=1e-7) for a, b in zip(expected_tensors, actual_tensors))
        assert continuous[1].state_dict()["param_groups"] == resumed[1].state_dict()["param_groups"]
        assert continuous[2].state_dict() == resumed[2].state_dict()
        assert continuous[3].state_dict() == resumed[3].state_dict()
        continuous[0].eval(); resumed[0].eval()
        with torch.inference_mode():
            assert torch.allclose(continuous[0](features), resumed[0](features), atol=1e-8, rtol=1e-7)
    finally:
        checkpoint.unlink(missing_ok=True); sidecar.unlink(missing_ok=True)
