"""
============================================================
Training Pipeline
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Features:
  - Mixed Precision Training (torch.cuda.amp)
  - Early Stopping with configurable patience
  - Checkpoint saving (best + latest)
  - CosineAnnealingLR scheduler with warm restarts
  - TensorBoard logging
  - Comprehensive metric tracking

Usage:
    python train.py --voice_dim 200 --eeg_dim 150 --gait_dim 40 \
        --epochs 100 --batch_size 32 --lr 1e-4 --device cuda
============================================================
"""

import argparse
import json
import os
import sys
import time

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with box-drawing/emoji characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    print(f"[WARN] TensorBoard not available ({e}). Training will proceed without logging to TensorBoard.")
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.multimodal import ParkinsonMultimodalModel
from src.training.losses   import DualTaskLoss
from src.training.evaluate import compute_all_metrics


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ParkinsonDataset(Dataset):
    """
    Dataset for multimodal Parkinson's features.

    Expects:
        data: Dict with keys:
            'voice': (N, voice_dim) array or None
            'eeg':   (N, eeg_dim) array or None
            'gait':  (N, gait_dim) array or None
            'updrs': (N,) float array
            'severity': (N,) int array {0=Mild, 1=Moderate, 2=Severe}
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.voice    = data.get("voice")
        self.eeg      = data.get("eeg")
        self.gait     = data.get("gait")
        self.updrs    = data["updrs"].astype(np.float32)
        self.severity = data["severity"].astype(np.int64)
        self.n        = len(self.updrs)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = {
            "updrs":    torch.tensor(self.updrs[idx]),
            "severity": torch.tensor(self.severity[idx]),
        }

        # Build modality mask
        mask = []

        if self.voice is not None:
            sample["voice"] = torch.tensor(self.voice[idx], dtype=torch.float32)
            mask.append(1.0)
        else:
            mask.append(0.0)

        if self.eeg is not None:
            sample["eeg"] = torch.tensor(self.eeg[idx], dtype=torch.float32)
            mask.append(1.0)
        else:
            mask.append(0.0)

        if self.gait is not None:
            sample["gait"] = torch.tensor(self.gait[idx], dtype=torch.float32)
            mask.append(1.0)
        else:
            mask.append(0.0)

        sample["modality_mask"] = torch.tensor(mask, dtype=torch.float32)
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Custom collate function supporting missing modalities.
    """
    keys = batch[0].keys()
    out  = {}

    for key in keys:
        tensors = [item[key] for item in batch if key in item]
        if len(tensors) == len(batch):
            out[key] = torch.stack(tensors)
        else:
            out[key] = None

    return out


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping monitor.

    Args:
        patience:  Steps without improvement before stopping
        min_delta: Minimum improvement to reset counter
        mode:      'min' (lower is better) or 'max' (higher is better)
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter   += 1
            print(f"[EarlyStopping] {self.counter}/{self.patience} (best={self.best_score:.4f})")
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Complete training loop for the Parkinson's multimodal model.
    """

    def __init__(
        self,
        model:            ParkinsonMultimodalModel,
        train_loader:     DataLoader,
        val_loader:       DataLoader,
        optimizer:        optim.Optimizer,
        loss_fn:          DualTaskLoss,
        scheduler:        optim.lr_scheduler._LRScheduler,
        device:           torch.device,
        checkpoint_dir:   Path,
        log_dir:          Path,
        epochs:           int = 100,
        patience:         int = 15,
        use_amp:          bool = True,
        grad_clip:        float = 1.0,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.loss_fn        = loss_fn
        self.scheduler      = scheduler
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir        = Path(log_dir)
        self.epochs         = epochs
        self.grad_clip      = grad_clip
        self.use_amp        = use_amp and torch.cuda.is_available()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer        = SummaryWriter(str(self.log_dir))
        self.scaler        = GradScaler(enabled=self.use_amp)
        self.early_stop    = EarlyStopping(patience=patience, mode="min")
        self.best_val_loss = float("inf")
        self.history       = []

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        reg_losses = []
        cls_losses = []
        all_updrs_pred, all_updrs_true   = [], []
        all_cls_pred,   all_cls_true     = [], []

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    voice_feats=batch.get("voice"),
                    eeg_feats=batch.get("eeg"),
                    gait_feats=batch.get("gait"),
                    modality_mask=batch.get("modality_mask"),
                )

                losses = self.loss_fn(
                    pred_updrs=outputs["updrs_pred"],
                    true_updrs=batch["updrs"].float(),
                    pred_logits=outputs["severity_logits"],
                    true_class=batch["severity"],
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            reg_losses.append(losses["regression"].item())
            cls_losses.append(losses["classification"].item())

            # Collect for metrics
            all_updrs_pred.extend(outputs["updrs_pred"].detach().cpu().numpy())
            all_updrs_true.extend(batch["updrs"].cpu().numpy())
            all_cls_pred.extend(
                outputs["severity_logits"].argmax(dim=-1).detach().cpu().numpy()
            )
            all_cls_true.extend(batch["severity"].cpu().numpy())

        metrics = compute_all_metrics(
            np.array(all_updrs_true),  np.array(all_updrs_pred),
            np.array(all_cls_true),    np.array(all_cls_pred),
        )
        metrics["loss_total"]      = total_loss / len(self.train_loader)
        metrics["loss_regression"] = np.mean(reg_losses)
        metrics["loss_cls"]        = np.mean(cls_losses)
        return metrics

    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        reg_losses = []
        cls_losses = []
        all_updrs_pred, all_updrs_true = [], []
        all_cls_pred,   all_cls_true   = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._batch_to_device(batch)

                outputs = self.model(
                    voice_feats=batch.get("voice"),
                    eeg_feats=batch.get("eeg"),
                    gait_feats=batch.get("gait"),
                    modality_mask=batch.get("modality_mask"),
                )

                losses = self.loss_fn(
                    pred_updrs=outputs["updrs_pred"],
                    true_updrs=batch["updrs"].float(),
                    pred_logits=outputs["severity_logits"],
                    true_class=batch["severity"],
                )

                total_loss += losses["total"].item()
                reg_losses.append(losses["regression"].item())
                cls_losses.append(losses["classification"].item())

                all_updrs_pred.extend(outputs["updrs_pred"].cpu().numpy())
                all_updrs_true.extend(batch["updrs"].cpu().numpy())
                all_cls_pred.extend(
                    outputs["severity_logits"].argmax(dim=-1).cpu().numpy()
                )
                all_cls_true.extend(batch["severity"].cpu().numpy())

        metrics = compute_all_metrics(
            np.array(all_updrs_true), np.array(all_updrs_pred),
            np.array(all_cls_true),   np.array(all_cls_pred),
        )
        metrics["loss_total"]      = total_loss / len(self.val_loader)
        metrics["loss_regression"] = np.mean(reg_losses)
        metrics["loss_cls"]        = np.mean(cls_losses)
        return metrics

    def _log_metrics(self, metrics: Dict, prefix: str, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        checkpoint = {
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "metrics":          metrics,
            "model_args": {
                "voice_input_dim": self.model.voice_input_dim,
                "eeg_input_dim":   self.model.eeg_input_dim,
                "gait_input_dim":  self.model.gait_input_dim,
                "embed_dim":       self.model.embed_dim,
                "fusion_d_model":  self.model.fusion_d_model,
            },
        }

        # Always save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
            print(f"  [OK] Saved best model (val_loss={metrics['loss_total']:.4f})")

    def train(self) -> List[Dict]:
        print(f"\n{'='*60}")
        print(f"  Training on device: {self.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Epochs: {self.epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)
            self._log_metrics(train_metrics, "train", epoch)

            # Validate
            val_metrics = self._val_epoch()
            self._log_metrics(val_metrics, "val", epoch)

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Loss {train_metrics['loss_total']:.4f} -> {val_metrics['loss_total']:.4f} | "
                f"Acc {val_metrics.get('accuracy', 0):.3f} | "
                f"MAE {val_metrics.get('mae', 0):.2f} | "
                f"F1 {val_metrics.get('f1', 0):.3f} | "
                f"LR {current_lr:.2e} | {elapsed:.1f}s"
            )

            # Log LR
            self.writer.add_scalar("train/lr", current_lr, epoch)

            # Save checkpoint
            is_best = val_metrics["loss_total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss_total"]
            self._save_checkpoint(epoch, val_metrics, is_best)

            # History
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            self.history.append(record)

            # Early stopping
            if self.early_stop.step(val_metrics["loss_total"]):
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        # Save history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)

        self.writer.close()
        print(f"\n✅ Training complete! Best val_loss: {self.best_val_loss:.4f}")
        return self.history


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def create_synthetic_data(
    n_samples: int = 200,
    voice_dim: int = 200,
    eeg_dim:   int = 150,
    gait_dim:  int = 40,
) -> Dict[str, np.ndarray]:
    """Create synthetic data for testing the training pipeline."""
    np.random.seed(42)
    severity = np.random.randint(0, 3, n_samples)
    updrs    = severity * 30 + np.random.randn(n_samples) * 5 + 10

    return {
        "voice":    np.random.randn(n_samples, voice_dim).astype(np.float32),
        "eeg":      np.random.randn(n_samples, eeg_dim).astype(np.float32),
        "gait":     np.random.randn(n_samples, gait_dim).astype(np.float32),
        "updrs":    np.clip(updrs, 0, 108).astype(np.float32),
        "severity": severity,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Parkinson's Multimodal Model")
    parser.add_argument("--voice_dim",     type=int,   default=200)
    parser.add_argument("--eeg_dim",       type=int,   default=150)
    parser.add_argument("--gait_dim",      type=int,   default=40)
    parser.add_argument("--embed_dim",     type=int,   default=256)
    parser.add_argument("--fusion_dim",    type=int,   default=512)
    parser.add_argument("--fusion_heads",  type=int,   default=8)
    parser.add_argument("--fusion_layers", type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--checkpoint_dir", type=str,  default="./checkpoints")
    parser.add_argument("--log_dir",        type=str,  default="./logs")
    parser.add_argument("--data_path",      type=str,  default=None,
                        help="Path to preprocessed .npz data file")
    parser.add_argument("--no_amp",         action="store_true",
                        help="Disable mixed precision training")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    # ── Load or create data ───────────────────────────────
    if args.data_path and Path(args.data_path).exists():
        print(f"[INFO] Loading data from {args.data_path}")
        raw = np.load(args.data_path, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
    else:
        print("[INFO] Using synthetic data for demonstration")
        data = create_synthetic_data(
            n_samples=200,
            voice_dim=args.voice_dim,
            eeg_dim=args.eeg_dim,
            gait_dim=args.gait_dim,
        )

    # ── Create Dataset & DataLoaders ──────────────────────
    dataset = ParkinsonDataset(data)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader   = DataLoader(
        val_set,   batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────
    model = ParkinsonMultimodalModel(
        voice_input_dim=args.voice_dim,
        eeg_input_dim=args.eeg_dim,
        gait_input_dim=args.gait_dim,
        embed_dim=args.embed_dim,
        fusion_d_model=args.fusion_dim,
        fusion_n_heads=args.fusion_heads,
        fusion_n_layers=args.fusion_layers,
        dropout=args.dropout,
    )

    # ── Optimizer & Scheduler ─────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # ── Loss ──────────────────────────────────────────────
    loss_fn = DualTaskLoss(
        lambda_reg=0.6,
        lambda_cls=0.4,
        label_smoothing=0.1,
        use_uncertainty=False,
    )

    # ── Train ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_dir=Path(args.log_dir),
        epochs=args.epochs,
        patience=args.patience,
        use_amp=not args.no_amp,
    )

    history = trainer.train()

    print(f"\n📊 Final validation metrics:")
    if history:
        final = history[-1]["val"]
        for k, v in final.items():
            print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
