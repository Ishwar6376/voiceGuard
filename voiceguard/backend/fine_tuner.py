"""
fine_tuner.py — Incremental fine-tuning of a speaker classification head

Strategy:
  1. The ECAPA-TDNN base encoder stays FROZEN (never fine-tuned).
  2. A lightweight 3-layer MLP is trained on top of the 192-dim embeddings.
  3. Binary classification: target speaker (label 0) vs impostor (label 1).
  4. Negative samples come from other enrolled speakers + synthetic perturbations.
  5. Training is fast (~2-5 seconds on CPU for typical datasets of 5-20 notes).

The fine-tuned head is saved per speaker and loaded for inference.
"""
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("fine_tuner")

# ─── Configuration ───────────────────────────────────────────────────────────
MODELS_DIR    = Path(__file__).parent / "storage" / "models"
EMBEDDING_DIM = 192   # must match model_manager.EMBEDDING_DIM
HIDDEN_DIM    = 128
DEFAULT_EPOCHS = 30
DEFAULT_LR     = 1e-3


# ─── Classifier Head ────────────────────────────────────────────────────────

class SpeakerClassifierHead(nn.Module):
    """
    Lightweight MLP classifier attached on top of frozen speaker embeddings.
    Input:  192-dim embedding
    Output: logits for [target_speaker, impostor]
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_speakers: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_speakers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── FineTuner ───────────────────────────────────────────────────────────────

class FineTuner:
    """
    Manages per-speaker classifier heads: training, saving, loading, and inference.
    """

    def __init__(self):
        self._classifiers: dict[str, SpeakerClassifierHead] = {}
        self._configs: dict[str, dict] = {}
        self._training = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        return self._training

    # ── Load / Check ──────────────────────────────────────────────────────

    def get_classifier(self, speaker_id: str) -> SpeakerClassifierHead | None:
        """
        Load the fine-tuned classifier for a speaker.
        Returns from in-memory cache, or loads from disk, or returns None.
        """
        # In-memory cache hit
        if speaker_id in self._classifiers:
            return self._classifiers[speaker_id]

        # Try loading from disk
        model_dir = MODELS_DIR / speaker_id
        checkpoint_path = model_dir / "classifier_head.pt"
        config_path = model_dir / "classifier_config.json"

        if not checkpoint_path.exists() or not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                config = json.load(f)

            classifier = SpeakerClassifierHead(
                input_dim=config.get("input_dim", EMBEDDING_DIM),
                hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
                num_speakers=config.get("num_speakers", 2),
            )
            classifier.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
            classifier.eval()

            self._classifiers[speaker_id] = classifier
            self._configs[speaker_id] = config
            logger.info(f"✅ Loaded fine-tuned classifier for '{speaker_id}'")
            return classifier

        except Exception as e:
            logger.error(f"Failed to load classifier for '{speaker_id}': {e}")
            return None

    def has_classifier(self, speaker_id: str) -> bool:
        """Check if a fine-tuned classifier exists (in memory or on disk)."""
        if speaker_id in self._classifiers:
            return True
        model_dir = MODELS_DIR / speaker_id
        return (model_dir / "classifier_head.pt").exists()

    def get_config(self, speaker_id: str) -> dict | None:
        """Get the training config for a speaker's classifier."""
        if speaker_id in self._configs:
            return self._configs[speaker_id]

        config_path = MODELS_DIR / speaker_id / "classifier_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return None

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, speaker_id: str, embedding: np.ndarray) -> dict | None:
        """
        Run the fine-tuned classifier on a speaker embedding.

        Returns:
            {
                "confidence": float (0-1, probability of being the target speaker),
                "is_target": bool,
            }
            or None if no classifier exists.
        """
        classifier = self.get_classifier(speaker_id)
        if classifier is None:
            return None

        config = self._configs.get(speaker_id, {})
        target_label = config.get("target_label", 0)

        with torch.no_grad():
            x = torch.from_numpy(embedding).float().unsqueeze(0)  # (1, 192)
            logits = classifier(x)
            probs = torch.softmax(logits, dim=1).squeeze()  # (num_speakers,)
            predicted = torch.argmax(probs).item()
            confidence = float(probs[target_label])

        return {
            "confidence": round(confidence, 4),
            "is_target": predicted == target_label,
        }

    # ── Training ──────────────────────────────────────────────────────────

    def run_fine_tune(
        self,
        speaker_id: str,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        augment: bool = True,
    ) -> dict:
        """
        Fine-tune a classification head for a specific speaker.

        Positive samples: speaker's voice note embeddings (+ augmentations).
        Negative samples: other speakers' embeddings (+ synthetic perturbations).

        Returns a result dict with training metrics.
        Raises ValueError if not enough voice notes.
        """
        # Lazy import to avoid circular dependency
        from voice_notes import (
            get_all_embeddings, can_fine_tune,
            VOICE_NOTES_DIR, MIN_NOTES_FOR_FINETUNE,
        )

        if not can_fine_tune(speaker_id):
            raise ValueError(
                f"Speaker '{speaker_id}' has fewer than {MIN_NOTES_FOR_FINETUNE} "
                f"voice notes. Add more before fine-tuning."
            )

        self._training = True
        try:
            return self._do_fine_tune(
                speaker_id, epochs, lr, augment,
                get_all_embeddings, VOICE_NOTES_DIR,
            )
        finally:
            self._training = False

    def _do_fine_tune(
        self,
        speaker_id: str,
        epochs: int,
        lr: float,
        augment: bool,
        get_all_embeddings,
        voice_notes_dir: Path,
    ) -> dict:
        """Internal training loop."""

        # ── 1. Gather positive embeddings ─────────────────────────────────
        positive_embeddings = get_all_embeddings(speaker_id)
        n_raw_positive = len(positive_embeddings)
        logger.info(
            f"🎯 Fine-tuning for '{speaker_id}': "
            f"{n_raw_positive} positive samples"
        )

        # ── 2. Gather negative embeddings (other speakers) ───────────────
        negative_embeddings = []
        if voice_notes_dir.exists():
            for other_dir in voice_notes_dir.iterdir():
                if other_dir.is_dir() and other_dir.name != speaker_id:
                    for embed_file in other_dir.glob("*_embed.npy"):
                        try:
                            negative_embeddings.append(np.load(embed_file))
                        except Exception:
                            pass

        # If too few negatives, synthesize via perturbation
        if len(negative_embeddings) < 5:
            logger.info("Synthesizing negative samples via perturbation")
            rng = np.random.default_rng(42)
            for emb in positive_embeddings:
                for _ in range(3):
                    noise = rng.standard_normal(emb.shape).astype(np.float32) * 0.5
                    perturbed = emb + noise
                    norm = np.linalg.norm(perturbed)
                    if norm > 0:
                        perturbed = perturbed / norm
                    negative_embeddings.append(perturbed)

        # ── 3. Augment positives ──────────────────────────────────────────
        augmented_positives = list(positive_embeddings)
        if augment:
            rng = np.random.default_rng(123)
            for emb in positive_embeddings:
                for _ in range(2):
                    noise = rng.standard_normal(emb.shape).astype(np.float32) * 0.05
                    aug = emb + noise
                    norm = np.linalg.norm(aug)
                    if norm > 0:
                        aug = aug / norm
                    augmented_positives.append(aug)

        n_positive = len(augmented_positives)
        n_negative = len(negative_embeddings)
        logger.info(f"Training data: {n_positive} positive, {n_negative} negative")

        # ── 4. Build tensors ──────────────────────────────────────────────
        X = np.array(augmented_positives + negative_embeddings, dtype=np.float32)
        y = np.array(
            [0] * n_positive + [1] * n_negative,
            dtype=np.int64,
        )

        # Shuffle
        rng = np.random.default_rng(777)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        # ── 5. Create and train classifier ────────────────────────────────
        classifier = SpeakerClassifierHead(
            input_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_speakers=2,
        )

        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        batch_size = min(32, len(X_tensor))

        classifier.train()
        best_loss = float("inf")
        best_state = None
        final_accuracy = 0.0

        for epoch in range(epochs):
            perm = torch.randperm(len(X_tensor))
            total_loss = 0.0
            correct = 0

            for i in range(0, len(X_tensor), batch_size):
                batch_idx = perm[i : i + batch_size]
                batch_x = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                logits = classifier(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idx)
                correct += (logits.argmax(1) == batch_y).sum().item()

            avg_loss = total_loss / len(X_tensor)
            accuracy = correct / len(X_tensor)
            final_accuracy = accuracy

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"loss={avg_loss:.4f}, acc={accuracy:.1%}"
                )

        # Restore best weights
        if best_state is not None:
            classifier.load_state_dict(best_state)

        # ── 6. Save checkpoint ────────────────────────────────────────────
        model_dir = MODELS_DIR / speaker_id
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(classifier.state_dict(), model_dir / "classifier_head.pt")

        config = {
            "input_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_speakers": 2,
            "target_label": 0,
            "label_map": {speaker_id: 0, "__impostor__": 1},
            "trained_at": datetime.now().isoformat(),
            "num_raw_positive": n_raw_positive,
            "num_positive": n_positive,
            "num_negative": n_negative,
            "epochs": epochs,
            "best_loss": round(best_loss, 6),
            "final_accuracy": round(final_accuracy, 4),
        }

        with open(model_dir / "classifier_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Update in-memory cache
        classifier.eval()
        self._classifiers[speaker_id] = classifier
        self._configs[speaker_id] = config

        logger.info(
            f"✅ Fine-tuning complete for '{speaker_id}': "
            f"acc={final_accuracy:.1%}, loss={best_loss:.4f}"
        )

        return {
            "success": True,
            "speaker_id": speaker_id,
            "epochs": epochs,
            "final_accuracy": round(final_accuracy, 4),
            "best_loss": round(best_loss, 6),
            "num_positive": n_positive,
            "num_negative": n_negative,
            "trained_at": config["trained_at"],
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def delete_classifier(self, speaker_id: str) -> bool:
        """Remove a speaker's fine-tuned classifier from disk and cache."""
        # Remove from cache
        self._classifiers.pop(speaker_id, None)
        self._configs.pop(speaker_id, None)

        # Remove from disk
        model_dir = MODELS_DIR / speaker_id
        if not model_dir.exists():
            return False

        deleted = False
        for f in model_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted = True
        try:
            model_dir.rmdir()
        except OSError:
            pass

        if deleted:
            logger.info(f"🗑️ Deleted classifier for speaker '{speaker_id}'")
        return deleted


# ─── Global singleton ────────────────────────────────────────────────────────
fine_tuner = FineTuner()
