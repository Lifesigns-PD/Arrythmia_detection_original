"""
models_v2.py — Enhanced CNN+Transformer with Auxiliary Feature Input
=====================================================================

SAFE ROLLOUT: This file is NEW and sits alongside the original models.py.
              The original CNNTransformerClassifier is UNCHANGED.

Architecture:
  Signal Pathway:  Raw ECG (B, 1250) → SmallCNN → TransformerEncoder → (B, 128)
  Feature Pathway: Extracted features (B, num_features) → Dense → (B, 64)
  Fusion:          Concatenate → Dense → Classification

Both pathways are necessary because:
  - CNN+Transformer learns waveform shape & rhythm from raw signal
  - Feature pathway provides explicit clinical measurements (HR, QRS width,
    QTc, ST deviation, etc.) that the CNN might miss or learn slowly

The model can also run in SIGNAL-ONLY mode (backward compatible) by passing
features=None — it then behaves identically to the original model.

Usage:
    from models_v2 import CNNTransformerWithFeatures

    # With features (new):
    model = CNNTransformerWithFeatures(num_classes=13, num_features=13)
    logits = model(signal_tensor, feature_tensor)

    # Without features (backward compat):
    model = CNNTransformerWithFeatures(num_classes=13, num_features=0)
    logits = model(signal_tensor)
"""

import torch
import torch.nn as nn

# Reuse the EXACT same CNN from the original model — no duplication
from models import SmallCNN


class CNNTransformerWithFeatures(nn.Module):
    """
    Enhanced version of CNNTransformerClassifier that accepts an optional
    auxiliary feature vector alongside the raw ECG signal.

    When num_features=0, this model is architecturally identical to the
    original CNNTransformerClassifier (signal-only mode).
    """

    TARGET_LEN = 1250  # 10 seconds at 125Hz

    def __init__(
        self,
        num_classes: int = 13,
        num_features: int = 13,
        cnn_channels: list = None,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.num_features = num_features

        # ── Signal Pathway (identical to original) ──────────────────────
        self.cnn = SmallCNN(in_ch=1, channels=cnn_channels)

        self.post_conv_proj = nn.Conv1d(
            self.cnn.out_dim, 128, kernel_size=1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Feature Pathway (new, only if num_features > 0) ────────────
        signal_out_dim = 128  # from transformer global-avg pool

        if num_features > 0:
            self.feature_net = nn.Sequential(
                # LayerNorm first: features have wildly different scales
                # (HR ~80 bpm, QRS ~100 ms, p_wave_amplitude ~0.1 mV, sqi ~80).
                # Without normalizing here, the first Linear layer learns to ignore
                # small-scale features because their gradients are ~1000x smaller.
                nn.LayerNorm(num_features),
                nn.Linear(num_features, 64),
                nn.BatchNorm1d(64),  # BN before activation (standard: Linear→BN→ReLU)
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            fusion_in = signal_out_dim + 64  # 128 + 64 = 192
        else:
            self.feature_net = None
            fusion_in = signal_out_dim  # 128

        # ── Fusion + Classification Head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, features=None):
        """
        Parameters
        ----------
        x : Tensor
            Raw ECG signal, shape (B, L) or (B, 1, L).
        features : Tensor or None
            Auxiliary feature vector, shape (B, num_features).
            Pass None for signal-only mode.

        Returns
        -------
        Tensor
            Class logits, shape (B, num_classes).
        """
        # ── Signal pathway ──────────────────────────────────────────────
        z = self.cnn(x)                 # (B, C, T')
        z = self.post_conv_proj(z)      # (B, 128, T')
        z = z.permute(0, 2, 1)          # (B, T', 128)
        z = self.transformer_encoder(z) # (B, T', 128)
        z_sig = z.mean(dim=1)           # (B, 128) — global average pool

        # ── Feature pathway (if enabled) ────────────────────────────────
        if self.feature_net is not None and features is not None:
            z_feat = self.feature_net(features)  # (B, 64)
            z_fused = torch.cat([z_sig, z_feat], dim=1)  # (B, 192)
        else:
            z_fused = z_sig  # (B, 128)

        return self.classifier(z_fused)  # (B, num_classes)


def load_v2_from_v1_checkpoint(ckpt_path, num_classes, num_features, device="cpu"):
    """
    Load an OLD v1 checkpoint into the NEW v2 model.

    The signal pathway weights are identical between v1 and v2, so they
    transfer directly.  The feature_net and extended classifier are
    initialized randomly (they need training).

    Returns
    -------
    model : CNNTransformerWithFeatures
        Model with signal pathway loaded from v1, feature pathway random.
    metadata : dict
        Checkpoint metadata (epoch, balanced_acc, etc.).
    """
    model = CNNTransformerWithFeatures(
        num_classes=num_classes, num_features=num_features
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    v1_state = state["model_state"]

    # Load matching keys (signal pathway), skip new keys (feature_net, new classifier)
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in v1_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    print(f"[v2 loader] Loaded {len(loaded_keys)} keys from v1 checkpoint")
    if skipped_keys:
        print(f"[v2 loader] Skipped {len(skipped_keys)} keys (shape mismatch): {skipped_keys[:5]}")
    print(f"[v2 loader] Feature pathway + classifier head: randomly initialized (needs training)")

    return model, state
