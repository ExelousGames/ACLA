"""
Imitation Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service implements imitation learning algorithms to learn from expert driving demonstrations.
It focuses on learning optimal racing lines and decision-making patterns from
professional or expert drivers' telemetry data.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import io
import base64
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Scikit-learn imports for trajectory learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, r2_score

# Import your telemetry models
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor
from .tire_grip_analysis_service import TireGripFeatureCatalog

warnings.filterwarnings('ignore', category=UserWarning)


class ExpertPositionDataset(Dataset):
    """Torch dataset for expert position learning."""

    def __init__(
        self,
        inputs: np.ndarray,
        track_indices: np.ndarray,
        regression_targets: Optional[np.ndarray] = None,
        classification_targets: Optional[np.ndarray] = None,
    ):
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)

        self.inputs = inputs.astype(np.float32)
        self.track_indices = track_indices.astype(np.int64)

        if regression_targets is not None and regression_targets.size > 0:
            self.regression_targets = regression_targets.astype(np.float32)
            self.has_regression = True
        else:
            self.regression_targets = None
            self.has_regression = False

        if classification_targets is not None and classification_targets.size > 0:
            self.classification_targets = classification_targets.astype(np.int64)
            self.has_classification = True
        else:
            self.classification_targets = None
            self.has_classification = False

        if len(self.inputs) == 0:
            raise ValueError("ExpertPositionDataset cannot be empty")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {
            'inputs': torch.from_numpy(self.inputs[idx])
        }

        sample['track_idx'] = torch.tensor(self.track_indices[idx], dtype=torch.long)

        if self.has_regression:
            sample['regression_targets'] = torch.from_numpy(self.regression_targets[idx])

        if self.has_classification:
            sample['classification_targets'] = torch.tensor(
                self.classification_targets[idx], dtype=torch.long
            )

        return sample


class NeuralExpertModel(nn.Module):
    """Multi-head neural network for expert trajectory imitation."""

    def __init__(
        self,
        input_dim: int,
        track_vocab_size: int,
        regression_dim: int,
        classification_dim: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_layers is None or not hidden_layers:
            hidden_layers = [128, 128, 64]

        self.track_vocab_size = track_vocab_size
        self.regression_dim = regression_dim
        self.classification_dim = classification_dim

        embed_dim = 0
        if track_vocab_size > 1:
            embed_dim = max(4, min(32, track_vocab_size // 2 + 1))
            self.track_embedding = nn.Embedding(track_vocab_size, embed_dim)
        else:
            self.track_embedding = None

        layers: List[nn.Module] = []
        current_dim = input_dim + embed_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        if regression_dim > 0:
            self.regression_head = nn.Linear(current_dim, regression_dim)
        else:
            self.regression_head = None

        if classification_dim > 0:
            self.classification_head = nn.Linear(current_dim, classification_dim)
        else:
            self.classification_head = None

    def forward(
        self,
        inputs: torch.Tensor,
        track_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)

        if self.track_embedding is not None and track_idx is not None:
            embedded = self.track_embedding(track_idx)
            features = torch.cat([inputs, embedded], dim=1)
        else:
            features = inputs

        backbone_out = self.backbone(features)

        outputs: Dict[str, torch.Tensor] = {}
        if self.regression_head is not None:
            outputs['regression'] = self.regression_head(backbone_out)
        if self.classification_head is not None:
            outputs['classification'] = self.classification_head(backbone_out)

        return outputs

class ExpertFeatureCatalog:
    """Canonical expert feature names for downstream models.
    All expert state feature keys must be declared here and referenced via the Enum
    to avoid drifting string literals across the codebase.
    """

    class ExpertOptimalFeature(str, Enum):
        # Optimal action predictions 
        EXPERT_OPTIMAL_SPEED = 'expert_optimal_speed'
        EXPERT_OPTIMAL_STEERING = 'expert_optimal_steering'
        EXPERT_OPTIMAL_THROTTLE = 'expert_optimal_throttle'
        EXPERT_OPTIMAL_BRAKE = 'expert_optimal_brake'
        EXPERT_OPTIMAL_GEAR = 'expert_optimal_gear'
        EXPERT_OPTIMAL_PLAYER_POS_X = 'expert_optimal_player_pos_x'
        EXPERT_OPTIMAL_PLAYER_POS_Y = 'expert_optimal_player_pos_y'
        EXPERT_OPTIMAL_PLAYER_POS_Z = 'expert_optimal_player_pos_z'
        EXPERT_OPTIMAL_TRACK_POSITION = 'expert_optimal_track_position'
        EXPERT_OPTIMAL_VELOCITY_X = 'expert_optimal_velocity_x'
        EXPERT_OPTIMAL_VELOCITY_Y = 'expert_optimal_velocity_y'
        EXPERT_OPTIMAL_VELOCITY_Z = 'expert_optimal_velocity_z'

    class ContextFeature(str, Enum):
        # Velocity direction alignment with expert
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment' # 1.0 if moving in the expert velocity direction, 0.0 opposite direction
        SPEED_DIFFERENCE = 'speed_difference' # Difference between current speed and expert optimal speed (km/h)
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line' # distance between current position and expert optimal racing line (meters)

    # Flat list for convenience (now only expert optimal + derived)
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]

@dataclass(frozen=True)
class SegmentImprovementConfig:
    """Centralized thresholds and heuristics used during segment improvement analysis."""

    expert_velocity_alignment: float = 0.9
    expert_speed_diff_max: float = 5.0
    expert_distance_max: float = 3.0

    driver_push_high_threshold: float = 0.4
    driver_push_trend_min: float = 0.01

    smoothing_window_min: int = 2
    smoothing_window_max: int = 5
    ema_span_min: int = 2


@dataclass
class SegmentImprovementSummary:
    """Structured container for telemetry segment improvement analysis results."""

    velocity_alignment_mean: float = 0.0
    velocity_alignment_trend: float = 0.0
    velocity_consistency_rate: float = 0.0
    velocity_expert_points: int = 0

    speed_difference_mean: float = 0.0
    speed_difference_trend: float = 0.0
    speed_consistency_rate: float = 0.0
    speed_expert_points: int = 0
    distance_to_line_mean: float = 0.0
    distance_to_line_trend: float = 0.0
    distance_consistency_rate: float = 0.0
    distance_expert_points: int = 0

    driver_push_available: bool = False
    driver_push_mean: float = 0.0
    driver_push_trend: float = 0.0
    driver_push_high_rate: float = 0.0

    overall_improvement_rate: float = 0.0
    overall_consistency_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ExpertPositionLearner:
    """Learn expert actions based on normalized track position across all tracks with neural networks."""

    TRACK_COLUMN_CANDIDATES = [
        'Static_track',
        'track_name',
        'TrackName',
        'metadata_track_name',
        'Metadata_track_name',
    ]

    def __init__(self):
        self.position_model: Optional[Dict[str, Any]] = None
        self.position_scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._neural_model: Optional[NeuralExpertModel] = None

    @staticmethod
    def _extract_track_series(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=str)

        for column in ExpertPositionLearner.TRACK_COLUMN_CANDIDATES:
            if column in df.columns:
                series = df[column].fillna('GLOBAL').astype(str)
                if series.str.strip().eq('').all():
                    continue
                series = series.replace('', 'GLOBAL')
                return series.reset_index(drop=True)

        # Fall back to a synthetic GLOBAL track label
        return pd.Series(['GLOBAL'] * len(df))

    def extract_position_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract normalized position input, targets, and track labels."""
        if df.empty:
            raise ValueError("Telemetry DataFrame is empty")

        input_features = pd.DataFrame()
        target_features = pd.DataFrame()

        if 'Graphics_normalized_car_position' in df.columns:
            input_features['normalized_position'] = df['Graphics_normalized_car_position']
        else:
            raise ValueError("Graphics_normalized_car_position not found - this is required for position-based learning")

        EO = ExpertFeatureCatalog.ExpertOptimalFeature

        # Expert action targets
        if 'Physics_steer_angle' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_STEERING.value] = df['Physics_steer_angle']
        if 'Physics_gas' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_THROTTLE.value] = df['Physics_gas']
        if 'Physics_brake' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_BRAKE.value] = df['Physics_brake']
        if 'Physics_gear' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_GEAR.value] = df['Physics_gear']

        # Positions
        if 'Graphics_player_pos_x' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_X.value] = df['Graphics_player_pos_x']
        if 'Graphics_player_pos_y' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = df['Graphics_player_pos_y']
        if 'Graphics_player_pos_z' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = df['Graphics_player_pos_z']

        # Velocities
        if 'Physics_velocity_x' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_X.value] = df['Physics_velocity_x']
        if 'Physics_velocity_y' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_Y.value] = df['Physics_velocity_y']
        if 'Physics_velocity_z' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_Z.value] = df['Physics_velocity_z']

        if 'Physics_speed_kmh' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_SPEED.value] = df['Physics_speed_kmh']

        if 'Graphics_normalized_car_position' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_TRACK_POSITION.value] = df['Graphics_normalized_car_position']

        input_features = input_features.fillna(0).reset_index(drop=True)
        target_features = target_features.fillna(0).reset_index(drop=True)
        track_series = self._extract_track_series(df)

        return {
            'input_features': input_features,
            'target_features': target_features,
            'track_labels': track_series,
        }

    def _prepare_targets(self, target_features: pd.DataFrame) -> Tuple[
        Optional[np.ndarray],
        List[str],
        Dict[str, StandardScaler],
        Optional[np.ndarray],
        Optional[str],
        Dict[int, int],
    ]:
        EO = ExpertFeatureCatalog.ExpertOptimalFeature

        regression_target_order: List[str] = []
        action_targets = [
            EO.EXPERT_OPTIMAL_STEERING.value,
            EO.EXPERT_OPTIMAL_THROTTLE.value,
            EO.EXPERT_OPTIMAL_BRAKE.value,
            EO.EXPERT_OPTIMAL_SPEED.value,
        ]
        position_targets = [
            EO.EXPERT_OPTIMAL_PLAYER_POS_X.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value,
        ]
        velocity_targets = [
            EO.EXPERT_OPTIMAL_VELOCITY_X.value,
            EO.EXPERT_OPTIMAL_VELOCITY_Y.value,
            EO.EXPERT_OPTIMAL_VELOCITY_Z.value,
        ]

        ordered_candidates = action_targets + position_targets + velocity_targets + [
            EO.EXPERT_OPTIMAL_TRACK_POSITION.value
        ]

        for target in ordered_candidates:
            if target in target_features.columns and target != EO.EXPERT_OPTIMAL_GEAR.value:
                regression_target_order.append(target)

        regression_matrix: Optional[np.ndarray] = None
        target_scalers: Dict[str, StandardScaler] = {}

        if regression_target_order:
            regression_matrix = np.zeros((len(target_features), len(regression_target_order)), dtype=np.float32)
            for idx, target_name in enumerate(regression_target_order):
                scaler = StandardScaler()
                values = target_features[[target_name]].values
                scaled = scaler.fit_transform(values)
                regression_matrix[:, idx] = scaled.flatten()
                target_scalers[target_name] = scaler

        classification_indices: Optional[np.ndarray] = None
        classification_target: Optional[str] = None
        index_to_label: Dict[int, int] = {}

        if EO.EXPERT_OPTIMAL_GEAR.value in target_features.columns:
            gear_values = target_features[EO.EXPERT_OPTIMAL_GEAR.value].fillna(0).astype(int)
            unique_gears = sorted({int(v) for v in gear_values.unique()}) or [0]
            class_to_index = {gear: idx for idx, gear in enumerate(unique_gears)}
            index_to_label = {idx: gear for gear, idx in class_to_index.items()}
            classification_indices = gear_values.map(class_to_index).to_numpy(dtype=np.int64)
            classification_target = EO.EXPERT_OPTIMAL_GEAR.value

        return (
            regression_matrix,
            regression_target_order,
            target_scalers,
            classification_indices,
            classification_target,
            index_to_label,
        )

    @staticmethod
    def _split_indices(
        total_size: int,
        classification_indices: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.arange(total_size)
        if total_size < 10:
            return indices, indices

        stratify = None
        if (
            classification_indices is not None
            and len(np.unique(classification_indices)) > 1
        ):
            stratify = classification_indices

        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        return train_idx, val_idx

    @staticmethod
    def _build_dataloader(dataset: ExpertPositionDataset, shuffle: bool) -> DataLoader:
        batch_size = min(256, max(8, len(dataset) // 4))
        if batch_size > len(dataset):
            batch_size = len(dataset)
        batch_size = max(batch_size, 1)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _instantiate_model(self, model_config: Dict[str, Any], state_dict: Optional[Dict[str, Any]] = None) -> NeuralExpertModel:
        regression_targets = model_config.get('regression_targets', [])
        classification_lookup = model_config.get('classification_index_to_label', {})

        model = NeuralExpertModel(
            input_dim=model_config['input_dim'],
            track_vocab_size=len(model_config.get('track_vocab', [])),
            regression_dim=len(regression_targets),
            classification_dim=len(classification_lookup) if model_config.get('classification_target') else 0,
            hidden_layers=model_config.get('hidden_layers', [128, 128, 64]),
        )

        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    def learn_expert_position_mapping(self, expert_df: pd.DataFrame) -> Dict[str, Any]:
        print(f"[INFO] Learning expert position mapping from {len(expert_df)} expert data points")

        features = self.extract_position_features(expert_df)
        input_features = features['input_features']
        target_features = features['target_features']
        track_labels = features['track_labels']

        if input_features.empty or target_features.empty:
            raise ValueError("Insufficient features extracted from expert telemetry")

        X = input_features[['normalized_position']].values.astype(np.float32)
        X_scaled = self.position_scaler.fit_transform(X)

        (
            regression_matrix,
            regression_targets,
            target_scalers,
            classification_indices,
            classification_target,
            index_to_label,
        ) = self._prepare_targets(target_features)

        track_series = track_labels.fillna('GLOBAL').astype(str)
        track_vocab = sorted(track_series.unique().tolist()) or ['GLOBAL']
        track_lookup = {name: idx for idx, name in enumerate(track_vocab)}
        track_indices = track_series.map(track_lookup).to_numpy(dtype=np.int64)

        train_idx, val_idx = self._split_indices(len(X_scaled), classification_indices)

        regression_train = regression_matrix[train_idx] if regression_matrix is not None else None
        regression_val = regression_matrix[val_idx] if regression_matrix is not None else None

        class_train = classification_indices[train_idx] if classification_indices is not None else None
        class_val = classification_indices[val_idx] if classification_indices is not None else None

        train_dataset = ExpertPositionDataset(
            X_scaled[train_idx],
            track_indices[train_idx],
            regression_targets=regression_train,
            classification_targets=class_train,
        )
        val_dataset = ExpertPositionDataset(
            X_scaled[val_idx],
            track_indices[val_idx],
            regression_targets=regression_val,
            classification_targets=class_val,
        )

        train_loader = self._build_dataloader(train_dataset, shuffle=True)
        val_loader = self._build_dataloader(val_dataset, shuffle=False)

        model_config = {
            'input_dim': X_scaled.shape[1],
            'track_vocab': track_vocab,
            'track_lookup': track_lookup,
            'default_track': track_vocab[0],
            'regression_targets': regression_targets,
            'classification_target': classification_target,
            'classification_index_to_label': {int(k): int(v) for k, v in index_to_label.items()},
            'hidden_layers': [128, 128, 64],
        }

        model = NeuralExpertModel(
            input_dim=model_config['input_dim'],
            track_vocab_size=len(track_vocab),
            regression_dim=len(regression_targets),
            classification_dim=len(index_to_label) if classification_target else 0,
            hidden_layers=model_config['hidden_layers'],
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion_regression = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        patience = 20
        max_epochs = 200

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            sample_count = 0

            for batch in train_loader:
                inputs = batch['inputs'].to(self.device)
                track_idx = batch['track_idx'].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs, track_idx)

                loss = 0.0

                if 'regression_targets' in batch and model.regression_head is not None:
                    regression_targets_tensor = batch['regression_targets'].to(self.device)
                    loss = loss + criterion_regression(outputs['regression'], regression_targets_tensor)

                if 'classification_targets' in batch and model.classification_head is not None:
                    classification_targets_tensor = batch['classification_targets'].to(self.device)
                    loss = loss + criterion_classification(outputs['classification'], classification_targets_tensor)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)

            if sample_count > 0:
                train_loss /= sample_count

            model.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['inputs'].to(self.device)
                    track_idx = batch['track_idx'].to(self.device)
                    outputs = model(inputs, track_idx)

                    batch_loss = 0.0
                    if 'regression_targets' in batch and model.regression_head is not None:
                        regression_targets_tensor = batch['regression_targets'].to(self.device)
                        batch_loss += criterion_regression(outputs['regression'], regression_targets_tensor)

                    if 'classification_targets' in batch and model.classification_head is not None:
                        classification_targets_tensor = batch['classification_targets'].to(self.device)
                        batch_loss += criterion_classification(outputs['classification'], classification_targets_tensor)

                    val_loss += batch_loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)

            if val_samples > 0:
                val_loss /= val_samples
            else:
                val_loss = train_loss

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        self._neural_model = model.eval()

        # Collect validation predictions for metrics
        regression_predictions = None
        classification_predictions = None
        if len(val_dataset) > 0:
            reg_outputs: List[np.ndarray] = []
            cls_outputs: List[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['inputs'].to(self.device)
                    track_idx = batch['track_idx'].to(self.device)
                    outputs = model(inputs, track_idx)

                    if model.regression_head is not None and 'regression' in outputs:
                        reg_outputs.append(outputs['regression'].cpu().numpy())

                    if model.classification_head is not None and 'classification' in outputs:
                        cls_outputs.append(torch.argmax(outputs['classification'], dim=1).cpu().numpy())

            if reg_outputs:
                regression_predictions = np.concatenate(reg_outputs, axis=0)
            if cls_outputs:
                classification_predictions = np.concatenate(cls_outputs, axis=0)

        performance_metrics: Dict[str, Dict[str, float]] = {}
        if regression_predictions is not None and regression_targets:
            for idx, target_name in enumerate(regression_targets):
                y_true = target_features.iloc[val_idx][target_name].to_numpy()
                scaler = target_scalers[target_name]
                y_pred_scaled = regression_predictions[:, idx].reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

                try:
                    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
                except Exception:
                    r2 = 0.0

                performance_metrics[target_name] = {
                    'type': 'regression',
                    'r2': r2,
                    'mse': float(mean_squared_error(y_true, y_pred)),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                }

        if (
            classification_target
            and classification_predictions is not None
            and classification_indices is not None
        ):
            y_true_indices = classification_indices[val_idx]
            index_to_label_map = model_config['classification_index_to_label']
            y_true_labels = [index_to_label_map.get(int(idx), 0) for idx in y_true_indices]
            y_pred_labels = [index_to_label_map.get(int(idx), 0) for idx in classification_predictions]

            try:
                accuracy = float(accuracy_score(y_true_labels, y_pred_labels))
            except Exception:
                accuracy = 0.0

            try:
                f1 = float(f1_score(y_true_labels, y_pred_labels, average='weighted'))
            except Exception:
                f1 = 0.0

            performance_metrics[classification_target] = {
                'type': 'classification',
                'accuracy': accuracy,
                'f1': f1,
            }

        self.position_model = {
            'model_state_dict': best_state,
            'model_config': model_config,
            'position_scaler': self.position_scaler,
            'target_scalers': target_scalers,
            'performance_metrics': performance_metrics,
            'input_features': ['normalized_position'],
            'target_features': list(target_features.columns),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
        }

        return {
            'modelData': self.position_model,
            'metadata': {
                'performance_metrics': performance_metrics,
                'input_features': ['normalized_position'],
                'target_features': list(target_features.columns),
                'models_trained': regression_targets + ([classification_target] if classification_target else []),
                'total_training_samples': len(expert_df),
                'track_vocab_size': len(track_vocab),
                'tracks': track_vocab,
            },
        }

    def _resolve_track_indices(
        self,
        track_names: Optional[Union[str, List[str]]],
        batch_size: int,
    ) -> np.ndarray:
        if not self.position_model:
            raise ValueError("Position model not trained")

        config = self.position_model['model_config']
        track_lookup = config.get('track_lookup', {})
        default_track = config.get('default_track', 'GLOBAL')

        if track_names is None:
            track_names = [default_track] * batch_size
        elif isinstance(track_names, str):
            track_names = [track_names] * batch_size
        elif len(track_names) == 1 and batch_size > 1:
            track_names = track_names * batch_size

        resolved = []
        for name in track_names:
            resolved.append(track_lookup.get(str(name), track_lookup.get(default_track, 0)))

        return np.array(resolved, dtype=np.int64)

    def _ensure_model(self) -> NeuralExpertModel:
        if not self.position_model:
            raise ValueError("Position model not trained")

        if self._neural_model is None:
            state_dict = self.position_model['model_state_dict']
            model_config = self.position_model['model_config']
            model = self._instantiate_model(model_config, state_dict)
            self._neural_model = model
        else:
            self._neural_model.to(self.device)
            self._neural_model.eval()

        return self._neural_model

    def predict_expert_actions_at_position(
        self,
        normalized_positions: Union[float, List[float], np.ndarray],
        track_names: Optional[Union[str, List[str]]] = None,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        if not self.position_model:
            raise ValueError("No position model trained. Call learn_expert_position_mapping() first.")

        single_position = isinstance(normalized_positions, (int, float))
        if single_position:
            positions_array = np.array([[normalized_positions]], dtype=np.float32)
        else:
            positions_array = np.array(normalized_positions, dtype=np.float32).reshape(-1, 1)

        positions_scaled = self.position_model['position_scaler'].transform(positions_array).astype(np.float32)
        track_indices = self._resolve_track_indices(track_names, positions_scaled.shape[0])

        model = self._ensure_model()

        inputs_tensor = torch.from_numpy(positions_scaled).to(self.device)
        track_tensor = torch.from_numpy(track_indices).to(self.device)

        regression_predictions: Dict[str, np.ndarray] = {}
        classification_predictions: Optional[np.ndarray] = None

        with torch.no_grad():
            outputs = model(inputs_tensor, track_tensor)

            if model.regression_head is not None and 'regression' in outputs:
                regression_output = outputs['regression'].cpu().numpy()
                for idx, target_name in enumerate(self.position_model['model_config'].get('regression_targets', [])):
                    scaler = self.position_model['target_scalers'][target_name]
                    regression_predictions[target_name] = scaler.inverse_transform(
                        regression_output[:, idx].reshape(-1, 1)
                    ).flatten()

            if model.classification_head is not None and 'classification' in outputs:
                logits = outputs['classification']
                classification_predictions = torch.argmax(logits, dim=1).cpu().numpy()

        results: List[Dict[str, float]] = []
        num_samples = positions_scaled.shape[0]
        index_to_label_map = self.position_model['model_config'].get('classification_index_to_label', {})
        classification_target = self.position_model['model_config'].get('classification_target')

        for sample_idx in range(num_samples):
            sample_result: Dict[str, float] = {}

            for target_name, values in regression_predictions.items():
                sample_result[target_name] = float(values[sample_idx])

            if classification_target and classification_predictions is not None:
                raw_label = index_to_label_map.get(int(classification_predictions[sample_idx]), 0)
                sample_result[classification_target] = float(raw_label)

            results.append(sample_result)

        if single_position:
            return results[0] if results else {}
        return results

    def debug_position_model(self) -> Dict[str, Any]:
        if not self.position_model:
            return {
                'status': 'No model available',
                'has_model': False,
                'error': 'Position model not trained yet',
            }

        config = self.position_model.get('model_config', {})
        return {
            'status': 'Model available',
            'has_model': True,
            'model_structure': {
                'tracks': config.get('track_vocab', []),
                'regression_targets': config.get('regression_targets', []),
                'classification_target': config.get('classification_target'),
                'input_dim': config.get('input_dim'),
                'hidden_layers': config.get('hidden_layers'),
                'performance_metrics': self.position_model.get('performance_metrics', {}),
            },
        }

    def validate_position_input(self, normalized_positions: Union[float, List[float], np.ndarray]) -> Dict[str, Any]:
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_analysis': {},
        }

        if not self.position_model:
            validation_results['valid'] = False
            validation_results['errors'].append("No position model trained")
            return validation_results

        if isinstance(normalized_positions, (int, float)):
            positions_array = np.array([normalized_positions], dtype=np.float32)
        else:
            positions_array = np.array(normalized_positions, dtype=np.float32).flatten()

        if positions_array.size == 0:
            validation_results['valid'] = False
            validation_results['errors'].append("No positions provided")
            return validation_results

        validation_results['input_analysis'] = {
            'count': int(positions_array.size),
            'min_value': float(np.min(positions_array)),
            'max_value': float(np.max(positions_array)),
            'mean_value': float(np.mean(positions_array)),
            'has_nan': bool(np.isnan(positions_array).any()),
            'has_inf': bool(np.isinf(positions_array).any()),
        }

        if np.isnan(positions_array).any():
            validation_results['valid'] = False
            validation_results['errors'].append("Input contains NaN values")
        if np.isinf(positions_array).any():
            validation_results['valid'] = False
            validation_results['errors'].append("Input contains infinite values")
        if np.any(positions_array < 0.0):
            validation_results['warnings'].append("Some positions are below 0.0; expected normalized range [0, 1]")
        if np.any(positions_array > 1.0):
            validation_results['warnings'].append("Some positions exceed 1.0; expected normalized range [0, 1]")

        return validation_results

    def validate_prediction_input(self, current_state: pd.DataFrame) -> Dict[str, Any]:
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_analysis': {},
        }

        if not self.position_model:
            validation_results['valid'] = False
            validation_results['errors'].append("No position model trained")
            return validation_results

        if current_state.empty:
            validation_results['valid'] = False
            validation_results['errors'].append("Empty telemetry state provided")
            return validation_results

        validation_results['input_analysis'] = {
            'shape': current_state.shape,
            'columns': list(current_state.columns),
        }

        if 'Graphics_normalized_car_position' not in current_state.columns:
            validation_results['valid'] = False
            validation_results['errors'].append("Graphics_normalized_car_position column missing")

        track_series = self._extract_track_series(current_state)
        if track_series.empty:
            validation_results['warnings'].append("No track metadata detected; default track profile will be used")

        return validation_results


class ExpertImitateLearningService:
    """Main imitation learning service that focuses on trajectory optimization"""
    
    def __init__(self, models_directory: str = "imitation_models"):
        """
        Initialize the imitation learning service
        
        Args:
            models_directory: Directory to save/load trained imitation models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.position_learner = ExpertPositionLearner()
        
        print(f"[INFO] ImitationLearningService initialized. Models directory: {self.models_directory}")
    
    def get_shared_data_cache(self):
        """Get shared data cache instance"""
        from .Training_data_cache_service import get_shared_data_cache
        return get_shared_data_cache()

    def train_ai_model(self, telemetry_data: List[Dict[str, Any]], learning_objectives: List[str] = None) -> Tuple[Dict[str, Any]]:
        """
        Learn from expert driving demonstrations
        
        Args:
            telemetry_data: List of expert telemetry data dictionaries
            learning_objectives: List of what to learn ('trajectory')
            
        Returns:
            Dictionary with trained models and learning insights, serialized objects and ready for storage
        """
        if learning_objectives is None:
            learning_objectives = ['trajectory']
        
        print(f"[INFO {self.__class__.__name__}] Learning from {len(telemetry_data)} expert demonstrations")
        print(f"[INFO {self.__class__.__name__}] Learning objectives: {learning_objectives}")

        # Convert to DataFrame
        telemetry_df = pd.DataFrame(telemetry_data)
        feature_processor = FeatureProcessor(telemetry_df)
        # Cleaned data
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        # Learn expert position mapping (this is the only learning model)
        if 'trajectory' in learning_objectives:
            print("[INFO] Learning expert position mapping...")
            results = self.position_learner.learn_expert_position_mapping(processed_df)
            results['learning_summary'] = self._generate_learning_summary(results)
        else:
            raise ValueError("No valid learning objectives provided. Expected 'trajectory'.")

        return results
    
    def predict_expert_actions(self, 
                             processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict what an expert would do in the current situation
        
        Args:
            processed_df: Current telemetry DataFrame (may be single row)
            
        Returns:
            Predicted expert actions and recommendations
        """
        predictions = {}
        
        # Check if models exist
        if not self.position_learner.position_model:
            print("[WARNING] No position model available")
            return {"error": "No trained models available"}
        
        try:
            
            # Extract normalized positions from the input data
            if 'Graphics_normalized_car_position' in processed_df.columns:
                normalized_positions = processed_df['Graphics_normalized_car_position'].values
                track_series = self.position_learner._extract_track_series(processed_df)
                optimal_actions = self.position_learner.predict_expert_actions_at_position(
                    normalized_positions,
                    track_names=track_series.tolist(),
                )
                
                # If multiple rows, average the predictions for consistency with old interface
                if isinstance(optimal_actions, list):
                    valid_actions = [action for action in optimal_actions if isinstance(action, dict)]

                    if not valid_actions:
                        predictions['optimal_actions'] = {
                            'error': 'Model returned no valid predictions for the provided telemetry'
                        }
                    else:
                        averaged_actions: Dict[str, float] = {}
                        all_keys = set().union(*(action.keys() for action in valid_actions)) if valid_actions else set()

                        for key in all_keys:
                            values = [action.get(key) for action in valid_actions if key in action and action.get(key) is not None]
                            if values:
                                averaged_actions[key] = float(np.mean(values))

                        if averaged_actions:
                            predictions['optimal_actions'] = averaged_actions
                        else:
                            predictions['optimal_actions'] = {
                                'error': 'Model predictions did not include actionable outputs'
                            }
                else:
                    predictions['optimal_actions'] = optimal_actions
            else:
                predictions['optimal_actions'] = {"error": "No normalized track position data available"}
                
        except Exception as e:
            raise Exception(f"[WARNING] Could not predict expert actions: {e}")
        
        # If no specific models are available, provide error
        if not predictions or all('error' in v for v in predictions.values() if isinstance(v, dict)):
            raise Exception("[Error] No valid model available for predictions")
        
        return predictions
    
    def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of learning results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'learning_completed': []
        }
        
        # Check if we have metadata
        if 'metadata' in results:
            summary['learning_completed'].append('position_learning')
            
            # Calculate average performance metric, handling both regression (r2) and classification (accuracy) models
            performance_metrics = results['metadata']['performance_metrics']
            
            # Separate regression and classification metrics
            r2_scores = [metrics['r2'] for metrics in performance_metrics.values() if 'r2' in metrics]
            accuracy_scores = [metrics['accuracy'] for metrics in performance_metrics.values() if 'accuracy' in metrics]
            
            # Calculate average scores
            avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            summary['position_summary'] = {
                'models_trained': len(results['metadata']['models_trained']),
                'input_features': len(results['metadata']['input_features']),
                'target_features': len(results['metadata']['target_features']),
                'avg_r2_score': avg_r2,
                'avg_accuracy_score': avg_accuracy,
                'regression_models': len(r2_scores),
                'classification_models': len(accuracy_scores),
                'total_training_samples': results['metadata']['total_training_samples']
            }
        
        return summary
 
    def extract_expert_state_for_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract the comparsion between current state and expert optimal state. it helps transformer model to understand the gap between non-expert and expert driver.

        Purpose:
            - Provide a clear comparison between the current telemetry state and the expert's optimal state.
            - Enable the transformer model to learn from the differences and improve non-expert driving behavior.

        Args:
            telemetry_data: List of cleaned telemetry records to predict on

        Returns:
            List of dictionaries, one per record, containing expert targets, delta-to-expert
            context metrics, and enriched driver/expert positional data for visualization.
        """
        
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        if not telemetry_data:
            return []
        if not self.position_learner.position_model:
            raise ValueError("No trained imitation models available. Train or load models before calling extract_expert_state_for_telemetry().")

        try:
            processed_df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to create DataFrame: {e}")

        expert_feature_rows: List[Dict[str, Any]] = []

        # Position models should already be loaded/deserialized
        if not self.position_learner.position_model:
            raise ValueError("Position model not loaded. Call train_ai_model() or deserialize_imitation_model() first.")

        def predict_expert_batch(batch_df: pd.DataFrame) -> List[Dict[str, float]]:
            """Predict expert actions for a batch of normalized positions - much faster than row-by-row"""
            if not self.position_learner.position_model:
                return [{} for _ in range(len(batch_df))]
            try:
                # Extract normalized positions from batch
                if 'Graphics_normalized_car_position' not in batch_df.columns:
                    raise ValueError("Graphics_normalized_car_position not found in batch data")
                
                normalized_positions = batch_df['Graphics_normalized_car_position'].values
                track_series = self.position_learner._extract_track_series(batch_df)
                batch_predictions = self.position_learner.predict_expert_actions_at_position(
                    normalized_positions,
                    track_names=track_series.tolist(),
                )
                return batch_predictions
            except Exception as e:
                raise Exception(f"Batch prediction failed: {e}")

        total_rows = len(processed_df)

        # OPTIMIZATION: Process in batches instead of row-by-row for massive speedup
        batch_size = min(1000, total_rows)  # Process up to 1000 rows at once

        # Process all data in batches
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            
            try:
                # Get batch DataFrame
                batch_df = processed_df.iloc[batch_start:batch_end]
                
                # Get predictions for entire batch at once
                batch_predictions = predict_expert_batch(batch_df)
                
                # Process each row in the batch
                for i, row_predictions in enumerate(batch_predictions):
                    row_features: Dict[str, Any] = {}
                    
                    # Only calculate velocity alignment with expert (no other features)
                    try:
                        current_row = batch_df.iloc[i]
                        # Current velocity values from telemetry
                        curr_velocity_x = float(current_row.get('Physics_velocity_x', 0.0))
                        curr_velocity_y = float(current_row.get('Physics_velocity_y', 0.0))
                        curr_velocity_z = float(current_row.get('Physics_velocity_z', 0.0))

                        # Expert optimal velocities from predictions (using ExpertOptimalFeature mapping)
                        EO = ExpertFeatureCatalog.ExpertOptimalFeature
                        exp_velocity_x = float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_X.value, curr_velocity_x))
                        exp_velocity_y = float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_Y.value, curr_velocity_y))
                        exp_velocity_z = float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_Z.value, curr_velocity_z))

                        # Calculate velocity alignment (dot product normalized)
                        # If moving in same direction as expert, alignment = 1.0
                        curr_velocity_vector = np.array([curr_velocity_x, curr_velocity_y, curr_velocity_z])
                        exp_velocity_vector = np.array([exp_velocity_x, exp_velocity_y, exp_velocity_z])
                        curr_velocity_magnitude = np.linalg.norm(curr_velocity_vector)
                        exp_velocity_magnitude = np.linalg.norm(exp_velocity_vector)
                        
                        if curr_velocity_magnitude > 1e-6 and exp_velocity_magnitude > 1e-6:
                            # Normalize vectors and calculate dot product
                            curr_velocity_norm = curr_velocity_vector / curr_velocity_magnitude
                            exp_velocity_norm = exp_velocity_vector / exp_velocity_magnitude
                            velocity_alignment = np.dot(curr_velocity_norm, exp_velocity_norm)
                        else:
                            velocity_alignment = 0.0

                        # Persist raw telemetry context for downstream visualization
                        current_pos_x = float(current_row.get('Graphics_player_pos_x', 0.0))
                        current_pos_y = float(current_row.get('Graphics_player_pos_y', 0.0))
                        current_pos_z = float(current_row.get('Graphics_player_pos_z', 0.0))
                        current_speed = float(current_row.get('Physics_speed_kmh', curr_velocity_magnitude))

                        # Store expert optimal predictions for visualization with safe fallbacks
                        expert_pos_x = float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, current_pos_x))
                        expert_pos_y = float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, current_pos_y))
                        expert_pos_z = float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value, current_pos_z))

                        row_features[EO.EXPERT_OPTIMAL_PLAYER_POS_X.value] = expert_pos_x
                        row_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = expert_pos_y
                        row_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = expert_pos_z

                        row_features[EO.EXPERT_OPTIMAL_VELOCITY_X.value] = exp_velocity_x
                        row_features[EO.EXPERT_OPTIMAL_VELOCITY_Y.value] = exp_velocity_y
                        row_features[EO.EXPERT_OPTIMAL_VELOCITY_Z.value] = exp_velocity_z

                        expert_speed = float(row_predictions.get(EO.EXPERT_OPTIMAL_SPEED.value, exp_velocity_magnitude))
                        row_features[EO.EXPERT_OPTIMAL_SPEED.value] = expert_speed

                        # Store only velocity alignment feature
                        row_features[ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value] = float(velocity_alignment)

                        # Calculate speed difference
                        speed_difference = expert_speed - current_speed
                        row_features[ContextFeature.SPEED_DIFFERENCE.value] = float(speed_difference)

                        # Calculate distance to expert line (negative if off to left, positive if off to right)
                        distance_to_expert_line = np.sqrt(
                            (expert_pos_x - current_pos_x) ** 2 +
                            (expert_pos_y - current_pos_y) ** 2 +
                            (expert_pos_z - current_pos_z) ** 2
                        )
                        row_features[ContextFeature.DISTANCE_TO_EXPERT_LINE.value] = float(distance_to_expert_line)

                    except Exception as _e:
                        raise Exception(f"Velocity alignment calculation failed: {_e}")

                    expert_feature_rows.append(row_features)
                    
            except Exception as e:
                raise Exception(f"[WARNING] Failed to process batch {batch_start}-{batch_end}: {e}")

        print(f"[INFO] Completed expert state extraction. Extracted features for {len(expert_feature_rows)} records")
        return expert_feature_rows
    
    def filter_optimal_telemetry_segments(
        self,
        telemetry_data: List[Dict[str, Any]],
        max_segment_length: int = 60,
        improvement_threshold: float = 0.55,
        consistency_threshold: float = 1.0,
        min_segment_length: int = 20,
        min_segments: int = 0,
        score_relaxation: float = 0.01,  # Keep a running best score and stop extending a window once the improvement/consistency score falls by more than
        tail_window_fraction: float = 0.25, # Fraction of the current candidate window to validate independently to avoid long, weak tails
    ) -> List[List[Dict[str, Any]]]:
        """
        Identify contiguous telemetry slices that demonstrate measurable improvement or
        sustained expert-level consistency.

        Segments are grown dynamically from each starting point until either the
        improvement or consistency criteria stops being satisfied, or the
        ``max_segment_length`` cap is reached. Only the portion of the telemetry
        that meets the selected criteria is returned, eliminating the fixed-length
        constraints used previously.

        Args:
            telemetry_data: Telemetry records enriched with context features.
            max_segment_length: Upper bound on the number of records a single
                segment may contain.
            improvement_threshold: Minimum overall improvement rate required for
                a segment to be accepted.
            consistency_threshold: Minimum overall consistency rate required for
                a segment to be accepted when improvement is below the threshold.
            min_segment_length: Smallest segment length to analyse before
                considering acceptance.
            min_segments: Minimum number of segments required; raises if the
                condition is not met.
            score_relaxation: Allowed drop in the passing score (0-1) after the
                best scoring point before the segment is closed. Tightening this
                value keeps segments focused around their strongest portion.
            tail_window_fraction: Fraction of the current candidate window to
                validate independently to avoid long, weak tails from being
                appended to an otherwise strong segment.

        Returns:
            A list of telemetry segments, where each segment is a list of
            dictionaries corresponding to contiguous telemetry samples.
        """

        print(f"[INFO] Filtering optimal telemetry segments from {len(telemetry_data)} records...")
        print(
            "[INFO] Using max_segment_length=%s, min_segment_length=%s, improvement_threshold=%.2f, consistency_threshold=%.2f"
            % (max_segment_length, min_segment_length, improvement_threshold, consistency_threshold)
        )

        if max_segment_length < min_segment_length:
            raise ValueError("max_segment_length must be greater than or equal to min_segment_length")

        if len(telemetry_data) < min_segment_length:
            print(
                f"[WARNING] Insufficient data for segment analysis. Need at least {min_segment_length} records, got {len(telemetry_data)}. Discarding this batch."
            )
            return []

        # Get context feature names from enum
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        required_features = [
            ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value,
            ContextFeature.SPEED_DIFFERENCE.value,
            ContextFeature.DISTANCE_TO_EXPERT_LINE.value,
        ]

        # Validate that required features exist in data
        if not telemetry_data:
            print("[WARNING] Empty telemetry data provided")
            return []

        first_record = telemetry_data[0]
        missing_features = [f for f in required_features if f not in first_record]
        if missing_features:
            raise ValueError(
                f"[ERROR] Missing required context features: {missing_features}, available: {list(first_record.keys())}"
            )

        # Convert to DataFrame for easier analysis
        try:
            df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to convert telemetry data to DataFrame: {e}")

        optimal_segments: List[List[Dict[str, Any]]] = []
        num_improvement_segments = 0
        num_consistency_segments = 0
        window_evaluations = 0

        score_relaxation = max(0.0, min(0.5, score_relaxation))
        tail_window_fraction = float(np.clip(tail_window_fraction, 0.05, 0.5))

        idx = 0
        total_records = len(df)
        while idx < total_records:
            remaining = total_records - idx
            if remaining < min_segment_length:
                break

            last_valid_end: Optional[int] = None
            last_pass_type: Optional[str] = None
            best_score: Optional[float] = None
            best_pass_type: Optional[str] = None
            max_end_index = min(total_records, idx + max_segment_length)
            evaluation_started = False

            for end_idx in range(idx + min_segment_length - 1, max_end_index):
                segment = df.iloc[idx : end_idx + 1]
                window_evaluations += 1
                evaluation_started = True

                summary = self._analyze_segment_improvement(segment)
                passes_improvement = summary.overall_improvement_rate >= improvement_threshold
                passes_consistency = summary.overall_consistency_rate >= consistency_threshold

                if passes_improvement or passes_consistency:
                    current_score = max(
                        summary.overall_improvement_rate if passes_improvement else 0.0,
                        summary.overall_consistency_rate if passes_consistency else 0.0,
                    )
                    current_pass_type = (
                        "improvement"
                        if passes_improvement and summary.overall_improvement_rate >= summary.overall_consistency_rate
                        else "consistency"
                    )

                    if not self._segment_tail_meets_criteria(
                        segment,
                        improvement_threshold,
                        consistency_threshold,
                        min_segment_length,
                        tail_window_fraction,
                    ):
                        if last_valid_end is not None:
                            break
                        idx += 1
                        break

                    if best_score is None or current_score > best_score:
                        best_score = current_score
                        best_pass_type = current_pass_type
                        last_valid_end = end_idx
                        last_pass_type = current_pass_type
                    elif current_score >= (best_score or 0.0) - score_relaxation:
                        last_valid_end = end_idx
                        last_pass_type = current_pass_type
                    else:
                        break
                else:
                    if last_valid_end is not None:
                        break
                    idx += 1
                    break
            else:
                if not evaluation_started:
                    idx += 1
                    continue

            if last_valid_end is not None:
                segment_df = df.iloc[idx : last_valid_end + 1]
                optimal_segments.append(segment_df.to_dict("records"))

                if (last_pass_type or best_pass_type) == "improvement":
                    num_improvement_segments += 1
                else:
                    num_consistency_segments += 1

                idx = last_valid_end + 1
            else:
                if evaluation_started:
                    continue
                idx += 1

        print("[INFO] Dynamic segment filtering analysis complete:")
        print(f"[INFO] - Original records: {len(telemetry_data)}")
        print(f"[INFO] - Windows evaluated: {window_evaluations}")
        print(f"[INFO] - Accepted segments: {len(optimal_segments)}")
        print(
            f"[INFO] - Improvement-based passes: {num_improvement_segments}, Consistency-based passes: {num_consistency_segments}"
        )

        # Ensure we have minimum required segments
        if len(optimal_segments) < min_segments:
            raise ValueError(
                f"[WARNING] Only found {len(optimal_segments)} optimal segments, which is less than the minimum required {min_segments}. Adjust parameters or provide more data."
            )

        return optimal_segments
    
    def _analyze_segment_improvement(self, segment: pd.DataFrame) -> SegmentImprovementSummary:
        """
        Analyze improvement trends vs consistency within a telemetry segment.

        Returns a structured dataclass that retains the original dictionary keys for
        backwards compatibility while providing attribute access and helper
        utilities.
        """

        config = SegmentImprovementConfig()
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        summary = SegmentImprovementSummary()

        smoothing_window = max(config.smoothing_window_min, min(config.smoothing_window_max, len(segment)))
        ema_span = max(config.ema_span_min, smoothing_window)

        def _smooth_series(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
            series = values if isinstance(values, pd.Series) else pd.Series(values)
            if len(series) <= 1:
                return series.to_numpy()

            median_smoothed = series.rolling(window=smoothing_window, min_periods=1, center=True).median()
            ema_smoothed = median_smoothed.ewm(span=ema_span, adjust=False).mean()
            return ema_smoothed.to_numpy()

        try:
            # Velocity alignment analysis
            velocity_series = segment[ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value]
            velocity_smoothed = _smooth_series(velocity_series)
            if len(velocity_smoothed) > 1:
                summary.velocity_alignment_mean = float(np.mean(velocity_smoothed))
                summary.velocity_alignment_trend = float(np.polyfit(range(len(velocity_smoothed)), velocity_smoothed, 1)[0])
                summary.velocity_expert_points = int(np.sum(velocity_smoothed >= config.expert_velocity_alignment))
                summary.velocity_consistency_rate = summary.velocity_expert_points / len(velocity_smoothed)

            # Speed difference analysis
            speed_diff_raw = segment[ContextFeature.SPEED_DIFFERENCE.value]
            speed_diff_smoothed = _smooth_series(speed_diff_raw)
            speed_has_samples = len(speed_diff_smoothed) > 1
            if speed_has_samples:
                abs_speed_diff = np.abs(speed_diff_smoothed)
                summary.speed_difference_mean = float(np.mean(abs_speed_diff))
                summary.speed_difference_trend = float(np.polyfit(range(len(abs_speed_diff)), abs_speed_diff, 1)[0])
                summary.speed_expert_points = int(np.sum(abs_speed_diff <= config.expert_speed_diff_max))
                summary.speed_consistency_rate = summary.speed_expert_points / len(abs_speed_diff)

            # Distance to line analysis
            distance_series = segment[ContextFeature.DISTANCE_TO_EXPERT_LINE.value]
            distance_smoothed = _smooth_series(distance_series)
            distance_has_samples = len(distance_smoothed) > 1
            if distance_has_samples:
                summary.distance_to_line_mean = float(np.mean(distance_smoothed))
                summary.distance_to_line_trend = float(np.polyfit(range(len(distance_smoothed)), distance_smoothed, 1)[0])
                summary.distance_expert_points = int(np.sum(distance_smoothed <= config.expert_distance_max))
                summary.distance_consistency_rate = summary.distance_expert_points / len(distance_smoothed)

            # Driver push-to-limit analysis (0-1 intensity provided by TireGripAnalysisService)
            tire_feature = TireGripFeatureCatalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value
            if tire_feature in segment.columns:
                push_series = pd.to_numeric(segment[tire_feature], errors='coerce').fillna(0.0)
                push_smoothed = _smooth_series(push_series)
                if len(push_smoothed) > 1:
                    summary.driver_push_available = True
                    summary.driver_push_mean = float(np.mean(push_smoothed))
                    sample_idx = np.arange(len(push_smoothed))
                    summary.driver_push_trend = float(np.polyfit(sample_idx, push_smoothed, 1)[0])
                    push_above_threshold_rate = float(np.mean(push_smoothed >= config.driver_push_high_threshold))
                    summary.driver_push_high_rate = push_above_threshold_rate
            else:
                summary.driver_push_available = False

            # Improvement and consistency calculations
            distance_improvement = distance_has_samples and summary.distance_to_line_trend < 0.0
            speed_improvement = speed_has_samples and summary.speed_difference_trend < 0.0
            
            improvement_criteria: List[bool] = [distance_improvement, speed_improvement]

            base_improvement_rate = 0.0
            if improvement_criteria:
                base_improvement_rate = sum(improvement_criteria) / len(improvement_criteria)

            if summary.driver_push_available:
                push_threshold_rate = float(np.clip(summary.driver_push_high_rate, 0.0, 1.0))
                base_improvement_rate *= push_threshold_rate

            summary.overall_improvement_rate = base_improvement_rate

            consistency_rates = [
                summary.velocity_consistency_rate,
                summary.speed_consistency_rate,
                summary.distance_consistency_rate,
            ]

            if consistency_rates:
                summary.overall_consistency_rate = sum(consistency_rates) / len(consistency_rates)

        except Exception as e:
            raise Exception(f"Error analyzing segment improvement: {e}")

        return summary

    def _segment_tail_meets_criteria(
        self,
        segment: pd.DataFrame,
        improvement_threshold: float,
        consistency_threshold: float,
        min_segment_length: int,
        tail_window_fraction: float,
    ) -> bool:
        """Ensure the trailing portion of a candidate window independently satisfies acceptance heuristics."""

        segment_length = len(segment)
        if segment_length < max(3, min_segment_length):
            return True

        tail_window = max(
            3,
            int(segment_length * tail_window_fraction),
            min_segment_length // 2,
        )

        if tail_window >= segment_length:
            tail_window = max(3, segment_length // 2)

        if tail_window <= 2:
            return True

        tail_segment = segment.iloc[-tail_window:]
        tail_summary = self._analyze_segment_improvement(tail_segment)

        tail_improvement = tail_summary.overall_improvement_rate >= improvement_threshold
        tail_consistency = tail_summary.overall_consistency_rate >= consistency_threshold

        return tail_improvement or tail_consistency
    
    # Visualization utilities moved to telemetry_segment_visualizer.visualize_optimal_segments

    def serialize_learning_model(self) -> Dict[str, Any]:
        """
        Memory-efficient serialization of trained models stored in the position learner
        
        Returns:
            Dictionary with serialized models ready for storage/transmission
        """
        if not self.position_learner.position_model:
            raise ValueError("No trained models available to serialize. Train models first.")
        
        print("[INFO] Serializing current position models (memory-efficient)...")
        
        try:
            model_state = self.position_learner.position_model.get('model_state_dict')
            if model_state is None:
                raise ValueError("Position model is missing state dictionary")

            cpu_state = {k: v.detach().cpu() for k, v in model_state.items()}

            serialized_target_scalers = {}
            for name, scaler in self.position_learner.position_model.get('target_scalers', {}).items():
                serialized_target_scalers[name] = self.serialize_data(scaler)

            result = {
                'model_state_dict': self.serialize_data(cpu_state),
                'position_scaler': self.serialize_data(self.position_learner.position_model['position_scaler']),
                'target_scalers': serialized_target_scalers,
                'model_config': self.position_learner.position_model.get('model_config', {}),
                'performance_metrics': self.position_learner.position_model.get('performance_metrics', {}),
                'input_features': self.position_learner.position_model.get('input_features', ['normalized_position']),
                'target_features': self.position_learner.position_model.get('target_features', []),
                'train_size': self.position_learner.position_model.get('train_size', 0),
                'val_size': self.position_learner.position_model.get('val_size', 0),
            }

            print("[INFO] Serialization completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to serialize imitation learning models: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    # Deserialize object inside 
    def deserialize_imitation_model(self, serialized_results: Dict[str, Any]) -> 'ExpertImitateLearningService':
        """
        Deserialize the serialized position models and load them directly into the position learner. After deserializing,
        the models are ready for immediate use in predictions.
        Args:
            serialized_results: Dictionary containing serialized models and metadata
            
        Returns:
            Self (ExpertImitateLearningService): The current instance with loaded models
        """
        try:
            print("[INFO] Deserializing imitation models...")
            
            # The serialized_results should be the direct position model structure
            if 'model_state_dict' not in serialized_results:
                raise ValueError("Serialized data missing model state")

            print("[INFO] Deserializing position model state...")
            state_dict = self.deserialize_data(serialized_results['model_state_dict'])

            position_scaler = self.deserialize_data(serialized_results['position_scaler']) if 'position_scaler' in serialized_results else StandardScaler()

            target_scalers_serialized = serialized_results.get('target_scalers', {})
            target_scalers = {
                name: self.deserialize_data(data) for name, data in target_scalers_serialized.items()
            }

            model_config = serialized_results.get('model_config', {})

            self.position_learner.position_model = {
                'model_state_dict': state_dict,
                'model_config': model_config,
                'position_scaler': position_scaler,
                'target_scalers': target_scalers,
                'performance_metrics': serialized_results.get('performance_metrics', {}),
                'input_features': serialized_results.get('input_features', ['normalized_position']),
                'target_features': serialized_results.get('target_features', []),
                'train_size': serialized_results.get('train_size', 0),
                'val_size': serialized_results.get('val_size', 0),
            }

            self.position_learner.position_scaler = position_scaler
            self.position_learner._neural_model = None

            print("[INFO] Successfully deserialized imitation learning model")
                
            return self
            
        except Exception as e:
            error_msg = f"Failed to deserialize imitation learning models: {str(e)}"
            raise RuntimeError(error_msg) from e
    

    # Memory-efficient serialize models function
    def serialize_data(self, data: any) -> str:
        """
        Memory-efficient serialization of trained models
        
        Args:
            data: Model data to serialize
            
        Returns:
            Serialized model data as base64 encoded string
        """
        try:
            # Use highest compression protocol for smaller output
            buffer = io.BytesIO()
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.seek(0)
            
            # Get raw bytes and immediately clear buffer to free memory
            raw_bytes = buffer.getvalue()
            buffer.close()  # Explicitly close to free memory
            
            # Encode to base64 in chunks to avoid memory spikes
            import binascii
            encoded_data = base64.b64encode(raw_bytes).decode('utf-8')
            
            # Clear intermediate data
            del raw_bytes
            import gc
            gc.collect()
            
            return encoded_data
                
        except Exception as e:
            print(f"[ERROR] Failed to serialize models: {e}")
            raise e
    
    def deserialize_data(self, model_data: str) -> Dict[str, Any]:
        """
        Memory-efficient deserialization of models from base64 string
        
        Args:
            model_data: Base64 encoded model data
        
        Returns:
            Dictionary containing deserialized models and metadata
        """
        try:
            # Decode from base64
            decoded_data = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize using pickle with memory-efficient buffer
            buffer = io.BytesIO(decoded_data)
            data_result = pickle.load(buffer)
            
            # Explicitly clean up memory
            buffer.close()
            del decoded_data
            import gc
            gc.collect()
            
            return data_result
            
        except Exception as e:
            raise Exception(f"Failed to deserialize imitation learning models: {str(e)}")
    
# Example usage and testing
if __name__ == "__main__":
    
    # Example workflow
    service = ExpertImitateLearningService()
    
    # 1. Train the model (this stores models in the class)
    # serialized_results = service.train_ai_model(expert_telemetry_data)

    # 3. Compare new telemetry with stored expert models
    # comparison = service.compare_telemetry_with_expert(incoming_telemetry)