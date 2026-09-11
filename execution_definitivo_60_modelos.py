# -*- coding: utf-8 -*-
# ============================================================================
# DEFINITIVE PAPER EXECUTION — 8 PARTITION METHODS / 60 ASSET–NETWORK PAIRS
# Deliverable ID: CRYPTO_PARTITIONS_8M_60C_V1_2026-08-11
# This is not the former schema-v2 controlled-protocol script.
# ============================================================================
"""
Script para descargar series financieras, generar particiones temporales,
entrenar modelos secuenciales, evaluar baselines estadísticos y exportar tablas,
gráficas, global_results.csv, pruebas inferenciales y metadatos reproducibles.
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback

PROCESS_START_PERF_COUNTER = time.perf_counter()
PROCESS_STARTED_AT_UTC = dt.datetime.now(dt.timezone.utc).isoformat()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("PYTHONHASHSEED", "123")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
if not hasattr(matplotlib.cm, 'register_cmap'):
    def register_cmap(name, cmap):
        pass
    matplotlib.cm.register_cmap = register_cmap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from codecarbon import OfflineEmissionsTracker
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tensorflow.keras.layers import Add, BatchNormalization, Dense, Dropout, Flatten, GRU, Input, LSTM, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

tf.config.experimental.enable_op_determinism()

CARBON_ACCOUNTING_BOUNDARY = (
    "All neural-network fitting required by one experimental configuration: "
    "one model.fit() for every train/validation/test fold defined by the selected "
    "temporal partitioning protocol. Each tracker starts immediately before its "
    "model.fit() and stops immediately afterwards. Data acquisition, cleaning, "
    "partition construction, scaler fitting, model construction, inference, "
    "metric computation, plotting, serialization, and Naive Persistence inference "
    "are excluded."
)
TUNING_CARBON_ACCOUNTING_BOUNDARY = (
    "Every model.fit call used to validate a hyperparameter candidate across "
    "the three calibration rounds and declared tuning seeds. Candidate "
    "construction, scaling, inference, scoring, checkpoint serialization, and "
    "the subsequent confirmatory experiment are excluded. Calibration "
    "emissions are reported separately and are never added to a model's "
    "experimental fold-fitting emissions."
)

RUN_SCHEMA_VERSION = 3
DELIVERABLE_ID = "CRYPTO_PARTITIONS_8M_60C_V1_2026-08-11"

SOFTWARE_DISTRIBUTIONS = (
    "codecarbon",
    "keras",
    "matplotlib",
    "numpy",
    "pandas",
    "ruptures",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tensorflow",
    "yfinance",
)

# The primary experiment uses one independently validated configuration for
# every asset/architecture pair. The matched-parameter protocol is retained
# only as an optional sensitivity analysis and must not be mixed with the
# individually tuned results.
ARCHITECTURE_PROTOCOL = "individually_tuned"
CONTROLLED_ARCHITECTURE_CONFIG = {
    "target_trainable_parameters": 50000,
    "parameter_tolerance_fraction": 0.10,
    "model_hidden_units": {
        "Multilayer Perceptron": 220,
        "LSTM": 64,
        "GRU": 74,
        "TRANSFORMER": 68,
        "BiLSTM": 45,
    },
    "recurrent_layers": 2,
    "mlp_hidden_layers": 2,
    "transformer_blocks": 2,
    "attention_heads": 4,
    "dropout": 0.10,
    "learning_rate": 0.001,
    "loss": "mae",
    "max_epochs": 150,
    "batch_size": 32,
    "early_stopping_patience": 15,
}
CHANGE_POINT_CONFIG = {
    "algorithm": "PELT",
    "cost_model": "rbf",
    "penalty": 10.0,
    "min_segment_size": 2,
    "jump": 5,
    "fit_scope": "method-specific pre-test prefix only; test horizon excluded",
}

optimal_hyperparams = {
    "Bitcoin": {
         "LSTM": {
             "units": [128, 64],
             "dropout": 0.15,
             "learning_rate": 0.0005,
             "epochs": 150,
             "extra_dense": True
         },
         "TRANSFORMER": {
             "d_model": 80,
             "num_heads": 4,
             "dropout": 0.1,
             "learning_rate": 0.0002,
             "epochs": 150
         }
    },
    "Ethereum": {
         "LSTM": {
             "units": [256, 128],
             "dropout": 0.05,
             "learning_rate": 0.0004,
             "epochs": 300,
             "extra_dense": True,
             "use_extra_layers": True
         },
         "Multilayer Perceptron": {
             "layers": [64, 32],
             "dropout": 0.10,
             "learning_rate": 0.0008,
             "epochs": 150
         }
    },
    "Litecoin": {
         "LSTM": {
             "units": [256, 128],
             "dropout": 0.05,
             "learning_rate": 0.0003,
             "epochs": 300,
             "extra_dense": True,
             "use_extra_layers": True
         },
         "Multilayer Perceptron": {
             "layers": [64, 32, 16],
             "dropout": 0.05,
             "learning_rate": 0.0005,
             "epochs": 150
         },
         "TRANSFORMER": {
             "d_model": 50,
             "num_heads": 2,
             "dropout": 0.05,
             "learning_rate": 0.0001,
             "epochs": 150
         }
    },
    "Bitcoin Cash": {
         "LSTM": {
             "units": [128, 64],
             "dropout": 0.15,
             "learning_rate": 0.0005,
             "epochs": 150,
             "extra_dense": True
         }
    },
    "Cardano": {
         "LSTM": {
             "units": [64, 32],
             "dropout": 0.10,
             "learning_rate": 0.0003,
             "epochs": 150,
             "extra_dense": True
         }
    },
    "Polkadot": {
         "LSTM": {
             "units": [128, 64],
             "dropout": 0.15,
             "learning_rate": 0.0005,
             "epochs": 150,
             "extra_dense": False
         }
    },
    "Solana": {
         "LSTM": {
             "units": [512, 256],
             "dropout": 0.05,
             "learning_rate": 0.0003,
             "epochs": 250,
             "extra_dense": True,
             "use_extra_layers": True
         },
         "Multilayer Perceptron": {
             "layers": [64, 32],
             "dropout": 0.10,
             "learning_rate": 0.001,
             "epochs": 150
         }
    },
    "Binance Coin": {
         "LSTM": {
             "units": [128, 64],
             "dropout": 0.10,
             "learning_rate": 0.0005,
             "epochs": 150,
             "extra_dense": True
         }
    },
    "TRON": {
         "LSTM": {
             "units": [256, 256],
             "dropout": 0.05,
             "learning_rate": 0.00035,
             "epochs": 150,
             "extra_dense": True,
             "use_extra_layers": True
         },
         "TRANSFORMER": {
             "d_model": 64,
             "num_heads": 4,
             "dropout": 0.1,
             "learning_rate": 0.0003,
             "epochs": 150
         }
    },
    "Monero": {
         "LSTM": {
             "units": [256, 128],
             "dropout": 0.10,
             "learning_rate": 0.0003,
             "epochs": 150,
             "extra_dense": True
         },
         "Multilayer Perceptron": {
             "layers": [64, 32],
             "dropout": 0.05,
             "learning_rate": 0.0005,
             "epochs": 150
         },
         "GRU": {
             "units": [256, 128],
             "dropout": 0.05,
             "learning_rate": 0.0003,
             "epochs": 150
         }
    },
    "Ripple": {
         "LSTM": {
             "units": [64, 32],
             "dropout": 0.05,
             "learning_rate": 0.0004,
             "epochs": 150,
             "extra_dense": True
         }
    },
    "Stellar": {
         "LSTM": {
             "units": [64, 32],
             "dropout": 0.05,
             "learning_rate": 0.0004,
             "epochs": 150,
             "extra_dense": True
         }
    }
}

# Expert initial candidates for the 40 asset/architecture pairs that were not
# explicitly configured in the supplied code. They are deliberately labelled
# as initial candidates rather than validated optima. The tuning workflow
# defined later evaluates all 60 pairs under one temporal protocol and writes a
# separate validated-hyperparameter file before the paper experiment is run.
EXPERT_INITIAL_HYPERPARAMS = {
    "Bitcoin": {
        "Multilayer Perceptron": {
            "layers": [128, 64], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [128, 64], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "BiLSTM": {
            "units": [64, 32], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
    },
    "Ethereum": {
        "GRU": {
            "units": [256, 128], "dropout": 0.10,
            "learning_rate": 0.0004, "epochs": 250,
        },
        "TRANSFORMER": {
            "d_model": 80, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 200,
        },
        "BiLSTM": {
            "units": [128, 64], "dropout": 0.10,
            "learning_rate": 0.0004, "epochs": 250,
        },
    },
    "Litecoin": {
        "GRU": {
            "units": [256, 128], "dropout": 0.05,
            "learning_rate": 0.0003, "epochs": 250,
        },
        "BiLSTM": {
            "units": [128, 64], "dropout": 0.05,
            "learning_rate": 0.0003, "epochs": 250,
        },
    },
    "Bitcoin Cash": {
        "Multilayer Perceptron": {
            "layers": [64, 32], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [128, 64], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 64, "num_heads": 4, "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "BiLSTM": {
            "units": [64, 32], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
    },
    "Cardano": {
        "Multilayer Perceptron": {
            "layers": [64, 32], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [64, 32], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 48, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
        "BiLSTM": {
            "units": [32, 16], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
    },
    "Polkadot": {
        "Multilayer Perceptron": {
            "layers": [64, 32], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [128, 64], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 64, "num_heads": 4, "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "BiLSTM": {
            "units": [64, 32], "dropout": 0.15,
            "learning_rate": 0.0005, "epochs": 150,
        },
    },
    "Solana": {
        "GRU": {
            "units": [512, 256], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 250,
        },
        "TRANSFORMER": {
            "d_model": 96, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 250,
        },
        "BiLSTM": {
            "units": [256, 128], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 250,
        },
    },
    "Binance Coin": {
        "Multilayer Perceptron": {
            "layers": [128, 64], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [128, 64], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 64, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "BiLSTM": {
            "units": [64, 32], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
    },
    "TRON": {
        "Multilayer Perceptron": {
            "layers": [128, 64, 32], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
        "GRU": {
            "units": [256, 128], "dropout": 0.05,
            "learning_rate": 0.00035, "epochs": 150,
        },
        "BiLSTM": {
            "units": [128, 64], "dropout": 0.05,
            "learning_rate": 0.00035, "epochs": 150,
        },
    },
    "Monero": {
        "TRANSFORMER": {
            "d_model": 64, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
        "BiLSTM": {
            "units": [128, 64], "dropout": 0.10,
            "learning_rate": 0.0003, "epochs": 150,
        },
    },
    "Ripple": {
        "Multilayer Perceptron": {
            "layers": [64, 32], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [64, 32], "dropout": 0.05,
            "learning_rate": 0.0004, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 48, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0004, "epochs": 150,
        },
        "BiLSTM": {
            "units": [32, 16], "dropout": 0.05,
            "learning_rate": 0.0004, "epochs": 150,
        },
    },
    "Stellar": {
        "Multilayer Perceptron": {
            "layers": [64, 32], "dropout": 0.10,
            "learning_rate": 0.0005, "epochs": 150,
        },
        "GRU": {
            "units": [64, 32], "dropout": 0.05,
            "learning_rate": 0.0004, "epochs": 150,
        },
        "TRANSFORMER": {
            "d_model": 48, "num_heads": 4, "dropout": 0.10,
            "learning_rate": 0.0004, "epochs": 150,
        },
        "BiLSTM": {
            "units": [32, 16], "dropout": 0.05,
            "learning_rate": 0.0004, "epochs": 150,
        },
    },
}

SUPPLIED_HYPERPARAMETER_PAIRS = {
    (asset_name, model_name)
    for asset_name, model_configs in optimal_hyperparams.items()
    for model_name in model_configs
}
for _asset_name, _model_configs in EXPERT_INITIAL_HYPERPARAMS.items():
    optimal_hyperparams.setdefault(_asset_name, {}).update(_model_configs)

HYPERPARAMETER_PROVENANCE = {
    (asset_name, model_name): (
        "supplied_configuration"
        if (asset_name, model_name) in SUPPLIED_HYPERPARAMETER_PAIRS
        else "expert_initial_candidate_requires_temporal_validation"
    )
    for asset_name, model_configs in optimal_hyperparams.items()
    for model_name in model_configs
}

# Filled from a rigorously generated validation manifest at runtime. A paper
# run using individually tuned architectures refuses unvalidated pairs unless
# the user explicitly selects the exploratory override.
HYPERPARAMETER_VALIDATION_RECORDS = {}

# %% Funciones de partición y creación de instancias
def crear_instancias(serie, tam_ventana):
    serie = np.asarray(serie)
    if len(serie) <= tam_ventana:
        return np.empty((0, tam_ventana)), np.array([]), np.array([], dtype=int)

    X = np.lib.stride_tricks.sliding_window_view(serie, tam_ventana)[:-1]
    y = serie[tam_ventana:]
    idx = np.arange(tam_ventana, len(serie))
    return X, y, idx

def _positions_between(length, start_fraction, end_fraction):
    """Return a non-empty half-open interval on a normalized time axis."""
    start = int(round(float(start_fraction) * length))
    end = int(round(float(end_fraction) * length))
    start = min(max(start, 0), length)
    end = min(max(end, 0), length)
    if end <= start:
        raise ValueError(
            f"Invalid normalized interval [{start_fraction}, {end_fraction}) "
            f"for a development sample of length {length}."
        )
    return np.arange(start, end, dtype=int)


def _make_fold(label, train_pos, val_pos, test_pos, purge_pos=None):
    fold = {
        "label": str(label),
        "train_pos": np.asarray(train_pos, dtype=int),
        "val_pos": np.asarray(val_pos, dtype=int),
        "test_pos": np.asarray(test_pos, dtype=int),
        "purge_pos": np.asarray(
            [] if purge_pos is None else purge_pos,
            dtype=int,
        ),
    }
    if any(
        values.size == 0
        for values in (
            fold["train_pos"],
            fold["val_pos"],
            fold["test_pos"],
        )
    ):
        raise ValueError(f"{label}: train, validation, and test must be non-empty.")
    return fold


def _contiguous_fold(
    development_size,
    label,
    train_bounds,
    validation_bounds,
    test_bounds,
):
    return _make_fold(
        label,
        _positions_between(development_size, *train_bounds),
        _positions_between(development_size, *validation_bounds),
        _positions_between(development_size, *test_bounds),
    )


def _information_overlap_positions(
    candidate_positions,
    held_out_positions,
    target_indices,
    window_size,
):
    """Return candidates whose [target-window, target] intervals overlap hold-out."""
    candidates = np.asarray(candidate_positions, dtype=int)
    held_out = np.asarray(held_out_positions, dtype=int)
    if candidates.size == 0 or held_out.size == 0:
        return np.array([], dtype=int)
    held_start = int(np.min(target_indices[held_out] - window_size))
    held_end = int(np.max(target_indices[held_out]))
    candidate_start = target_indices[candidates] - window_size
    candidate_end = target_indices[candidates]
    return candidates[
        (candidate_start <= held_end) & (candidate_end >= held_start)
    ]


def _choose_contiguous_validation(safe_positions, test_positions, requested_size):
    """Choose a deterministic contiguous validation block near one outer test."""
    safe = np.asarray(np.unique(safe_positions), dtype=int)
    test = np.asarray(test_positions, dtype=int)
    if safe.size < requested_size + 1:
        raise ValueError("Purged K-Fold leaves too few safe samples for validation.")
    runs = []
    run_start = 0
    for offset in np.flatnonzero(np.diff(safe) != 1) + 1:
        runs.append(safe[run_start:offset])
        run_start = offset
    runs.append(safe[run_start:])
    eligible = [run for run in runs if len(run) >= requested_size]
    if not eligible:
        raise ValueError("No contiguous Purged K-Fold validation block is large enough.")
    before = [run for run in eligible if int(run[-1]) < int(test[0])]
    if before:
        return before[-1][-requested_size:]
    after = [run for run in eligible if int(run[0]) > int(test[-1])]
    if after:
        return after[0][:requested_size]
    selected = max(eligible, key=len)
    return selected[-requested_size:]


def obtener_particion(serie, tam_ventana, metodo):
    """Build the method-specific train/validation/test folds shown in the protocol."""
    print(
        f"[INFO] {METODOS_MAP[metodo]}: generating method-specific folds "
        f"for series of length {len(serie)}"
    )
    X, _y, idx = crear_instancias(serie, tam_ventana)
    nb = len(X)
    if nb < 60:
        raise ValueError(
            f"{metodo}: at least 60 supervised instances are required; got {nb}."
        )
    folds = []
    method_metadata = {
        "timeline": "complete normalized supervised sample",
        "test_policy": (
            "method-specific red blocks are genuine out-of-sample tests; "
            "there is no additional external test horizon"
        ),
        "metric_aggregation": (
            "concatenate every fold's out-of-sample predictions and targets, "
            "sort by target index, then compute each metric once"
        ),
    }

    if metodo == "no_aleatoria":
        folds.append(_contiguous_fold(
            nb, "chronological_holdout",
            (0.00, 0.70), (0.70, 0.85), (0.85, 1.00),
        ))
    elif metodo == "ventana_movil2":
        schedules = (
            ((0.00, 0.40), (0.40, 0.50), (0.50, 0.60)),
            ((0.00, 0.60), (0.60, 0.70), (0.70, 0.80)),
            ((0.00, 0.80), (0.80, 0.90), (0.90, 1.00)),
        )
        for number, bounds in enumerate(schedules, start=1):
            folds.append(_contiguous_fold(nb, f"expanding_round_{number}", *bounds))
    elif metodo == "con_solape":
        schedules = (
            ((0.00, 0.30), (0.30, 0.40), (0.40, 0.50)),
            ((0.20, 0.50), (0.50, 0.60), (0.60, 0.70)),
            ((0.40, 0.70), (0.70, 0.80), (0.80, 0.90)),
        )
        for number, bounds in enumerate(schedules, start=1):
            folds.append(_contiguous_fold(nb, f"rolling_round_{number}", *bounds))
    elif metodo == "time_series_split":
        schedules = (
            ((0.00, 0.30), (0.30, 0.40), (0.40, 0.50)),
            ((0.10, 0.40), (0.40, 0.50), (0.50, 0.60)),
            ((0.20, 0.50), (0.50, 0.60), (0.60, 0.70)),
        )
        for number, bounds in enumerate(schedules, start=1):
            folds.append(_contiguous_fold(nb, f"fixed_window_round_{number}", *bounds))
    elif metodo == "purged_kfold":
        all_positions = np.arange(nb, dtype=int)
        test_folds = [np.asarray(values, dtype=int) for values in np.array_split(all_positions, 3)]
        embargo_size = max(1, int(round(0.01 * nb)))
        validation_size = max(tam_ventana + 1, int(round(0.10 * nb)))
        for number, test_pos in enumerate(test_folds, start=1):
            outer_candidates = np.setdiff1d(all_positions, test_pos, assume_unique=True)
            test_purge = _information_overlap_positions(
                outer_candidates, test_pos, idx, tam_ventana
            )
            embargo_start = int(test_pos[-1]) + 1
            embargo_pos = np.arange(
                embargo_start,
                min(nb, embargo_start + embargo_size),
                dtype=int,
            )
            safe_for_validation = np.setdiff1d(
                outer_candidates,
                np.unique(np.concatenate([test_purge, embargo_pos])),
                assume_unique=True,
            )
            val_pos = _choose_contiguous_validation(
                safe_for_validation, test_pos, validation_size
            )
            remaining = np.setdiff1d(safe_for_validation, val_pos, assume_unique=True)
            validation_purge = _information_overlap_positions(
                remaining, val_pos, idx, tam_ventana
            )
            train_pos = np.setdiff1d(remaining, validation_purge, assume_unique=True)
            purge_pos = np.unique(np.concatenate([test_purge, validation_purge]))
            fold = _make_fold(
                f"purged_fold_{number}",
                train_pos,
                val_pos,
                test_pos,
                purge_pos,
            )
            fold["embargo_pos"] = embargo_pos
            folds.append(fold)
        method_metadata.update({
            "n_folds": 3,
            "test_coverage": "three contiguous test folds cover every supervised target once",
            "sample_information_interval": f"[t-{tam_ventana}, t]",
            "purging_rule": "remove training information intervals overlapping test or validation",
            "embargo_fraction_after_test": 0.01,
            "embargo_size_target_positions": embargo_size,
            "validation_rule": "contiguous safe training block nearest the outer test",
            "adaptation_note": (
                "López de Prado Purged K-Fold adapted from event-label intervals "
                "to overlapping lag-information intervals"
            ),
        })
    elif metodo == "change_point_partitioning":
        test_pos = _positions_between(nb, 0.75, 1.00)
        test_start = int(test_pos[0])
        first_test_raw_index = int(idx[test_start])
        nominal_training_position = int(round(0.40 * nb))
        nominal_training_raw_index = int(idx[nominal_training_position])
        pretest_series = np.asarray(serie[:first_test_raw_index], dtype=np.float64)
        algorithm = rpt.Pelt(
            model=CHANGE_POINT_CONFIG["cost_model"],
            min_size=CHANGE_POINT_CONFIG["min_segment_size"],
            jump=CHANGE_POINT_CONFIG["jump"],
        ).fit(pretest_series)
        detected_raw = [
            int(value)
            for value in algorithm.predict(pen=CHANGE_POINT_CONFIG["penalty"])
        ]
        minimum_train = max(tam_ventana + 1, int(round(0.20 * nb)))
        minimum_validation = max(tam_ventana + 1, int(round(0.10 * nb)))
        maximum_train = test_start - minimum_validation
        candidates = []
        for raw_index in detected_raw:
            position = int(np.searchsorted(idx, raw_index, side="left"))
            if minimum_train <= position <= maximum_train:
                candidates.append(raw_index)
        fallback_used = not candidates
        selected_candidate = (
            min(candidates, key=lambda value: abs(value - nominal_training_raw_index))
            if candidates else nominal_training_raw_index
        )
        selected_position = int(np.searchsorted(idx, selected_candidate, side="left"))
        selected_position = min(max(selected_position, minimum_train), maximum_train)
        folds.append(_make_fold(
            "pelt_pretest_split",
            np.arange(0, selected_position, dtype=int),
            np.arange(selected_position, test_start, dtype=int),
            test_pos,
        ))
        method_metadata.update({
            "pelt_configuration": dict(CHANGE_POINT_CONFIG),
            "pelt_fit_raw_interval": [0, first_test_raw_index],
            "test_fraction": 0.25,
            "detected_change_points_raw_indices": detected_raw,
            "admissible_change_points_raw_indices": candidates,
            "selected_change_point_raw_index": int(idx[selected_position]),
            "selected_change_point_supervised_position": selected_position,
            "nominal_training_boundary_fraction": 0.40,
            "fallback_to_nominal_boundary": bool(fallback_used),
        })
    elif metodo == "pre-paper_no_shuffle":
        folds.append(_contiguous_fold(
            nb, "sequential_without_shuffle",
            (0.00, 0.60), (0.60, 0.80), (0.80, 1.00),
        ))
    elif metodo == "timeseries_generator":
        schedules = (
            ((0.00, 0.30), (0.30, 0.40), (0.40, 0.50)),
            ((0.20, 0.60), (0.60, 0.65), (0.65, 0.75)),
        )
        for number, bounds in enumerate(schedules, start=1):
            folds.append(_contiguous_fold(nb, f"generator_round_{number}", *bounds))
        method_metadata["generation_rule"] = (
            "two predefined causal temporal scenarios on the complete normalized timeline"
        )
    else:
        raise ValueError(f"Unknown partition method: {metodo}")

    all_test_pos = np.concatenate([
        np.asarray(fold["test_pos"], dtype=int) for fold in folds
    ])
    if len(np.unique(all_test_pos)) != len(all_test_pos):
        raise RuntimeError(f"{metodo}: a target appears in more than one test fold.")
    plan = {
        "method": metodo,
        "folds": folds,
        "all_test_pos": np.sort(all_test_pos),
        "test_size": int(len(all_test_pos)),
        "n_supervised": int(nb),
        "window_size": int(tam_ventana),
        "metadata": method_metadata,
    }
    validate_causal_partition(serie, plan, metodo)
    return plan


def fit_scaler_from_training(X_train, y_train):
    """Fit one value scaler using observations available to the fit only."""
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot fit a scaler on an empty training subset.")
    train_values = np.concatenate([
        np.asarray(X_train, dtype=np.float64).reshape(-1),
        np.asarray(y_train, dtype=np.float64).reshape(-1),
    ])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values.reshape(-1, 1))
    return scaler


def transform_X_with_scaler(scaler, values):
    values = np.asarray(values, dtype=np.float64)
    return scaler.transform(values.reshape(-1, 1)).reshape(values.shape)


def transform_y_with_scaler(scaler, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return scaler.transform(values.reshape(-1, 1)).reshape(-1)


def scale_partition_from_training(partition):
    """Fit MinMax scaling on fold-training values and transform all splits."""
    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        idx_train, idx_val, idx_test
    ) = partition
    if any(len(part) == 0 for part in (X_train, X_val, X_test)):
        raise ValueError("Cannot scale a partition containing an empty split.")

    # Values outside the fold-training range are intentionally allowed to
    # transform below zero or above one.
    scaler = fit_scaler_from_training(X_train, y_train)

    scaled_partition = (
        transform_X_with_scaler(scaler, X_train),
        transform_y_with_scaler(scaler, y_train),
        transform_X_with_scaler(scaler, X_val),
        transform_y_with_scaler(scaler, y_val),
        transform_X_with_scaler(scaler, X_test),
        transform_y_with_scaler(scaler, y_test),
        np.asarray(idx_train, dtype=int),
        np.asarray(idx_val, dtype=int),
        np.asarray(idx_test, dtype=int),
    )
    return scaled_partition, scaler


def validate_causal_partition(serie, plan, method):
    """Validate alignment and information isolation for method-specific folds."""
    window_size = int(plan.get("window_size", WINDOW_SIZE))
    X, y, idx = crear_instancias(serie, window_size)
    n_supervised = len(X)
    raw = np.asarray(serie, dtype=np.float64)

    def validate_positions(label, positions):
        positions = np.asarray(positions, dtype=int)
        if positions.size == 0:
            raise RuntimeError(f"{method}/{label}: subset is empty.")
        if np.any(positions < 0) or np.any(positions >= n_supervised):
            raise RuntimeError(f"{method}/{label}: target position is out of range.")
        target_indices = idx[positions]
        if not np.allclose(y[positions], raw[target_indices], rtol=0.0, atol=0.0):
            raise RuntimeError(f"{method}/{label}: target/index misalignment.")
        expected_windows = np.stack([
            raw[target_index - window_size:target_index]
            for target_index in target_indices
        ])
        if not np.allclose(X[positions], expected_windows, rtol=0.0, atol=0.0):
            raise RuntimeError(f"{method}/{label}: causal window misalignment.")
        return positions

    if not plan.get("folds"):
        raise RuntimeError(f"{method}: no train/validation/test folds were generated.")

    observed_test_positions = []
    for fold in plan["folds"]:
        label = fold["label"]
        train_pos = validate_positions(f"{label}/train", fold["train_pos"])
        val_pos = validate_positions(f"{label}/validation", fold["val_pos"])
        test_pos = validate_positions(
            f"{label}/test",
            fold["test_pos"],
        )
        observed_test_positions.append(test_pos)
        if (
            np.intersect1d(train_pos, val_pos).size
            or np.intersect1d(train_pos, test_pos).size
            or np.intersect1d(val_pos, test_pos).size
        ):
            raise RuntimeError(f"{method}/{label}: fold subsets overlap.")
        if method != "purged_kfold":
            if not (
                int(train_pos.max()) < int(val_pos.min())
                and int(val_pos.max()) < int(test_pos.min())
            ):
                raise RuntimeError(
                    f"{method}/{label}: non-purged fold is not chronologically ordered."
                )
        else:
            purge_pos = np.asarray(fold["purge_pos"], dtype=int)
            embargo_pos = np.asarray(fold.get("embargo_pos", []), dtype=int)
            expected_embargo_size = int(
                plan["metadata"]["embargo_size_target_positions"]
            )
            expected_embargo = np.arange(
                int(test_pos[-1]) + 1,
                min(
                    n_supervised,
                    int(test_pos[-1]) + 1 + expected_embargo_size,
                ),
                dtype=int,
            )
            if not np.array_equal(embargo_pos, expected_embargo):
                raise RuntimeError(
                    f"{method}/{label}: embargo positions do not match the "
                    "declared one-percent post-test rule."
                )
            if np.intersect1d(train_pos, purge_pos).size:
                raise RuntimeError(f"{method}/{label}: purged targets remain in training.")
            if np.intersect1d(train_pos, embargo_pos).size:
                raise RuntimeError(f"{method}/{label}: embargoed targets remain in training.")
            for held_out_name, held_out_pos in (
                ("validation", val_pos),
                ("test", test_pos),
            ):
                forbidden = _information_overlap_positions(
                    train_pos, held_out_pos, idx, window_size
                )
                if forbidden.size:
                    raise RuntimeError(
                        f"{method}/{label}: {len(forbidden)} training information "
                        f"intervals overlap the {held_out_name} information interval."
                    )

    observed_test = np.concatenate(observed_test_positions)
    if len(np.unique(observed_test)) != len(observed_test):
        raise RuntimeError(f"{method}: a supervised target is tested more than once.")
    declared_test = np.asarray(plan["all_test_pos"], dtype=int)
    if not np.array_equal(np.sort(observed_test), np.sort(declared_test)):
        raise RuntimeError(f"{method}: declared and observed test targets differ.")
    if int(plan["test_size"]) != len(observed_test):
        raise RuntimeError(f"{method}: declared test size is inconsistent.")
    if method == "purged_kfold" and not np.array_equal(
        np.sort(observed_test), np.arange(n_supervised, dtype=int)
    ):
        raise RuntimeError(
            "purged_kfold: the three outer test folds must cover every "
            "supervised target exactly once."
        )
    if method == "purged_kfold":
        if len(plan["folds"]) != 3:
            raise RuntimeError("purged_kfold: exactly three outer folds are required.")
        for fold in plan["folds"]:
            if np.any(np.diff(np.asarray(fold["test_pos"], dtype=int)) != 1):
                raise RuntimeError(
                    "purged_kfold: every outer test fold must be contiguous."
                )

def chronological_holdout(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "no_aleatoria")

def sequential_split_no_shuffle(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "pre-paper_no_shuffle")

def expanding_window_validation(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "ventana_movil2")

def overlapping_window_partitioning(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "con_solape")

def timeseries_split_fixed_cv(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "time_series_split")

def generator_based_temporal_partitioning(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "timeseries_generator")

def purged_kfold_cv(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "purged_kfold")

def changepoint_based_partitioning(serie, tam_ventana):
    return obtener_particion(serie, tam_ventana, "change_point_partitioning")

# %% BLOQUE DE EVALUACIÓN DE MODELOS
# Individually tuned asset/model configurations are the primary protocol.
# The matched-parameter controlled protocol remains an optional sensitivity run.
modelos = {
    "Multilayer Perceptron": None,  # Se construye a continuación
    "LSTM": None,
    "GRU": None,
    "TRANSFORMER": None,
    "BiLSTM": None
}

def construir_modelo_perceptron(tam_ventana, asset_name=None):
    if ARCHITECTURE_PROTOCOL == "controlled":
        units = CONTROLLED_ARCHITECTURE_CONFIG["model_hidden_units"]["Multilayer Perceptron"]
        dropout = CONTROLLED_ARCHITECTURE_CONFIG["dropout"]
        lr = CONTROLLED_ARCHITECTURE_CONFIG["learning_rate"]
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            Flatten(),
            Dense(units, activation="relu"),
            Dropout(dropout),
            Dense(units, activation="relu"),
            Dropout(dropout),
            Dense(1, activation="linear"),
        ])
    elif asset_name is not None and asset_name in optimal_hyperparams and "Multilayer Perceptron" in optimal_hyperparams[asset_name]:
        params = optimal_hyperparams[asset_name]["Multilayer Perceptron"]
        layers = params["layers"]
        dropout = params["dropout"]
        lr = params["learning_rate"]
        print(f"[INFO] Building configured MLP model for {asset_name}: {params}")
        model = Sequential([Input(shape=(tam_ventana, 1)), Flatten()])
        for units in layers:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
    else:
        print("[INFO] Building default MLP model")
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        lr = LEARNING_RATE
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=(CONTROLLED_ARCHITECTURE_CONFIG["loss"] if ARCHITECTURE_PROTOCOL == "controlled" else "mae"),
    )
    return model

def construir_modelo_lstm(tam_ventana, asset_name):
    if ARCHITECTURE_PROTOCOL == "controlled":
        units = CONTROLLED_ARCHITECTURE_CONFIG["model_hidden_units"]["LSTM"]
        dropout = CONTROLLED_ARCHITECTURE_CONFIG["dropout"]
        lr = CONTROLLED_ARCHITECTURE_CONFIG["learning_rate"]
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            LSTM(units, return_sequences=True, dropout=dropout),
            LSTM(units, return_sequences=False, dropout=dropout),
            Dense(1, activation="linear"),
        ])
    elif asset_name is not None and asset_name in optimal_hyperparams and "LSTM" in optimal_hyperparams[asset_name]:
        params = optimal_hyperparams[asset_name]["LSTM"]
        print(f"[INFO] Building configured LSTM model for {asset_name}: {params}")
        model = Sequential()
        model.add(Input(shape=(tam_ventana, 1)))
        model.add(LSTM(params["units"][0], return_sequences=True, dropout=params["dropout"], recurrent_dropout=0.0))
        model.add(LSTM(params["units"][1], return_sequences=False, dropout=params["dropout"], recurrent_dropout=0.0))
        if params.get("use_extra_layers", False):
            model.add(BatchNormalization())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(1, activation='linear'))
        else:
            if params.get("extra_dense", False):
                model.add(Dense(32, activation='relu'))
                model.add(Dropout(params["dropout"]))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='linear'))
        lr = params["learning_rate"]
    else:
        params = {"units": [128, 64], "dropout": 0.3, "learning_rate": 0.001}
        print(f"[INFO] Building default LSTM model for {asset_name} with parameters: {params}")
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            LSTM(params["units"][0], return_sequences=True, dropout=params["dropout"], recurrent_dropout=0.0),
            LSTM(params["units"][1], dropout=params["dropout"], recurrent_dropout=0.0),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        lr = params["learning_rate"]
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=(CONTROLLED_ARCHITECTURE_CONFIG["loss"] if ARCHITECTURE_PROTOCOL == "controlled" else "mse"),
    )
    return model

def construir_modelo_gru(tam_ventana, asset_name):
    if ARCHITECTURE_PROTOCOL == "controlled":
        units = CONTROLLED_ARCHITECTURE_CONFIG["model_hidden_units"]["GRU"]
        dropout = CONTROLLED_ARCHITECTURE_CONFIG["dropout"]
        lr = CONTROLLED_ARCHITECTURE_CONFIG["learning_rate"]
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            GRU(units, return_sequences=True, dropout=dropout),
            GRU(units, return_sequences=False, dropout=dropout),
            Dense(1, activation="linear"),
        ])
    elif asset_name is not None and asset_name in optimal_hyperparams and "GRU" in optimal_hyperparams[asset_name]:
        params = optimal_hyperparams[asset_name]["GRU"]
        print(f"[INFO] Building configured GRU model for {asset_name}: {params}")
        model = Sequential([Input(shape=(tam_ventana, 1))])
        model.add(GRU(params["units"][0], return_sequences=True, dropout=params["dropout"], recurrent_dropout=0.0))
        model.add(GRU(params["units"][1], return_sequences=False, dropout=params["dropout"], recurrent_dropout=0.0))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        lr = params["learning_rate"]
    else:
        params = {"units": [128, 64], "dropout": 0.1, "learning_rate": 0.001}
        print(f"[INFO] Building default GRU model for {asset_name} with parameters: {params}")
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            GRU(params["units"][0], return_sequences=True, dropout=params["dropout"], recurrent_dropout=0.0),
            GRU(params["units"][1], dropout=params["dropout"], recurrent_dropout=0.0),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        lr = params["learning_rate"]
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=(CONTROLLED_ARCHITECTURE_CONFIG["loss"] if ARCHITECTURE_PROTOCOL == "controlled" else "mse"),
    )
    return model

def construir_modelo_transformer(tam_ventana, asset_name=None):
    if ARCHITECTURE_PROTOCOL == "controlled":
        d_model = CONTROLLED_ARCHITECTURE_CONFIG["model_hidden_units"]["TRANSFORMER"]
        num_heads = CONTROLLED_ARCHITECTURE_CONFIG["attention_heads"]
        dropout_rate = CONTROLLED_ARCHITECTURE_CONFIG["dropout"]
        lr = CONTROLLED_ARCHITECTURE_CONFIG["learning_rate"]
    elif asset_name is not None and asset_name in optimal_hyperparams and "TRANSFORMER" in optimal_hyperparams[asset_name]:
        params = optimal_hyperparams[asset_name]["TRANSFORMER"]
        print(f"[INFO] Building configured TRANSFORMER model for {asset_name}: {params}")
        d_model = params["d_model"]
        num_heads = params["num_heads"]
        dropout_rate = params["dropout"]
        lr = params["learning_rate"]
    else:
        d_model = 50
        num_heads = 2
        dropout_rate = 0.2
        lr = LEARNING_RATE
        print(f"[INFO] Building default TRANSFORMER model for {asset_name}")
    inputs = Input(shape=(tam_ventana, 1))
    x = Dense(d_model, activation='relu')(inputs)
    n_blocks = (
        CONTROLLED_ARCHITECTURE_CONFIG["transformer_blocks"]
        if ARCHITECTURE_PROTOCOL == "controlled" else 1
    )
    key_dim = max(1, d_model // num_heads)
    for _ in range(n_blocks):
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )(x, x, use_causal_mask=True)
        x = LayerNormalization()(Add()([x, Dropout(dropout_rate)(attn_output)]))
        feed_forward = Dense(d_model, activation="relu")(x)
        feed_forward = Dropout(dropout_rate)(feed_forward)
        x = LayerNormalization()(Add()([x, feed_forward]))
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=(CONTROLLED_ARCHITECTURE_CONFIG["loss"] if ARCHITECTURE_PROTOCOL == "controlled" else "mae"),
    )
    return model

def construir_modelo_bidi(tam_ventana, asset_name=None):
    # Bidirectionality is confined to the supplied historical lag window
    # [s_(t-tam_ventana), ..., s_(t-1)]; s_t and later observations are absent.
    if ARCHITECTURE_PROTOCOL == "controlled":
        units = CONTROLLED_ARCHITECTURE_CONFIG["model_hidden_units"]["BiLSTM"]
        dropout = CONTROLLED_ARCHITECTURE_CONFIG["dropout"]
        lr = CONTROLLED_ARCHITECTURE_CONFIG["learning_rate"]
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            tf.keras.layers.Bidirectional(
                LSTM(units, return_sequences=True, dropout=dropout),
                merge_mode="ave",
            ),
            tf.keras.layers.Bidirectional(
                LSTM(units, return_sequences=False, dropout=dropout),
                merge_mode="ave",
            ),
            Dense(1, activation="linear"),
        ])
    elif asset_name is not None and asset_name in optimal_hyperparams and "BiLSTM" in optimal_hyperparams[asset_name]:
        params = optimal_hyperparams[asset_name]["BiLSTM"]
        print(f"[INFO] Building configured BiLSTM model for {asset_name}: {params}")
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            tf.keras.layers.Bidirectional(LSTM(params["units"][0], return_sequences=True, dropout=params["dropout"], recurrent_dropout=0.0)),
            tf.keras.layers.Bidirectional(LSTM(params["units"][1], dropout=params["dropout"], recurrent_dropout=0.0)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        lr = params["learning_rate"]
    else:
        print(f"[INFO] Building default BiLSTM model for {asset_name}")
        model = Sequential([
            Input(shape=(tam_ventana, 1)),
            tf.keras.layers.Bidirectional(LSTM(50)),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        lr = 0.001
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=(CONTROLLED_ARCHITECTURE_CONFIG["loss"] if ARCHITECTURE_PROTOCOL == "controlled" else "mae"),
    )
    return model

modelos = {
    "Multilayer Perceptron": construir_modelo_perceptron,
    "LSTM": construir_modelo_lstm,
    "GRU": construir_modelo_gru,
    "TRANSFORMER": construir_modelo_transformer,
    "BiLSTM": construir_modelo_bidi,
    "Naive Persistence": None,
}
BASELINE_MODELS = {"Naive Persistence"}
NEURAL_MODELS = set(modelos).difference(BASELINE_MODELS)
AVAILABLE_MODELS = list(modelos.keys())
METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "r2": "R²",
    "nse": "NSE",
    "kge": "KGE",
    "time": "Fitting time (s)",
    "emissions": "Emissions (kgCO2eq)"
}


def validate_controlled_parameter_budget(model_name, trainable_parameters):
    """Ensure the controlled architectures remain within the declared budget."""
    if ARCHITECTURE_PROTOCOL != "controlled" or model_name not in NEURAL_MODELS:
        return
    target = CONTROLLED_ARCHITECTURE_CONFIG["target_trainable_parameters"]
    tolerance = CONTROLLED_ARCHITECTURE_CONFIG["parameter_tolerance_fraction"]
    relative_difference = abs(trainable_parameters - target) / target
    if relative_difference > tolerance:
        raise RuntimeError(
            f"Controlled model {model_name} has {trainable_parameters} trainable "
            f"parameters, outside the declared {tolerance:.0%} tolerance around "
            f"the target budget of {target}. Update the controlled widths before "
            "using this run in the paper."
        )

def calcular_nse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{y_true.size} != {y_pred.size}"
        )

    if y_true.size < 2:
        return np.nan

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("NSE cannot be computed with NaN or infinite values.")

    residual_sum_squares = np.sum(
        (y_true - y_pred) ** 2,
        dtype=np.float64
    )
    total_sum_squares = np.sum(
        (y_true - np.mean(y_true)) ** 2,
        dtype=np.float64
    )

    if total_sum_squares == 0.0:
        return np.nan

    return float(1.0 - residual_sum_squares / total_sum_squares)

def calcular_kge(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return np.nan

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    if std_true == 0 or mean_true == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        return np.nan

    alpha = std_pred / std_true
    beta = np.mean(y_pred) / mean_true
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def safe_command(command, timeout=120):
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_versions():
    versions = {}
    for distribution in SOFTWARE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def build_dataset_summary(series_to_analyze):
    """Record the exact observed dates/counts used after cleaning."""
    summary = {}
    for asset_name in series_to_analyze:
        asset_dates = dates_dict[asset_name]
        hash_payload = np.column_stack([
            pd.DatetimeIndex(asset_dates).asi8,
            np.asarray(series_dict[asset_name], dtype=np.float64).view("int64"),
        ]).tobytes()
        summary[asset_name] = {
            "data_source": "Yahoo Finance accessed through yfinance",
            "ticker": activos[asset_name]["ticker"],
            "field": "unadjusted daily Close",
            "interval": "1d",
            "auto_adjust": False,
            "requested_start": activos[asset_name]["start"],
            "requested_end_exclusive": activos[asset_name]["end"],
            "observed_start": (
                pd.Timestamp(asset_dates[0]).strftime("%Y-%m-%d")
                if len(asset_dates) else None
            ),
            "observed_end": (
                pd.Timestamp(asset_dates[-1]).strftime("%Y-%m-%d")
                if len(asset_dates) else None
            ),
            "observation_count": int(len(asset_dates)),
            "cleaned_series_sha256": hashlib.sha256(hash_payload).hexdigest(),
            "cleaning": (
                "coerce Close to numeric; remove NaN/infinite values; remove duplicate "
                "timestamps keeping the last; sort chronologically; reject non-positive prices"
            ),
        }
    return summary


def build_experiment_identity(
    args,
    series_to_analyze,
    models_to_analyze,
    methods_to_analyze,
    run_seeds,
    carbon_config,
    dataset_summary,
):
    """Create the immutable identity used to accept or reject checkpoints."""
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "deliverable_id": DELIVERABLE_ID,
        "source_sha256": file_sha256(os.path.abspath(__file__)),
        "software_versions": distribution_versions(),
        "assets": list(series_to_analyze),
        "models": list(models_to_analyze),
        "methods": list(methods_to_analyze),
        "seeds": [int(value) for value in run_seeds],
        "dataset_hashes": {
            asset: dataset_summary[asset]["cleaned_series_sha256"]
            for asset in series_to_analyze
        },
        "window_size": WINDOW_SIZE,
        "test_protocol": "method-specific fold tests pooled out of sample",
        "partition_configurations": {
            method: PARTITION_CONFIGURATIONS[method]
            for method in methods_to_analyze
        },
        "change_point_configuration": CHANGE_POINT_CONFIG,
        "architecture_protocol": ARCHITECTURE_PROTOCOL,
        "controlled_architecture_config": (
            CONTROLLED_ARCHITECTURE_CONFIG
            if ARCHITECTURE_PROTOCOL == "controlled" else None
        ),
        "individually_tuned_hyperparameters": (
            {
                asset: {
                    model: optimal_hyperparams[asset][model]
                    for model in models_to_analyze
                    if model in NEURAL_MODELS
                }
                for asset in series_to_analyze
            }
            if ARCHITECTURE_PROTOCOL == "individually_tuned" else None
        ),
        "hyperparameter_validation_file": (
            os.path.abspath(args.validated_hyperparameters)
            if getattr(args, "validated_hyperparameters", None) else None
        ),
        "hyperparameter_validation_file_sha256": (
            file_sha256(args.validated_hyperparameters)
            if getattr(args, "validated_hyperparameters", None) else None
        ),
        "epochs_override": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "emissions_tracking": bool(not args.no_emissions),
        "carbon_configuration": carbon_config,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return identity, hashlib.sha256(canonical).hexdigest()


def initialize_run_manifest(checkpoint_dir, identity, fingerprint):
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, "run_manifest.json")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    resume_count = 0
    first_started_at = now
    previous_cumulative_wall_time = 0.0
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            previous = json.load(manifest_file)
        if previous.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                "Checkpoint manifest fingerprint mismatch; use a different "
                "checkpoint directory for this experiment."
            )
        resume_count = int(previous.get("resume_count", 0)) + 1
        first_started_at = previous.get("first_started_at_utc", now)
        previous_cumulative_wall_time = float(
            previous.get(
                "cumulative_wall_time_seconds",
                previous.get("invocation_wall_time_seconds", 0.0),
            )
        )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "run_fingerprint": fingerprint,
        "first_started_at_utc": first_started_at,
        "current_invocation_started_at_utc": now,
        "resume_count": resume_count,
        "previous_cumulative_wall_time_seconds": previous_cumulative_wall_time,
        "identity": identity,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest


def finalize_run_manifest(manifest_path, manifest, **updates):
    completed = dict(manifest)
    completed.update(updates)
    completed["updated_at_utc"] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    atomic_write_json(manifest_path, completed)
    return completed


def preflight_runtime(require_gpu, carbon_config):
    """Fail before a long run if GPU execution or CodeCarbon is unavailable."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "No TensorFlow GPU is visible. The paper run was not started. "
            "Use --allow-cpu only for an explicitly labelled CPU run."
        )
    device_name = None
    if gpu_devices:
        print(
            "[INFO] Running a real TensorFlow operation on GPU:0. On a new "
            "RTX architecture, the first CUDA PTX compilation may take time."
        )
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(
                tf.ones((16, 16), dtype=tf.float32),
                tf.ones((16, 16), dtype=tf.float32),
            )
        device_name = probe.device
        if require_gpu and "GPU" not in str(device_name).upper():
            raise RuntimeError(
                f"TensorFlow probe was not placed on the GPU: {device_name}"
            )

    carbon_preflight = {"enabled": bool(carbon_config.get("enabled"))}
    if carbon_config.get("enabled"):
        output_dir = os.path.abspath(carbon_config["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        descriptor, probe_path = tempfile.mkstemp(
            prefix=".codecarbon_write_probe_", dir=output_dir
        )
        os.close(descriptor)
        os.unlink(probe_path)
        tracker_arguments = {
            "project_name": "crypto_forecasting_preflight",
            "country_iso_code": carbon_config["country_iso_code"],
            "measure_power_secs": carbon_config["measure_power_secs"],
            "tracking_mode": carbon_config["tracking_mode"],
            "save_to_file": False,
            "log_level": "error",
        }
        if carbon_config.get("region"):
            tracker_arguments["region"] = carbon_config["region"]
        tracker = OfflineEmissionsTracker(**tracker_arguments)
        tracker.start()
        try:
            # Force a small real operation while the tracker is active. Its
            # measurement is discarded and excluded from the declared boundary.
            _ = tf.reduce_sum(tf.ones((256, 256))).numpy()
        finally:
            preflight_emissions = tracker.stop()
        if preflight_emissions is None:
            raise RuntimeError("CodeCarbon preflight returned no emissions value.")
        carbon_preflight.update({
            "successful": True,
            "discarded_probe_emissions_kgco2eq": float(preflight_emissions),
            "included_in_reported_results": False,
        })
    result = {
        "gpu_required": bool(require_gpu),
        "visible_gpu_devices": [device.name for device in gpu_devices],
        "tensorflow_probe_device": device_name,
        "codecarbon": carbon_preflight,
    }
    print(
        "[INFO] Runtime preflight passed. TensorFlow probe device: "
        f"{device_name or 'CPU'}"
    )
    return result


def write_environment_artifacts(
    args,
    series_to_analyze,
    models_to_analyze,
    methods_to_analyze,
    run_seeds,
    carbon_config,
    dataset_summary,
    run_fingerprint=None,
    preflight=None,
):
    pip_freeze = safe_command([sys.executable, "-m", "pip", "freeze"])
    nvidia_smi = safe_command([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader"
    ])
    codecarbon_detect = safe_command(["codecarbon", "detect"])

    try:
        tensorflow_build = tf.sysconfig.get_build_info()
    except Exception as error:
        tensorflow_build = {"error": f"{type(error).__name__}: {error}"}

    software_parent = os.path.dirname(os.path.abspath(args.software_output))
    os.makedirs(software_parent, exist_ok=True)
    software_text = pip_freeze["stdout"]
    if not software_text:
        software_text = "pip freeze failed: " + pip_freeze["stderr"]

    atomic_write_text(
        args.software_output,
        software_text.rstrip() + "\n",
    )

    metadata = {
        "deliverable_id": DELIVERABLE_ID,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_file": os.path.abspath(__file__),
        "source_sha256": file_sha256(os.path.abspath(__file__)),
        "command_line": [sys.executable, *sys.argv],
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "software_versions": distribution_versions(),
        "tensorflow_build": tensorflow_build,
        "tensorflow_visible_devices": {
            "cpu": [
                device.name
                for device in tf.config.list_physical_devices("CPU")
            ],
            "gpu": [
                device.name
                for device in tf.config.list_physical_devices("GPU")
            ],
        },
        "nvidia_smi": nvidia_smi,
        "codecarbon_detect": codecarbon_detect,
        "pip_freeze_file": os.path.abspath(args.software_output),
        "experiment": {
            "seeds": [int(value) for value in run_seeds],
            "number_of_repetitions": len(run_seeds),
            "window_size": WINDOW_SIZE,
            "epoch_selection": (
                "independent validation-monitored early stopping inside each fold; "
                "no final refit and no epoch transfer between folds"
            ),
            "test_evaluation_protocol": (
                "The red block in every partition fold is its genuine test. For "
                "multi-round methods, unique out-of-sample predictions are pooled "
                "chronologically and every metric is computed once on that union."
            ),
            "selected_assets": series_to_analyze,
            "selected_models": models_to_analyze,
            "selected_partition_methods": methods_to_analyze,
            "experimental_evaluation_start": args.evaluation_start,
            "validated_hyperparameters_file": (
                os.path.abspath(args.validated_hyperparameters)
                if args.validated_hyperparameters else None
            ),
            "partition_configurations": {
                method: PARTITION_CONFIGURATIONS[method]
                for method in methods_to_analyze
            },
            "architecture_protocol": ARCHITECTURE_PROTOCOL,
            "controlled_architecture_config": (
                CONTROLLED_ARCHITECTURE_CONFIG
                if ARCHITECTURE_PROTOCOL == "controlled" else None
            ),
            "hyperparameter_provenance": {
                f"{asset}::{model}": HYPERPARAMETER_PROVENANCE.get(
                    (asset, model), "not applicable"
                )
                for asset in series_to_analyze
                for model in models_to_analyze
                if model in NEURAL_MODELS
            },
            "hyperparameter_validation_records": {
                f"{asset}::{model}": _compact_validation_record(
                    HYPERPARAMETER_VALIDATION_RECORDS.get((asset, model))
                )
                for asset in series_to_analyze
                for model in models_to_analyze
                if model in NEURAL_MODELS
            },
            "change_point_configuration": CHANGE_POINT_CONFIG,
            "epochs_override": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "dataset_summary": dataset_summary,
        "run_fingerprint": run_fingerprint,
        "runtime_preflight": preflight,
        "codecarbon_configuration": carbon_config,
        "carbon_accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
    }

    environment_parent = os.path.dirname(
        os.path.abspath(args.environment_output)
    )
    os.makedirs(environment_parent, exist_ok=True)

    atomic_write_json(args.environment_output, metadata)

    print(
        f"[INFO] Software environment saved as "
        f"'{args.software_output}'"
    )
    print(
        f"[INFO] Environment metadata saved as "
        f"'{args.environment_output}'"
    )


def codecarbon_data_to_dict(tracker):
    emissions_data = getattr(tracker, "final_emissions_data", None)

    if emissions_data is None:
        return {}

    if dataclasses.is_dataclass(emissions_data):
        return dataclasses.asdict(emissions_data)

    if hasattr(emissions_data, "to_dict"):
        return emissions_data.to_dict()

    if hasattr(emissions_data, "__dict__"):
        return {
            key: value
            for key, value in vars(emissions_data).items()
            if not key.startswith("_")
        }

    return {}


def sanitize_run_component(value):
    cleaned = "".join(
        character if str(character).isalnum() else "_"
        for character in str(value)
    )
    return "_".join(part for part in cleaned.split("_") if part)


def create_emissions_tracker(carbon_config, run_context):
    if not carbon_config or not carbon_config.get("country_iso_code"):
        raise ValueError(
            "The real three-letter country code is required "
            "when CodeCarbon is enabled."
        )

    os.makedirs(carbon_config["output_dir"], exist_ok=True)

    context = run_context or {}
    project_name = "__".join([
        "crypto_forecasting",
        sanitize_run_component(
            context.get("asset", "unknown_asset")
        ),
        sanitize_run_component(
            context.get("model", "unknown_model")
        ),
        sanitize_run_component(
            context.get("method", "unknown_method")
        ),
        f"seed_{context.get('seed', seed)}",
        sanitize_run_component(
            context.get("stage", "fold_fit")
        ),
    ])

    configured_output = carbon_config["output_file"]
    output_stem, output_extension = os.path.splitext(configured_output)
    output_extension = output_extension or ".csv"
    attempt_token = f"pid_{os.getpid()}_{time.time_ns()}"
    stage_output_file = (
        f"{output_stem}__{sanitize_run_component(project_name)}"
        f"__{attempt_token}"
        f"{output_extension}"
    )

    tracker_arguments = {
        "project_name": project_name,
        "country_iso_code": carbon_config["country_iso_code"],
        "measure_power_secs": carbon_config["measure_power_secs"],
        "tracking_mode": carbon_config["tracking_mode"],
        "output_dir": carbon_config["output_dir"],
        # One file per fitting stage avoids ambiguous duplicate rows when a
        # process is interrupted and subsequently resumed.
        "output_file": stage_output_file,
        "save_to_file": True,
        "log_level": "info",
    }

    if carbon_config.get("region"):
        tracker_arguments["region"] = carbon_config["region"]

    return OfflineEmissionsTracker(**tracker_arguments)


def atomic_write_json(path, payload):
    """Durably publish one JSON file without exposing a partial result."""
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".json",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                to_jsonable(payload),
                output_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, absolute_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def atomic_write_text(path, text_value, encoding="utf-8"):
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_text_", suffix=".txt", dir=parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding=encoding, newline=""
        ) as output_file:
            output_file.write(text_value)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, absolute_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def atomic_save_npz(path, **arrays):
    """Durably publish compressed NumPy arrays for one completed result."""
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".npz",
        dir=parent,
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_path, **arrays)
        with open(temporary_path, "rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, absolute_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def combination_identifier(asset, model, method, run_seed):
    readable = "__".join([
        sanitize_run_component(asset),
        sanitize_run_component(model),
        sanitize_run_component(method),
        f"seed_{int(run_seed)}",
    ])
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{digest}"


def fold_checkpoint_paths(checkpoint_dir, combination_id, fold_label):
    base = os.path.join(
        checkpoint_dir, "folds", combination_id,
        sanitize_run_component(fold_label),
    )
    return f"{base}.json", f"{base}__predictions.npz"


def load_fold_checkpoint(
    checkpoint_dir,
    combination_id,
    fold_label,
    run_fingerprint,
):
    if not checkpoint_dir:
        return None
    metadata_path, predictions_path = fold_checkpoint_paths(
        checkpoint_dir, combination_id, fold_label
    )
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported fold checkpoint: {metadata_path}")
    if payload.get("status") != "completed":
        raise RuntimeError(f"Incomplete fold checkpoint: {metadata_path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(
            f"Fold checkpoint belongs to a different experiment: {metadata_path}"
        )
    if payload.get("combination_id") != combination_id:
        raise RuntimeError(f"Fold checkpoint key mismatch: {metadata_path}")
    if payload.get("fold_label") != fold_label:
        raise RuntimeError(f"Fold checkpoint label mismatch: {metadata_path}")
    if not os.path.exists(predictions_path):
        raise RuntimeError(f"Fold prediction checkpoint is missing: {predictions_path}")
    with np.load(predictions_path, allow_pickle=False) as arrays:
        result = payload["fold_result"]
        for key in (
            "predictions_original", "targets_original",
            "predictions_scaled", "targets_scaled",
        ):
            result[key] = arrays[key].astype(np.float64, copy=True)
        result["raw_target_indices"] = arrays["raw_target_indices"].astype(
            int, copy=True
        )
        result["test_dates"] = arrays["test_dates"].astype(
            "datetime64[D]", copy=True
        )
    return result


def save_fold_checkpoint(
    checkpoint_dir,
    combination_id,
    fold_label,
    run_fingerprint,
    fold_result,
):
    if not checkpoint_dir:
        return
    metadata_path, predictions_path = fold_checkpoint_paths(
        checkpoint_dir, combination_id, fold_label
    )
    array_keys = (
        "predictions_original", "targets_original",
        "predictions_scaled", "targets_scaled",
        "raw_target_indices", "test_dates",
    )
    atomic_save_npz(
        predictions_path,
        predictions_original=np.asarray(
            fold_result["predictions_original"], dtype=np.float64
        ),
        targets_original=np.asarray(
            fold_result["targets_original"], dtype=np.float64
        ),
        predictions_scaled=np.asarray(
            fold_result["predictions_scaled"], dtype=np.float64
        ),
        targets_scaled=np.asarray(
            fold_result["targets_scaled"], dtype=np.float64
        ),
        raw_target_indices=np.asarray(
            fold_result["raw_target_indices"], dtype=np.int64
        ),
        test_dates=np.asarray(fold_result["test_dates"], dtype="datetime64[D]"),
    )
    metadata_result = {
        key: value for key, value in fold_result.items() if key not in array_keys
    }
    atomic_write_json(metadata_path, {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "combination_id": combination_id,
        "fold_label": fold_label,
        "prediction_file": os.path.relpath(
            predictions_path, os.path.dirname(metadata_path)
        ),
        "fold_result": metadata_result,
    })


def entrenar_evaluar_modelo(
    funcion_modelo,
    tam_ventana,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs,
    model_label=None,
    batch_size=None,
    early_stopping_patience=None,
    track_emissions=True,
    carbon_config=None,
    run_context=None
):
    model_label = model_label or getattr(
        funcion_modelo,
        "__name__",
        "model"
    )
    print(
        f"[INFO] Training model {model_label} "
        f"for {epochs} epochs"
    )

    model = funcion_modelo(tam_ventana)
    trainable_parameters = int(model.count_params())
    validate_controlled_parameter_budget(model_label, trainable_parameters)

    callbacks = []
    if (
        early_stopping_patience is not None
        and early_stopping_patience > 0
    ):
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True
        ))

    tracker = None
    emisiones = None
    accounting_boundary = (run_context or {}).get(
        "accounting_boundary", CARBON_ACCOUNTING_BOUNDARY
    )
    carbon_metadata = {
        "tracking_enabled": bool(track_emissions),
        "accounting_boundary": accounting_boundary,
        "declared_country_iso_code": (
            carbon_config or {}
        ).get("country_iso_code"),
        "declared_region": (
            carbon_config or {}
        ).get("region"),
        "run_context": run_context or {},
    }

    if track_emissions:
        tracker = create_emissions_tracker(
            carbon_config,
            run_context
        )
        tracker.start()

    inicio = time.perf_counter()

    try:
        history = model.fit(
            X_train.reshape(-1, tam_ventana, 1),
            y_train,
            validation_data=(
                X_val.reshape(-1, tam_ventana, 1),
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=0
        )
    finally:
        tiempo_ejecucion = time.perf_counter() - inicio

        if tracker is not None:
            emisiones = tracker.stop()

    if tracker is not None:
        detailed_metadata = codecarbon_data_to_dict(tracker)

        if not detailed_metadata:
            raise RuntimeError(
                "CodeCarbon did not expose final_emissions_data; "
                "do not use this run for the paper."
            )

        carbon_metadata.update(detailed_metadata)

        if emisiones is None:
            raise RuntimeError(
                "CodeCarbon did not return an emissions value; "
                "do not use this run for the paper."
            )

        detailed_emissions = carbon_metadata.get("emissions")

        if (
            detailed_emissions is not None
            and not np.isclose(
                float(emisiones),
                float(detailed_emissions),
                rtol=1e-9,
                atol=1e-15
            )
        ):
            raise RuntimeError(
                "CodeCarbon scalar and detailed emissions "
                "values are inconsistent."
            )

        if not carbon_metadata.get("run_id"):
            raise RuntimeError(
                "CodeCarbon did not record a run_id; "
                "do not use this run for the paper."
            )

    emisiones_str = (
        f"{float(emisiones):.7f}"
        if emisiones is not None
        else "N/A"
    )

    print(
        f"[INFO] Training completed in "
        f"{tiempo_ejecucion:.2f} seconds"
    )
    print(
        f"[INFO] Estimated emissions: "
        f"{emisiones_str} kgCO2eq"
    )

    preds = model.predict(
        X_test.reshape(-1, tam_ventana, 1),
        verbose=0
    ).flatten()

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    epochs_trained = len(history.history.get("loss", []))
    validation_history = history.history.get("val_loss", [])
    best_epoch = (
        int(np.argmin(validation_history) + 1)
        if validation_history else None
    )
    return (
        mse,
        rmse,
        mae,
        mape,
        tiempo_ejecucion,
        emisiones,
        preds,
        y_test,
        carbon_metadata,
        epochs_trained,
        best_epoch,
        trainable_parameters,
    )


def _date_range_for_positions(dates_arr, raw_target_indices, positions):
    positions = np.asarray(positions, dtype=int)
    if positions.size == 0:
        return "not applicable"
    selected_dates = dates_arr[raw_target_indices[positions]]
    return (
        f"{np.datetime_as_string(selected_dates.min(), unit='D')} - "
        f"{np.datetime_as_string(selected_dates.max(), unit='D')}"
    )


def _canonical_efficiency_scores(targets, predictions, context):
    nse = calcular_nse(targets, predictions)
    if np.isnan(nse):
        return np.nan, np.nan
    r2 = float(r2_score(targets, predictions))
    if not np.isclose(r2, nse, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise RuntimeError(
            f"Unexpected discrepancy between R2 and NSE for {context}: "
            f"R2={r2}, NSE={nse}"
        )
    return float(nse), float(nse)


def _pooled_metrics(
    targets_original,
    predictions_original,
    targets_scaled,
    predictions_scaled,
    context="method-specific out-of-sample predictions",
):
    mse = float(mean_squared_error(targets_original, predictions_original))
    mse_scaled = float(mean_squared_error(targets_scaled, predictions_scaled))
    r2, nse = _canonical_efficiency_scores(
        targets_original,
        predictions_original,
        context,
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(targets_original, predictions_original)),
        "mape": float(
            100.0 * mean_absolute_percentage_error(
                targets_original,
                predictions_original,
            )
        ),
        "mse_scaled": mse_scaled,
        "rmse_scaled": float(np.sqrt(mse_scaled)),
        "mae_scaled": float(
            mean_absolute_error(targets_scaled, predictions_scaled)
        ),
        "r2": r2,
        "nse": nse,
        "kge": calcular_kge(targets_original, predictions_original),
    }


def _partition_result_metadata(plan):
    fold_summaries = []
    for fold in plan["folds"]:
        fold_summaries.append({
            "label": fold["label"],
            "n_train": int(len(fold["train_pos"])),
            "n_validation": int(len(fold["val_pos"])),
            "n_test": int(len(fold["test_pos"])),
            "n_purged": int(len(fold.get("purge_pos", []))),
            "n_embargoed": int(len(fold.get("embargo_pos", []))),
        })
    return {
        "protocol": (
            "each fold is fitted on its training block, monitored on its "
            "validation block, and evaluated on its red test block"
        ),
        "test_fraction_evaluated": float(plan["test_size"] / plan["n_supervised"]),
        "test_observations": int(plan["test_size"]),
        "out_of_sample_aggregation": (
            "concatenate fold predictions, sort by target date, and compute "
            "every metric once on the pooled out-of-sample observations"
        ),
        "folds": fold_summaries,
        "method_metadata": plan["metadata"],
    }


def _pooled_fold_arrays(fold_results, context):
    """Concatenate, validate, and chronologically order unique OOF predictions."""
    if not fold_results:
        raise ValueError(f"{context}: no fold results are available.")
    keys = (
        "predictions_original", "targets_original",
        "predictions_scaled", "targets_scaled",
        "raw_target_indices", "test_dates",
    )
    arrays = {
        key: np.concatenate([np.asarray(fold[key]) for fold in fold_results])
        for key in keys
    }
    raw_indices = arrays["raw_target_indices"].astype(int, copy=False)
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise RuntimeError(
            f"{context}: an out-of-sample target was predicted more than once."
        )
    order = np.argsort(raw_indices, kind="stable")
    return {key: np.asarray(value)[order] for key, value in arrays.items()}


def _fold_metadata_without_arrays(fold_result):
    array_keys = {
        "predictions_original", "targets_original",
        "predictions_scaled", "targets_scaled",
        "raw_target_indices", "test_dates",
    }
    return {
        key: value for key, value in fold_result.items()
        if key not in array_keys
    }


def _compact_validation_record(record):
    if not record:
        return None
    keys = (
        "asset", "model", "status", "initial_provenance",
        "selected_candidate_id", "objective", "objective_value",
        "validation_rounds", "validation_seeds",
        "calibration_observed_start", "calibration_observed_end",
        "calibration_observation_count", "calibration_series_sha256",
        "validated_at_utc",
    )
    return {key: record.get(key) for key in keys if key in record}


def _architecture_metadata(asset_name, model_name):
    if ARCHITECTURE_PROTOCOL == "controlled":
        return {
            "config": dict(CONTROLLED_ARCHITECTURE_CONFIG),
            "provenance": "controlled_sensitivity_protocol",
            "validated": True,
            "validation_record": {
                "status": "not_applicable_to_individually_tuned_protocol"
            },
        }
    pair = (asset_name, model_name)
    if pair not in HYPERPARAMETER_PROVENANCE:
        raise RuntimeError(
            f"Missing hyperparameter configuration for {asset_name}/{model_name}."
        )
    validation_record = HYPERPARAMETER_VALIDATION_RECORDS.get(pair)
    return {
        "config": json.loads(json.dumps(
            optimal_hyperparams[asset_name][model_name]
        )),
        "provenance": HYPERPARAMETER_PROVENANCE[pair],
        "validated": bool(
            validation_record
            and validation_record.get("status") == "validated"
        ),
        "validation_record": _compact_validation_record(validation_record),
    }


def evaluate_model_asset(
    model_name,
    funcion_modelo,
    asset_name,
    serie,
    metodo,
    epochs_default,
    partition_plan=None,
    epochs_override=None,
    batch_size=None,
    early_stopping_patience=None,
    track_emissions=True,
    carbon_config=None,
    run_seed=123,
    checkpoint_dir=None,
    run_fingerprint=None,
):
    """Fit every temporal fold and score pooled out-of-sample predictions once."""
    if model_name in BASELINE_MODELS:
        raise ValueError(
            "Naive Persistence must be evaluated with evaluate_naive_asset()."
        )
    print(
        f"[INFO] Evaluating model '{model_name}' on asset '{asset_name}' "
        f"using {METODOS_MAP[metodo]}"
    )
    if partition_plan is None:
        partition_plan = obtener_particion(serie, WINDOW_SIZE, metodo)
    validate_causal_partition(serie, partition_plan, metodo)
    X_raw, y_raw, raw_target_indices = crear_instancias(serie, WINDOW_SIZE)
    dates_arr = dates_dict[asset_name].to_numpy()

    if ARCHITECTURE_PROTOCOL == "controlled":
        maximum_epochs = int(CONTROLLED_ARCHITECTURE_CONFIG["max_epochs"])
    else:
        maximum_epochs = int(
            optimal_hyperparams.get(asset_name, {})
            .get(model_name, {})
            .get("epochs", epochs_default)
        )
    if epochs_override is not None:
        maximum_epochs = int(epochs_override)
    if maximum_epochs <= 0:
        raise ValueError("The maximum epoch count must be greater than zero.")

    model_factory = lambda size: funcion_modelo(size, asset_name)
    combination_id = combination_identifier(
        asset_name, model_name, metodo, run_seed
    )
    fold_results = []
    for fold_index, fold in enumerate(partition_plan["folds"], start=1):
        fold_label = str(fold["label"])
        saved_fold = load_fold_checkpoint(
            checkpoint_dir, combination_id, fold_label, run_fingerprint
        )
        if saved_fold is not None:
            print(f"[INFO] Resuming completed fold '{fold_label}'")
            fold_results.append(saved_fold)
            continue

        train_pos = np.asarray(fold["train_pos"], dtype=int)
        val_pos = np.asarray(fold["val_pos"], dtype=int)
        test_pos = np.asarray(fold["test_pos"], dtype=int)
        scaled_partition, fold_scaler = scale_partition_from_training((
            X_raw[train_pos], y_raw[train_pos],
            X_raw[val_pos], y_raw[val_pos],
            X_raw[test_pos], y_raw[test_pos],
            raw_target_indices[train_pos],
            raw_target_indices[val_pos],
            raw_target_indices[test_pos],
        ))
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            _idx_train, _idx_val, _idx_test,
        ) = scaled_partition
        fold_seed = int((int(run_seed) + 1009 * fold_index) % (2 ** 31 - 1))
        set_reproducible_seed(fold_seed)
        (
            _mse_scaled, _rmse_scaled, _mae_scaled, mape_scaled,
            fold_time, fold_emissions, predictions_scaled, targets_scaled,
            fold_carbon_metadata, epochs_trained, best_epoch,
            trainable_parameters,
        ) = entrenar_evaluar_modelo(
            model_factory, WINDOW_SIZE,
            X_train, y_train, X_val, y_val, X_test, y_test,
            maximum_epochs, model_name,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            track_emissions=track_emissions,
            carbon_config=carbon_config,
            run_context={
                "asset": asset_name,
                "model": model_name,
                "method": metodo,
                "seed": int(run_seed),
                "fold_seed": fold_seed,
                "stage": f"fit_{fold_label}",
            },
        )
        if best_epoch is None:
            raise RuntimeError(
                f"No validation-loss history was recorded for fold {fold_label}."
            )
        predictions_scaled = np.asarray(
            predictions_scaled, dtype=np.float64
        ).reshape(-1)
        targets_scaled = np.asarray(targets_scaled, dtype=np.float64).reshape(-1)
        predictions_original = fold_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).reshape(-1)
        targets_original = y_raw[test_pos].astype(np.float64, copy=True)
        inverse_targets = fold_scaler.inverse_transform(
            targets_scaled.reshape(-1, 1)
        ).reshape(-1)
        if not np.allclose(
            inverse_targets, targets_original, rtol=1e-12, atol=1e-9
        ):
            raise RuntimeError(
                f"{asset_name}/{metodo}/{fold_label}: inverse-scaled targets "
                "do not match original prices."
            )
        test_raw_indices = raw_target_indices[test_pos]
        fold_result = {
            "fold_index": int(fold_index),
            "label": fold_label,
            "fold_seed": fold_seed,
            "best_epoch": int(best_epoch),
            "epochs_trained": int(epochs_trained),
            "elapsed_seconds": float(fold_time),
            "emissions": (
                None if fold_emissions is None else float(fold_emissions)
            ),
            "trainable_parameters": int(trainable_parameters),
            "carbon_run_id": fold_carbon_metadata.get("run_id"),
            "carbon_metadata": fold_carbon_metadata,
            "n_train": int(len(train_pos)),
            "n_validation": int(len(val_pos)),
            "n_test": int(len(test_pos)),
            "n_purged": int(len(fold.get("purge_pos", []))),
            "n_embargoed": int(len(fold.get("embargo_pos", []))),
            "train_range": _date_range_for_positions(
                dates_arr, raw_target_indices, train_pos
            ),
            "validation_range": _date_range_for_positions(
                dates_arr, raw_target_indices, val_pos
            ),
            "test_range": _date_range_for_positions(
                dates_arr, raw_target_indices, test_pos
            ),
            "scaler_train_min_usd": float(fold_scaler.data_min_[0]),
            "scaler_train_max_usd": float(fold_scaler.data_max_[0]),
            "metrics": _pooled_metrics(
                targets_original, predictions_original,
                targets_scaled, predictions_scaled,
                context=f"{asset_name}/{model_name}/{metodo}/{fold_label}",
            ),
            "mape_scaled_fraction": float(mape_scaled),
            "predictions_original": predictions_original,
            "targets_original": targets_original,
            "predictions_scaled": predictions_scaled,
            "targets_scaled": targets_scaled,
            "raw_target_indices": test_raw_indices,
            "test_dates": dates_arr[test_raw_indices],
        }
        save_fold_checkpoint(
            checkpoint_dir, combination_id, fold_label,
            run_fingerprint, fold_result
        )
        fold_results.append(fold_result)

    pooled = _pooled_fold_arrays(
        fold_results, f"{asset_name}/{model_name}/{metodo}"
    )
    metrics = _pooled_metrics(
        pooled["targets_original"], pooled["predictions_original"],
        pooled["targets_scaled"], pooled["predictions_scaled"],
        context=f"{asset_name}/{model_name}/{metodo}/pooled_oof",
    )
    total_time = float(sum(fold["elapsed_seconds"] for fold in fold_results))
    stage_emissions = [fold["emissions"] for fold in fold_results]
    if track_emissions:
        if any(value is None for value in stage_emissions):
            raise RuntimeError("A tracked fold fit has no emissions value.")
        total_emissions = float(sum(stage_emissions))
    else:
        total_emissions = None
    parameter_counts = {
        int(fold["trainable_parameters"]) for fold in fold_results
    }
    if len(parameter_counts) != 1:
        raise RuntimeError("Trainable parameter count changed between folds.")

    train_positions = np.unique(np.concatenate([
        np.asarray(fold["train_pos"], dtype=int)
        for fold in partition_plan["folds"]
    ]))
    validation_positions = np.unique(np.concatenate([
        np.asarray(fold["val_pos"], dtype=int)
        for fold in partition_plan["folds"]
    ]))
    test_positions = np.asarray(partition_plan["all_test_pos"], dtype=int)
    if not np.array_equal(
        pooled["raw_target_indices"].astype(int),
        np.sort(raw_target_indices[test_positions]),
    ):
        raise RuntimeError(
            f"{asset_name}/{metodo}: pooled predictions do not match "
            "the partition's declared test targets."
        )

    stage_metadata = [fold["carbon_metadata"] for fold in fold_results]
    carbon_run_ids = [
        metadata.get("run_id") for metadata in stage_metadata
        if metadata.get("run_id")
    ]
    architecture = _architecture_metadata(asset_name, model_name)
    training_loss = (
        CONTROLLED_ARCHITECTURE_CONFIG["loss"]
        if ARCHITECTURE_PROTOCOL == "controlled"
        else ("mse" if model_name in {"LSTM", "GRU"} else "mae")
    )
    partition_configuration = {
        **PARTITION_CONFIGURATIONS[metodo],
        **_partition_result_metadata(partition_plan),
    }
    if metodo == "change_point_partitioning":
        change_metadata = partition_plan["metadata"]

        def raw_index_date(raw_index):
            if 0 <= int(raw_index) < len(dates_arr):
                return np.datetime_as_string(dates_arr[int(raw_index)], unit="D")
            return None

        partition_configuration["pelt_detected_change_point_dates"] = [
            raw_index_date(value)
            for value in change_metadata["detected_change_points_raw_indices"]
        ]
        partition_configuration["pelt_selected_change_point_date"] = raw_index_date(
            change_metadata["selected_change_point_raw_index"]
        )

    fold_summaries = [
        _fold_metadata_without_arrays(fold) for fold in fold_results
    ]
    result = {
        "model": model_name,
        "asset": asset_name,
        "method": metodo,
        "partition_configuration": partition_configuration,
        "seed": int(run_seed),
        **metrics,
        "time": total_time,
        "inference_time_seconds": None,
        "emissions": total_emissions,
        "architecture_protocol": ARCHITECTURE_PROTOCOL,
        "architecture_config": architecture["config"],
        "hyperparameter_provenance": architecture["provenance"],
        "hyperparameters_validated": architecture["validated"],
        "hyperparameter_validation_record": architecture["validation_record"],
        "optimizer": "Adam",
        "learning_rate": architecture["config"].get("learning_rate"),
        "training_loss": training_loss,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "epochs_trained": int(sum(
            fold["epochs_trained"] for fold in fold_results
        )),
        "fold_best_epochs": [
            int(fold["best_epoch"]) for fold in fold_results
        ],
        "epoch_selection_rule": (
            "early stopping is applied independently inside each fold; "
            "there is no final refit or cross-fold epoch transfer"
        ),
        "fold_count": int(len(fold_results)),
        "fold_results": fold_summaries,
        "trainable_parameters": int(next(iter(parameter_counts))),
        "carbon_run_id": carbon_run_ids[0] if len(carbon_run_ids) == 1 else None,
        "carbon_run_ids": carbon_run_ids,
        "carbon_accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
        "carbon_metadata": {
            "tracking_enabled": bool(track_emissions),
            "accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
            "aggregation_rule": "sum of all method-specific fold-fit emissions",
            "total_emissions_kgco2eq": total_emissions,
            "stages": stage_metadata,
        },
        "preds_list": [(
            pooled["predictions_original"].astype(np.float64),
            pooled["targets_original"].astype(np.float64),
        )],
        "n_train": int(len(train_positions)),
        "n_val": int(len(validation_positions)),
        "n_train_observations_across_folds": int(sum(
            len(fold["train_pos"]) for fold in partition_plan["folds"]
        )),
        "n_validation_observations_across_folds": int(sum(
            len(fold["val_pos"]) for fold in partition_plan["folds"]
        )),
        "n_test": int(len(test_positions)),
        "fold_test_counts": [
            int(len(fold["test_pos"])) for fold in partition_plan["folds"]
        ],
        "fold_test_ranges": [fold["test_range"] for fold in fold_summaries],
        "train_range": _date_range_for_positions(
            dates_arr, raw_target_indices, train_positions
        ),
        "valid_range": _date_range_for_positions(
            dates_arr, raw_target_indices, validation_positions
        ),
        "test_range": _date_range_for_positions(
            dates_arr, raw_target_indices, test_positions
        ),
        "total_samples": int(len(X_raw)),
        "n_original": int(len(serie)),
        "metric_units": (
            "MSE: USD^2; RMSE/MAE: USD; MAPE: percent; "
            "R2/NSE/KGE: dimensionless"
        ),
        "scaler": (
            "one MinMaxScaler(0,1) per fold, fitted only on that fold's "
            "training observations"
        ),
        "scaler_train_min_usd": [
            float(fold["scaler_train_min_usd"]) for fold in fold_results
        ],
        "scaler_train_max_usd": [
            float(fold["scaler_train_max_usd"]) for fold in fold_results
        ],
        "input_tensor_definition": (
            f"X_t=[s_(t-{WINDOW_SIZE}),...,s_(t-1)] with shape "
            f"({WINDOW_SIZE},1)"
        ),
        "prediction_target_definition": "y_t=s_t (one-step-ahead forecast)",
        "uses_post_target_observations": False,
        "training_may_follow_test": metodo == "purged_kfold",
        "test_protocol": (
            "method-specific red blocks; no external common test and no final refit"
        ),
        "metric_aggregation_rule": (
            "pool all unique fold test predictions and calculate metrics once"
        ),
        "naive_mse": None,
        "naive_rmse": None,
        "naive_mae": None,
        "naive_mape": None,
        "mse_skill_vs_naive": None,
        "mae_skill_vs_naive": None,
        "idx_test": pooled["raw_target_indices"].astype(int),
        "test_dates": pooled["test_dates"].astype("datetime64[D]"),
    }
    print(
        f"[INFO] Evaluation completed for model '{model_name}', asset "
        f"'{asset_name}' using {METODOS_MAP[metodo]}"
    )
    return result


def evaluate_naive_asset(
    asset_name,
    serie,
    metodo,
    partition_plan=None,
    carbon_config=None,
):
    """Evaluate last-observation persistence on every test fold of one method."""
    if partition_plan is None:
        partition_plan = obtener_particion(serie, WINDOW_SIZE, metodo)
    validate_causal_partition(serie, partition_plan, metodo)
    X_raw, y_raw, raw_target_indices = crear_instancias(serie, WINDOW_SIZE)
    dates_arr = dates_dict[asset_name].to_numpy()
    inference_start = time.perf_counter()
    fold_results = []
    for fold_index, fold in enumerate(partition_plan["folds"], start=1):
        train_pos = np.asarray(fold["train_pos"], dtype=int)
        val_pos = np.asarray(fold["val_pos"], dtype=int)
        test_pos = np.asarray(fold["test_pos"], dtype=int)
        scaler = fit_scaler_from_training(
            X_raw[train_pos], y_raw[train_pos]
        )
        predictions_original = np.asarray(
            X_raw[test_pos, -1], dtype=np.float64
        ).reshape(-1)
        targets_original = y_raw[test_pos].astype(np.float64, copy=True)
        predictions_scaled = transform_y_with_scaler(
            scaler, predictions_original
        )
        targets_scaled = transform_y_with_scaler(scaler, targets_original)
        test_raw_indices = raw_target_indices[test_pos]
        fold_results.append({
            "fold_index": int(fold_index),
            "label": str(fold["label"]),
            "n_train": int(len(train_pos)),
            "n_validation": int(len(val_pos)),
            "n_test": int(len(test_pos)),
            "n_purged": int(len(fold.get("purge_pos", []))),
            "n_embargoed": int(len(fold.get("embargo_pos", []))),
            "train_range": _date_range_for_positions(
                dates_arr, raw_target_indices, train_pos
            ),
            "validation_range": _date_range_for_positions(
                dates_arr, raw_target_indices, val_pos
            ),
            "test_range": _date_range_for_positions(
                dates_arr, raw_target_indices, test_pos
            ),
            "scaler_train_min_usd": float(scaler.data_min_[0]),
            "scaler_train_max_usd": float(scaler.data_max_[0]),
            "predictions_original": predictions_original,
            "targets_original": targets_original,
            "predictions_scaled": predictions_scaled,
            "targets_scaled": targets_scaled,
            "raw_target_indices": test_raw_indices,
            "test_dates": dates_arr[test_raw_indices],
        })
    inference_time = float(time.perf_counter() - inference_start)
    pooled = _pooled_fold_arrays(
        fold_results, f"{asset_name}/Naive Persistence/{metodo}"
    )
    metrics = _pooled_metrics(
        pooled["targets_original"], pooled["predictions_original"],
        pooled["targets_scaled"], pooled["predictions_scaled"],
        context=f"{asset_name}/Naive Persistence/{metodo}/pooled_oof",
    )
    train_positions = np.unique(np.concatenate([
        np.asarray(fold["train_pos"], dtype=int)
        for fold in partition_plan["folds"]
    ]))
    validation_positions = np.unique(np.concatenate([
        np.asarray(fold["val_pos"], dtype=int)
        for fold in partition_plan["folds"]
    ]))
    test_positions = np.asarray(partition_plan["all_test_pos"], dtype=int)
    fold_summaries = [
        _fold_metadata_without_arrays(fold) for fold in fold_results
    ]
    return {
        "model": "Naive Persistence",
        "asset": asset_name,
        "method": metodo,
        "partition_configuration": {
            **PARTITION_CONFIGURATIONS[metodo],
            **_partition_result_metadata(partition_plan),
        },
        "seed": 0,
        **metrics,
        "time": 0.0,
        "inference_time_seconds": inference_time,
        "emissions": None,
        "architecture_protocol": "statistical_baseline",
        "architecture_config": "last-observation persistence",
        "hyperparameter_provenance": "not applicable",
        "hyperparameters_validated": None,
        "hyperparameter_validation_record": None,
        "optimizer": "not applicable",
        "learning_rate": None,
        "training_loss": "not applicable",
        "batch_size": None,
        "early_stopping_patience": None,
        "epochs_trained": 0,
        "fold_best_epochs": [],
        "epoch_selection_rule": "not applicable; no fitting",
        "fold_count": int(len(fold_results)),
        "fold_results": fold_summaries,
        "trainable_parameters": 0,
        "carbon_run_id": None,
        "carbon_run_ids": [],
        "carbon_accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
        "carbon_metadata": {
            "tracking_enabled": False,
            "not_applicable_reason": (
                "Naive Persistence has no model.fit stage; measured inference "
                "lies outside the declared training-emissions boundary."
            ),
            "accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
            "declared_country_iso_code": (carbon_config or {}).get(
                "country_iso_code"
            ),
            "declared_region": (carbon_config or {}).get("region"),
        },
        "preds_list": [(
            pooled["predictions_original"].astype(np.float64),
            pooled["targets_original"].astype(np.float64),
        )],
        "n_train": int(len(train_positions)),
        "n_val": int(len(validation_positions)),
        "n_train_observations_across_folds": int(sum(
            len(fold["train_pos"]) for fold in partition_plan["folds"]
        )),
        "n_validation_observations_across_folds": int(sum(
            len(fold["val_pos"]) for fold in partition_plan["folds"]
        )),
        "n_test": int(len(test_positions)),
        "fold_test_counts": [
            int(len(fold["test_pos"])) for fold in partition_plan["folds"]
        ],
        "fold_test_ranges": [fold["test_range"] for fold in fold_summaries],
        "train_range": _date_range_for_positions(
            dates_arr, raw_target_indices, train_positions
        ),
        "valid_range": _date_range_for_positions(
            dates_arr, raw_target_indices, validation_positions
        ),
        "test_range": _date_range_for_positions(
            dates_arr, raw_target_indices, test_positions
        ),
        "total_samples": int(len(X_raw)),
        "n_original": int(len(serie)),
        "metric_units": (
            "MSE: USD^2; RMSE/MAE: USD; MAPE: percent; "
            "R2/NSE/KGE: dimensionless"
        ),
        "scaler": (
            "one reporting MinMaxScaler(0,1) per fold, fitted only on the "
            "corresponding fold-training observations"
        ),
        "scaler_train_min_usd": [
            float(fold["scaler_train_min_usd"]) for fold in fold_results
        ],
        "scaler_train_max_usd": [
            float(fold["scaler_train_max_usd"]) for fold in fold_results
        ],
        "input_tensor_definition": (
            f"prediction_t=s_(t-1); lag window available was "
            f"[s_(t-{WINDOW_SIZE}),...,s_(t-1)]"
        ),
        "prediction_target_definition": "y_t=s_t (one-step-ahead forecast)",
        "uses_post_target_observations": False,
        "training_may_follow_test": False,
        "test_protocol": (
            "same method-specific red test blocks and target dates as neural models"
        ),
        "metric_aggregation_rule": (
            "pool all unique fold test predictions and calculate metrics once"
        ),
        "naive_mse": metrics["mse"],
        "naive_rmse": metrics["rmse"],
        "naive_mae": metrics["mae"],
        "naive_mape": metrics["mape"],
        "mse_skill_vs_naive": 0.0,
        "mae_skill_vs_naive": 0.0,
        "idx_test": pooled["raw_target_indices"].astype(int),
        "test_dates": pooled["test_dates"].astype("datetime64[D]"),
    }

# %% Configuración de la semilla
seed = 123
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


def set_reproducible_seed(run_seed):
    """Reset Python, NumPy, and TensorFlow RNG state for one repetition."""
    tf.keras.backend.clear_session()
    random.seed(run_seed)
    np.random.seed(run_seed)
    tf.keras.utils.set_random_seed(run_seed)

# %% Parámetros globales
WINDOW_SIZE = 5
LEARNING_RATE = 0.001

EPOCHS_DICT = {
    "Multilayer Perceptron": 150,
    "LSTM": 150,
    "GRU": 150,
    "TRANSFORMER": 150,
    "BiLSTM": 150,
    "Naive Persistence": 0,
}

# Nombres oficiales de métodos
METODOS_MAP = {
    "no_aleatoria": "Chronological Hold-Out",
    "ventana_movil2": "Expanding Window Validation",
    "con_solape": "Overlapping (Rolling) Window Partitioning",
    "time_series_split": "TimeSeriesSplit (Fixed-Window CV)",
    "purged_kfold": "Purged K-Fold Cross-Validation",
    "change_point_partitioning": "Change-Point Based Partitioning",
    "pre-paper_no_shuffle": "Sequential Split (Without Shuffle)",
    "timeseries_generator": "Generator-Based Temporal Partitioning",
}
PARTITION_CONFIGURATIONS = {
    "no_aleatoria": {
        "rounds": [{"train": [0.00, 0.70], "validation": [0.70, 0.85], "test": [0.85, 1.00]}],
    },
    "pre-paper_no_shuffle": {
        "rounds": [{"train": [0.00, 0.60], "validation": [0.60, 0.80], "test": [0.80, 1.00]}],
        "implementation": "chronological sequential split; no shuffling",
    },
    "ventana_movil2": {
        "rounds": [
            {"train": [0.00, 0.40], "validation": [0.40, 0.50], "test": [0.50, 0.60]},
            {"train": [0.00, 0.60], "validation": [0.60, 0.70], "test": [0.70, 0.80]},
            {"train": [0.00, 0.80], "validation": [0.80, 0.90], "test": [0.90, 1.00]},
        ],
    },
    "con_solape": {
        "rounds": [
            {"train": [0.00, 0.30], "validation": [0.30, 0.40], "test": [0.40, 0.50]},
            {"train": [0.20, 0.50], "validation": [0.50, 0.60], "test": [0.60, 0.70]},
            {"train": [0.40, 0.70], "validation": [0.70, 0.80], "test": [0.80, 0.90]},
        ],
        "input_window": WINDOW_SIZE,
        "forecast_horizon": 1,
    },
    "time_series_split": {
        "rounds": [
            {"train": [0.00, 0.30], "validation": [0.30, 0.40], "test": [0.40, 0.50]},
            {"train": [0.10, 0.40], "validation": [0.40, 0.50], "test": [0.50, 0.60]},
            {"train": [0.20, 0.50], "validation": [0.50, 0.60], "test": [0.60, 0.70]},
        ],
        "window_type": "fixed",
    },
    "timeseries_generator": {
        "rounds": [
            {"train": [0.00, 0.30], "validation": [0.30, 0.40], "test": [0.40, 0.50]},
            {"train": [0.20, 0.60], "validation": [0.60, 0.65], "test": [0.65, 0.75]},
        ],
        "input_window": WINDOW_SIZE,
        "forecast_horizon": 1,
    },
    "purged_kfold": {
        "outer_test_folds": 3,
        "outer_test_rule": "three contiguous folds cover every supervised target exactly once",
        "information_interval": f"[t-{WINDOW_SIZE}, t]",
        "purging_rule": "remove training samples whose information intervals overlap validation or test",
        "embargo_fraction_after_test": 0.01,
        "validation_rule": "contiguous safe block selected from the outer training set",
        "training_rule": (
            "all admissible outer-training targets outside validation, purge, and embargo zones"
        ),
        "adaptation": "López de Prado event-label Purged K-Fold adapted to lag-information intervals",
    },
    "change_point_partitioning": {
        **CHANGE_POINT_CONFIG,
        "test": [0.75, 1.00],
        "change_point_search_scope": "strictly before the method-specific test",
        "nominal_training_boundary": 0.40,
    },
}
METODOS_PARTICION = list(METODOS_MAP)
RESULT_TABLE_HEADERS = [
    "Model", "Method", "Seed", "MSE (USD²)", "RMSE (USD)", "MAE (USD)", "MAPE (%)", "R²", "NSE", "KGE", "Fitting time (s)",
    "Emissions (kgCO2eq)", "Train", "Validation", "Test",
    "Train Range", "Validation Range", "Test Range",
    "Total", "Original N"
]
GLOBAL_CSV_COLUMNS = [
    "model", "asset", "method", "partition_configuration", "seed", "mse", "rmse", "mae", "mape",
    "mse_scaled", "rmse_scaled", "mae_scaled", "r2", "nse", "kge", "time",
    "inference_time_seconds", "emissions",
    "architecture_protocol", "architecture_config", "optimizer", "learning_rate",
    "training_loss", "hyperparameter_provenance", "hyperparameters_validated",
    "hyperparameter_validation_record", "batch_size", "early_stopping_patience",
    "epochs_trained", "fold_best_epochs", "epoch_selection_rule", "fold_count",
    "fold_results", "trainable_parameters", "n_train", "n_val",
    "n_train_observations_across_folds", "n_validation_observations_across_folds",
    "n_test", "fold_test_counts", "fold_test_ranges",
    "train_range", "valid_range", "test_range",
    "total_samples", "n_original", "metric_units", "scaler", "scaler_train_min_usd",
    "scaler_train_max_usd", "input_tensor_definition", "prediction_target_definition",
    "uses_post_target_observations", "training_may_follow_test",
    "test_protocol", "metric_aggregation_rule", "naive_mse", "naive_rmse",
    "naive_mae", "naive_mape", "mse_skill_vs_naive", "mae_skill_vs_naive",
    "carbon_run_id", "carbon_run_ids",
    "carbon_accounting_boundary",
    "carbon_metadata", "prediction_file"
    ]

# Diccionario con todos los activos a entrenar
activos = {
    "Bitcoin": {"ticker": "BTC-USD", "start": "2009-01-01", "end": "2025-01-01"},
    "Ethereum": {"ticker": "ETH-USD", "start": "2015-01-01", "end": "2025-01-01"},
    "Litecoin": {"ticker": "LTC-USD", "start": "2011-01-01", "end": "2025-01-01"},
    "Bitcoin Cash": {"ticker": "BCH-USD", "start": "2017-01-01", "end": "2025-01-01"},
    "Cardano": {"ticker": "ADA-USD", "start": "2017-01-01", "end": "2025-01-01"},
    "Polkadot": {"ticker": "DOT-USD", "start": "2020-01-01", "end": "2025-01-01"},
    "Solana": {"ticker": "SOL-USD", "start": "2020-01-01", "end": "2025-01-01"},
    "Binance Coin": {"ticker": "BNB-USD", "start": "2017-01-01", "end": "2025-01-01"},
    "TRON": {"ticker": "TRX-USD", "start": "2017-01-01", "end": "2025-01-01"},
    "Monero": {"ticker": "XMR-USD", "start": "2014-01-01", "end": "2025-01-01"},
    "Ripple": {"ticker": "XRP-USD", "start": "2012-01-01", "end": "2025-01-01"},
    "Stellar": {"ticker": "XLM-USD", "start": "2014-01-01", "end": "2025-01-01"}
}

_EXPECTED_HYPERPARAMETER_PAIRS = {
    (asset_name, model_name)
    for asset_name in activos
    for model_name in NEURAL_MODELS
}
_CONFIGURED_HYPERPARAMETER_PAIRS = {
    (asset_name, model_name)
    for asset_name, model_configs in optimal_hyperparams.items()
    for model_name in model_configs
}
if _CONFIGURED_HYPERPARAMETER_PAIRS != _EXPECTED_HYPERPARAMETER_PAIRS:
    raise RuntimeError(
        "Hyperparameter coverage must be exactly 12 assets x 5 neural models. "
        f"Missing={sorted(_EXPECTED_HYPERPARAMETER_PAIRS - _CONFIGURED_HYPERPARAMETER_PAIRS)}, "
        f"unexpected={sorted(_CONFIGURED_HYPERPARAMETER_PAIRS - _EXPECTED_HYPERPARAMETER_PAIRS)}"
    )
_EXPERT_HYPERPARAMETER_PAIRS = {
    (asset_name, model_name)
    for asset_name, model_configs in EXPERT_INITIAL_HYPERPARAMS.items()
    for model_name in model_configs
}
if (
    len(SUPPLIED_HYPERPARAMETER_PAIRS) != 20
    or _EXPERT_HYPERPARAMETER_PAIRS
    != _EXPECTED_HYPERPARAMETER_PAIRS - SUPPLIED_HYPERPARAMETER_PAIRS
):
    raise RuntimeError(
        "Expected exactly 20 supplied configurations and 40 non-overlapping "
        "expert initial candidates covering all remaining pairs."
    )

# Raw series and dates. Scaling is deliberately performed only after each
# train/validation/test partition has been created.
series_dict = {}
dates_dict = {}


def parse_selection(selection, available, label):
    if selection is None:
        return None
    selection = selection.strip()
    if not selection or selection.lower() == "all":
        return list(available)

    selected = [item.strip() for item in selection.split(",") if item.strip()]
    invalid = [item for item in selected if item not in available]
    if invalid:
        raise ValueError(
            f"Invalid {label}: {', '.join(invalid)}. Available options: all, {', '.join(available)}"
        )
    return selected


def prompt_selection(label, available):
    print(f"Available {label} options: all, " + ", ".join(available))
    raw_value = input(f"Enter the {label} to analyze (options: all, " + ", ".join(available) + "): ")
    return parse_selection(raw_value, available, label)


def parse_seeds(value):
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one integer seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds
def _validate_hyperparameter_config(model_name, config):
    """Validate one serializable architecture/optimizer candidate."""
    if not isinstance(config, dict):
        raise ValueError(f"{model_name}: hyperparameters must be a dictionary.")
    common = {"dropout", "learning_rate", "epochs"}
    missing = common.difference(config)
    if missing:
        raise ValueError(
            f"{model_name}: missing hyperparameters {sorted(missing)}."
        )
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError(f"{model_name}: dropout must lie in [0,1).")
    if float(config["learning_rate"]) <= 0.0:
        raise ValueError(f"{model_name}: learning_rate must be positive.")
    if int(config["epochs"]) <= 0:
        raise ValueError(f"{model_name}: epochs must be positive.")
    if model_name == "Multilayer Perceptron":
        widths = config.get("layers")
    elif model_name in {"LSTM", "GRU", "BiLSTM"}:
        widths = config.get("units")
        if not isinstance(widths, list) or len(widths) != 2:
            raise ValueError(f"{model_name}: exactly two recurrent widths are required.")
    elif model_name == "TRANSFORMER":
        if int(config.get("d_model", 0)) <= 0:
            raise ValueError("TRANSFORMER: d_model must be positive.")
        if int(config.get("num_heads", 0)) <= 0:
            raise ValueError("TRANSFORMER: num_heads must be positive.")
        widths = [config["d_model"]]
    else:
        raise ValueError(f"Unsupported neural model: {model_name}")
    if (
        not isinstance(widths, list)
        or not widths
        or any(int(value) <= 0 for value in widths)
    ):
        raise ValueError(f"{model_name}: all layer widths must be positive.")


def _scaled_width(value, factor):
    return max(8, int(round((float(value) * float(factor)) / 8.0) * 8))


def generate_hyperparameter_candidates(model_name, initial_config, trials):
    """Generate a deterministic local search with the expert value first."""
    if trials <= 0:
        raise ValueError("tuning trials must be greater than zero.")
    _validate_hyperparameter_config(model_name, initial_config)
    width_factors = (1.0, 0.75, 1.25, 1.50)
    dropout_offsets = (0.0, -0.05, 0.05, 0.10)
    learning_rate_factors = (1.0, 0.5, 2.0, 0.75, 1.5)
    if model_name == "TRANSFORMER":
        base_heads = int(initial_config["num_heads"])
        head_options = tuple(dict.fromkeys(
            [base_heads, 2, 4, 8]
        ))
    else:
        base_heads = None
        head_options = (None,)

    candidates = []
    for width_factor, dropout_offset, lr_factor, heads in itertools.product(
        width_factors,
        dropout_offsets,
        learning_rate_factors,
        head_options,
    ):
        candidate = json.loads(json.dumps(initial_config))
        candidate["dropout"] = round(
            min(0.50, max(0.0, float(initial_config["dropout"]) + dropout_offset)),
            4,
        )
        candidate["learning_rate"] = float(
            float(initial_config["learning_rate"]) * lr_factor
        )
        if model_name == "Multilayer Perceptron":
            candidate["layers"] = [
                int(value) if width_factor == 1.0
                else _scaled_width(value, width_factor)
                for value in initial_config["layers"]
            ]
        elif model_name in {"LSTM", "GRU", "BiLSTM"}:
            candidate["units"] = [
                int(value) if width_factor == 1.0
                else _scaled_width(value, width_factor)
                for value in initial_config["units"]
            ]
        else:
            candidate["d_model"] = (
                int(initial_config["d_model"])
                if width_factor == 1.0 else _scaled_width(
                    initial_config["d_model"], width_factor
                )
            )
            candidate["num_heads"] = int(heads)
        distance = (
            int(width_factor != 1.0)
            + int(dropout_offset != 0.0)
            + int(lr_factor != 1.0)
            + int(heads != base_heads)
        )
        candidates.append((
            distance,
            abs(width_factor - 1.0),
            abs(dropout_offset),
            abs(np.log(lr_factor)),
            0 if heads == base_heads else int(heads),
            candidate,
        ))
    candidates.sort(key=lambda item: item[:-1])
    unique = []
    seen = set()
    for *_, candidate in candidates:
        canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(candidate)
        if len(unique) == trials:
            break
    if len(unique) < trials:
        print(
            f"[WARNING] {model_name}: only {len(unique)} unique local "
            f"candidates could be generated (requested {trials})."
        )
    return unique


def _hyperparameter_tuning_folds(n_supervised):
    schedules = (
        ((0.00, 0.50), (0.50, 0.65)),
        ((0.00, 0.65), (0.65, 0.80)),
        ((0.00, 0.80), (0.80, 1.00)),
    )
    return [
        {
            "label": f"calibration_round_{number}",
            "train_pos": _positions_between(n_supervised, *train_bounds),
            "val_pos": _positions_between(n_supervised, *validation_bounds),
        }
        for number, (train_bounds, validation_bounds) in enumerate(
            schedules, start=1
        )
    ]


def _calibration_series_hash(values, dates):
    payload = np.column_stack([
        pd.DatetimeIndex(dates).asi8,
        np.asarray(values, dtype=np.float64).view("int64"),
    ]).tobytes()
    return hashlib.sha256(payload).hexdigest()


def run_temporal_hyperparameter_optimization(
    series_to_analyze,
    models_to_analyze,
    calibration_end_date,
    trials,
    tuning_seeds,
    output_path,
    batch_size,
    early_stopping_patience,
    track_emissions,
    carbon_config,
):
    """Tune asset/model configurations on a strictly earlier calibration period."""
    global ARCHITECTURE_PROTOCOL
    neural_models = [
        model for model in models_to_analyze if model in NEURAL_MODELS
    ]
    if not neural_models:
        raise ValueError("Hyperparameter optimization requires a neural model.")
    cutoff = pd.Timestamp(calibration_end_date)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    calibration = {}
    calibration_hashes = {}
    for asset_name in series_to_analyze:
        dates = pd.DatetimeIndex(dates_dict[asset_name])
        mask = dates <= cutoff
        values = np.asarray(series_dict[asset_name], dtype=np.float64)[mask]
        selected_dates = dates[mask]
        if len(values) <= WINDOW_SIZE + 60:
            raise ValueError(
                f"{asset_name}: only {len(values)} calibration observations "
                f"exist on or before {cutoff.date()}."
            )
        calibration[asset_name] = (values, selected_dates)
        calibration_hashes[asset_name] = _calibration_series_hash(
            values, selected_dates
        )

    initial_candidates = {
        asset: {
            model: optimal_hyperparams[asset][model]
            for model in neural_models
        }
        for asset in series_to_analyze
    }
    identity = {
        "schema_version": 1,
        "deliverable_id": DELIVERABLE_ID,
        "source_sha256": file_sha256(os.path.abspath(__file__)),
        "software_versions": distribution_versions(),
        "protocol": "three expanding-window temporal validation rounds",
        "validation_data_relation": "disjoint_calibration_prefix",
        "calibration_end_date_inclusive": cutoff.strftime("%Y-%m-%d"),
        "window_size": WINDOW_SIZE,
        "assets": list(series_to_analyze),
        "models": list(neural_models),
        "trials_per_pair": int(trials),
        "tuning_seeds": [int(value) for value in tuning_seeds],
        "batch_size": int(batch_size),
        "early_stopping_patience": int(early_stopping_patience),
        "emissions_tracking": bool(track_emissions),
        "carbon_configuration": carbon_config,
        "carbon_accounting_boundary": TUNING_CARBON_ACCOUNTING_BOUNDARY,
        "calibration_hashes": calibration_hashes,
        "initial_candidates": initial_candidates,
    }
    fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "tuning_fingerprint": fingerprint,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": identity,
        "pairs": {},
    }
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as input_file:
            previous = json.load(input_file)
        if previous.get("tuning_fingerprint") != fingerprint:
            raise RuntimeError(
                "The tuning output belongs to a different calibration "
                "experiment; choose another --tuning-output path."
            )
        manifest = previous
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()

    previous_protocol = ARCHITECTURE_PROTOCOL
    ARCHITECTURE_PROTOCOL = "individually_tuned"
    try:
        total_pairs = len(series_to_analyze) * len(neural_models)
        completed_pairs = 0
        for asset_name in series_to_analyze:
            values, selected_dates = calibration[asset_name]
            X_raw, y_raw, raw_indices = crear_instancias(
                values, WINDOW_SIZE
            )
            folds = _hyperparameter_tuning_folds(len(X_raw))
            for model_name in neural_models:
                pair_key = f"{asset_name}::{model_name}"
                initial = initial_candidates[asset_name][model_name]
                candidates = generate_hyperparameter_candidates(
                    model_name, initial, trials
                )
                pair_record = manifest["pairs"].setdefault(pair_key, {
                    "asset": asset_name,
                    "model": model_name,
                    "status": "running",
                    "initial_provenance": HYPERPARAMETER_PROVENANCE[
                        (asset_name, model_name)
                    ],
                    "initial_config": initial,
                    "trials": [],
                })
                finished_ids = {
                    trial["candidate_id"]
                    for trial in pair_record.get("trials", [])
                    if trial.get("status") == "completed"
                }
                for candidate_index, candidate in enumerate(candidates, start=1):
                    canonical = json.dumps(
                        candidate, sort_keys=True, separators=(",", ":")
                    )
                    candidate_id = hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()[:16]
                    if candidate_id in finished_ids:
                        continue
                    print(
                        f"[TUNING] {asset_name}/{model_name}: candidate "
                        f"{candidate_index}/{len(candidates)}"
                    )
                    optimal_hyperparams[asset_name][model_name] = candidate
                    validation_scores = []
                    best_epochs = []
                    fit_times = []
                    fit_emissions = []
                    carbon_run_ids = []
                    parameter_counts = []
                    for tuning_seed in tuning_seeds:
                        for fold_index, fold in enumerate(folds, start=1):
                            train_pos = fold["train_pos"]
                            val_pos = fold["val_pos"]
                            scaler = fit_scaler_from_training(
                                X_raw[train_pos], y_raw[train_pos]
                            )
                            X_train = transform_X_with_scaler(
                                scaler, X_raw[train_pos]
                            )
                            y_train = transform_y_with_scaler(
                                scaler, y_raw[train_pos]
                            )
                            X_val = transform_X_with_scaler(
                                scaler, X_raw[val_pos]
                            )
                            y_val = transform_y_with_scaler(
                                scaler, y_raw[val_pos]
                            )
                            fit_seed = int(
                                (int(tuning_seed) + 1009 * fold_index)
                                % (2 ** 31 - 1)
                            )
                            set_reproducible_seed(fit_seed)
                            output = entrenar_evaluar_modelo(
                                lambda size: modelos[model_name](
                                    size, asset_name
                                ),
                                WINDOW_SIZE,
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_val,
                                y_val,
                                int(candidate["epochs"]),
                                model_name,
                                batch_size=batch_size,
                                early_stopping_patience=early_stopping_patience,
                                track_emissions=track_emissions,
                                carbon_config=carbon_config,
                                run_context={
                                    "asset": asset_name,
                                    "model": model_name,
                                    "stage": "hyperparameter_calibration",
                                    "candidate_id": candidate_id,
                                    "fold": fold["label"],
                                    "seed": int(tuning_seed),
                                    "accounting_boundary": (
                                        TUNING_CARBON_ACCOUNTING_BOUNDARY
                                    ),
                                },
                            )
                            validation_scores.append(float(output[1]))
                            fit_times.append(float(output[4]))
                            best_epochs.append(int(output[10]))
                            parameter_counts.append(int(output[11]))
                            if output[5] is not None:
                                fit_emissions.append(float(output[5]))
                            if output[8].get("run_id"):
                                carbon_run_ids.append(output[8]["run_id"])
                    if track_emissions and len(fit_emissions) != len(
                        validation_scores
                    ):
                        raise RuntimeError(
                            f"Incomplete tuning emissions for {pair_key}/"
                            f"{candidate_id}."
                        )
                    pair_record["trials"].append({
                        "candidate_id": candidate_id,
                        "candidate_index": int(candidate_index),
                        "status": "completed",
                        "config": candidate,
                        "mean_validation_rmse_scaled": float(
                            np.mean(validation_scores)
                        ),
                        "std_validation_rmse_scaled": float(
                            np.std(validation_scores, ddof=1)
                            if len(validation_scores) > 1 else 0.0
                        ),
                        "validation_scores": validation_scores,
                        "best_epochs": best_epochs,
                        "fit_time_seconds": float(sum(fit_times)),
                        "emissions_kgco2eq": (
                            float(sum(fit_emissions))
                            if track_emissions else None
                        ),
                        "carbon_run_ids": carbon_run_ids,
                        "carbon_accounting_boundary": (
                            TUNING_CARBON_ACCOUNTING_BOUNDARY
                        ),
                        "trainable_parameters": int(parameter_counts[0]),
                        "evaluations": int(len(validation_scores)),
                    })
                    pair_record["updated_at_utc"] = dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat()
                    atomic_write_json(output_path, manifest)

                completed_trials = [
                    trial for trial in pair_record["trials"]
                    if trial.get("status") == "completed"
                ]
                if len(completed_trials) != len(candidates):
                    raise RuntimeError(
                        f"Incomplete tuning trials for {pair_key}."
                    )
                winner = min(
                    completed_trials,
                    key=lambda trial: (
                        trial["mean_validation_rmse_scaled"],
                        trial["trainable_parameters"],
                        trial["candidate_id"],
                    ),
                )
                pair_record.update({
                    "status": "validated",
                    "selected_candidate_id": winner["candidate_id"],
                    "selected_config": winner["config"],
                    "objective": "mean validation RMSE on fold-training-scaled values",
                    "objective_value": winner[
                        "mean_validation_rmse_scaled"
                    ],
                    "selection_tie_break": (
                        "lower trainable parameter count, then candidate hash"
                    ),
                    "validation_rounds": len(folds),
                    "validation_seeds": [int(value) for value in tuning_seeds],
                    "calibration_observed_start": selected_dates[
                        0
                    ].strftime("%Y-%m-%d"),
                    "calibration_observed_end": selected_dates[
                        -1
                    ].strftime("%Y-%m-%d"),
                    "calibration_observation_count": int(len(values)),
                    "calibration_series_sha256": calibration_hashes[asset_name],
                    "validated_at_utc": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                })
                optimal_hyperparams[asset_name][model_name] = json.loads(
                    json.dumps(winner["config"])
                )
                HYPERPARAMETER_VALIDATION_RECORDS[
                    (asset_name, model_name)
                ] = pair_record
                HYPERPARAMETER_PROVENANCE[(asset_name, model_name)] = (
                    "temporally_validated_local_search_from_"
                    + pair_record["initial_provenance"]
                )
                completed_pairs += 1
                manifest["completed_pair_count"] = int(completed_pairs)
                manifest["expected_pair_count"] = int(total_pairs)
                atomic_write_json(output_path, manifest)
    finally:
        ARCHITECTURE_PROTOCOL = previous_protocol

    manifest.update({
        "status": "completed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_pair_count": int(len(manifest["pairs"])),
        "expected_pair_count": int(
            len(series_to_analyze) * len(neural_models)
        ),
        "selected_configurations": {
            asset: {
                model: manifest["pairs"][
                    f"{asset}::{model}"
                ]["selected_config"]
                for model in neural_models
            }
            for asset in series_to_analyze
        },
        "scientific_use_constraint": (
            "Use these configurations only on observations strictly later "
            "than calibration_end_date_inclusive, unless the validation data "
            "were independently replaced and documented."
        ),
    })
    atomic_write_json(output_path, manifest)
    print(
        f"[INFO] Temporal hyperparameter validation completed for "
        f"{len(manifest['pairs'])} asset/model pairs: '{output_path}'"
    )
    return manifest


def load_validated_hyperparameters(
    path,
    required_assets,
    required_models,
    evaluation_start=None,
):
    """Load, validate, and activate a completed tuning manifest."""
    with open(path, "r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("status") != "completed":
        raise RuntimeError("The hyperparameter validation manifest is incomplete.")
    identity = manifest.get("identity", {})
    relation = identity.get("validation_data_relation")
    calibration_end = identity.get("calibration_end_date_inclusive")
    if relation == "disjoint_calibration_prefix":
        if not evaluation_start:
            raise RuntimeError(
                "--evaluation-start is required with a calibration-prefix "
                "manifest so validation and experimental observations remain disjoint."
            )
        if pd.Timestamp(evaluation_start) <= pd.Timestamp(calibration_end):
            raise RuntimeError(
                "--evaluation-start must be strictly later than the "
                f"calibration cutoff {calibration_end}."
            )

    neural_models = [
        model for model in required_models if model in NEURAL_MODELS
    ]
    pairs = manifest.get("pairs", {})
    missing = []
    for asset_name in required_assets:
        for model_name in neural_models:
            pair_key = f"{asset_name}::{model_name}"
            record = pairs.get(pair_key)
            if not record or record.get("status") != "validated":
                missing.append(pair_key)
                continue
            config = record.get("selected_config")
            _validate_hyperparameter_config(model_name, config)
            optimal_hyperparams[asset_name][model_name] = json.loads(
                json.dumps(config)
            )
            HYPERPARAMETER_VALIDATION_RECORDS[
                (asset_name, model_name)
            ] = record
            HYPERPARAMETER_PROVENANCE[(asset_name, model_name)] = (
                "temporally_validated_configuration"
            )
    if missing:
        raise RuntimeError(
            "Validated hyperparameters are missing for: "
            + ", ".join(missing)
        )
    return manifest


def restrict_to_evaluation_period(evaluation_start):
    """Restrict every asset to the predeclared post-calibration experiment."""
    global series_dict, dates_dict
    start = pd.Timestamp(evaluation_start)
    for asset_name in list(series_dict):
        dates = pd.DatetimeIndex(dates_dict[asset_name])
        mask = dates >= start
        values = np.asarray(series_dict[asset_name], dtype=np.float64)[mask]
        selected_dates = dates[mask]
        if len(values) <= WINDOW_SIZE + 60:
            raise ValueError(
                f"{asset_name}: only {len(values)} observations remain on or "
                f"after {start.date()}."
            )
        series_dict[asset_name] = values
        dates_dict[asset_name] = selected_dates
        print(
            f"[INFO] Experimental period for {asset_name}: "
            f"{selected_dates[0].date()} to {selected_dates[-1].date()} "
            f"({len(values)} observations)"
        )



def asset_cache_path(cache_dir, asset_name):
    safe_name = asset_name.replace(" ", "_").replace("/", "_")
    info = activos[asset_name]
    filename = (
        f"{safe_name}_{info['ticker']}_{info['start']}_{info['end']}_raw_close.csv"
    )
    return os.path.join(cache_dir, filename)


def normalize_market_series(data, asset_name):
    """Normalize one downloaded/cached Close series and reject bad inputs."""
    if isinstance(data, pd.DataFrame):
        if data.empty:
            data = pd.Series(dtype=np.float64)
        elif "Close" in data.columns:
            data = data["Close"]
        else:
            data = data.iloc[:, 0]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = pd.Series(data).copy()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()]
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)
    data = pd.to_numeric(data, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if data.empty:
        raise ValueError(f"No valid Close data are available for {asset_name}.")
    if np.any(data.to_numpy(dtype=np.float64) <= 0):
        raise ValueError(
            f"Non-positive closing price found for {asset_name}; "
            "MAPE would be undefined."
        )
    data.name = "Close"
    return data


def write_market_cache_atomic(cache_path, data):
    parent = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_market_", suffix=".csv", dir=parent
    )
    os.close(descriptor)
    try:
        data.to_frame(name="Close").to_csv(
            temporary_path, encoding="utf-8", lineterminator="\n"
        )
        os.replace(temporary_path, cache_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def load_asset_data(asset_name, cache_dir=None, refresh_data=False):
    info = activos[asset_name]
    cache_path = asset_cache_path(cache_dir, asset_name) if cache_dir else None
    data = None
    if cache_path and os.path.exists(cache_path) and not refresh_data:
        print(f"[INFO] Loading cached data for {asset_name} from {cache_path}")
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            data = normalize_market_series(cached, asset_name)
        except Exception as cache_error:
            print(
                f"[WARNING] Ignoring invalid or empty cache for {asset_name}: "
                f"{cache_error}"
            )
            data = None
    if data is None:
        print(f"[INFO] Downloading data for {asset_name} (ticker: {info['ticker']})...")
        downloaded = yf.download(
            info["ticker"],
            start=info["start"],
            end=info["end"],
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False
        )
        if downloaded is None or downloaded.empty:
            raise ValueError(
                f"Yahoo Finance returned no data for {asset_name} "
                f"({info['ticker']}); the existing cache was not overwritten."
            )
        if "Close" not in downloaded.columns:
            raise ValueError(
                f"Yahoo Finance response has no Close column for {asset_name}."
            )
        data = downloaded["Close"]
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        data = normalize_market_series(data, asset_name)
        if cache_path:
            write_market_cache_atomic(cache_path, data)
            print(f"[INFO] Cached data for {asset_name} in {cache_path}")
    return normalize_market_series(data, asset_name)


def download_series(selected_assets, cache_dir=None, refresh_data=False):
    loaded_series = {}
    loaded_dates = {}

    for nombre in selected_assets:
        data = load_asset_data(nombre, cache_dir, refresh_data)
        loaded_series[nombre] = data.to_numpy(dtype=np.float64).reshape(-1)
        loaded_dates[nombre] = data.index
        print(f"[INFO] Download completed for {nombre}. Records: {len(data)}")

    return loaded_series, loaded_dates


def finish_plot(fig=None, show_plots=False):
    if show_plots:
        plt.show()
    else:
        plt.close(fig if fig is not None else plt.gcf())


def _contiguous_spans(positions, scale):
    positions = np.asarray(positions, dtype=int)
    if positions.size == 0:
        return []
    split_points = np.where(np.diff(positions) != 1)[0] + 1
    groups = np.split(positions, split_points)
    return [
        (float(group[0] * scale), float(len(group) * scale))
        for group in groups
    ]


def run_partition_visualizations(
    methods_to_analyze=None,
    show_plots=False,
    output_dir="paper_figures",
):
    """Draw the eight method-specific train/validation/test protocols."""
    methods_to_analyze = methods_to_analyze or METODOS_PARTICION
    os.makedirs(output_dir, exist_ok=True)
    # Exactly 1,000 supervised targets makes percentages transparent. The
    # deterministic level shifts give PELT meaningful candidate breakpoints.
    synthetic_series = np.concatenate([
        np.linspace(100.0, 140.0, 255),
        np.linspace(190.0, 240.0, 250),
        np.linspace(150.0, 210.0, 250),
        np.linspace(260.0, 330.0, 250),
    ])
    plans = {
        method: obtener_particion(synthetic_series, WINDOW_SIZE, method)
        for method in methods_to_analyze
    }
    row_count = sum(len(plan["folds"]) for plan in plans.values())
    fig, axis = plt.subplots(figsize=(14, max(8.0, 0.52 * row_count + 2.6)))
    colors = {
        "train": "#377eb8",
        "validation": "#ff9f1c",
        "test": "#e15759",
        "purge": "#a7a7a7",
        "embargo": "#8d6e63",
    }
    y_position = float(row_count)
    method_ticks = []
    row_height = 0.62
    protocol_metadata = {}
    for method in methods_to_analyze:
        plan = plans[method]
        scale = 100.0 / plan["n_supervised"]
        group_rows = []
        protocol_metadata[method] = _partition_result_metadata(plan)
        for fold_number, fold in enumerate(plan["folds"], start=1):
            group_rows.append(y_position)
            for key, color_key in (
                ("train_pos", "train"),
                ("val_pos", "validation"),
                ("test_pos", "test"),
                ("purge_pos", "purge"),
                ("embargo_pos", "embargo"),
            ):
                for span in _contiguous_spans(fold.get(key, []), scale):
                    axis.broken_barh(
                        [span],
                        (y_position - row_height / 2.0, row_height),
                        facecolors=colors[color_key],
                        edgecolors="white",
                        linewidth=0.4,
                    )
            axis.text(
                100.8,
                y_position,
                f"r{fold_number}",
                va="center",
                fontsize=7.5,
                color="#444444",
            )
            y_position -= 1.0
        method_ticks.append((float(np.mean(group_rows)), METODOS_MAP[method]))
        y_position -= 0.38

    axis.set_xlim(0, 104)
    axis.set_ylim(y_position + 0.2, row_count + 1.4)
    axis.set_xlabel("Normalized supervised timeline (%)")
    axis.set_yticks([value for value, _ in method_ticks])
    axis.set_yticklabels([label for _, label in method_ticks], fontsize=8.5)
    axis.set_xticks(np.arange(0, 101, 10))
    axis.grid(axis="x", linestyle=":", alpha=0.35)
    axis.set_axisbelow(True)
    axis.set_title(
        "Method-specific temporal train/validation/test folds",
        fontsize=13,
        weight="bold",
        pad=24,
    )
    axis.legend(
        handles=[
            Patch(facecolor=colors["train"], label="Fold training"),
            Patch(facecolor=colors["validation"], label="Early-stopping validation"),
            Patch(facecolor=colors["test"], label="Out-of-sample test"),
            Patch(facecolor=colors["purge"], label="Purged targets"),
            Patch(facecolor=colors["embargo"], label="Embargoed targets"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    png_path = os.path.join(output_dir, "temporal_partition_protocol.png")
    pdf_path = os.path.join(output_dir, "temporal_partition_protocol.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    atomic_write_json(
        os.path.join(output_dir, "temporal_partition_protocol.json"),
        protocol_metadata,
    )
    print(f"[INFO] Revised partition diagram saved as '{png_path}' and '{pdf_path}'")
    finish_plot(fig, show_plots)


def write_individual_tables(asset_name, asset_results):
    print(f"[INFO] INDIVIDUAL RESULTS FOR ASSET '{asset_name}'")
    for model, res_list in asset_results.items():
        if not res_list:
            continue

        tabla_individual = []
        for r in res_list:
            emis_str = f"{r['emissions']:.7f}" if r["emissions"] is not None else "N/A"
            tabla_individual.append([
                model,
                METODOS_MAP[r["method"]],
                r["seed"],
                f"{r['mse']:.4f}" if r["mse"] is not None else "N/A",
                f"{r['rmse']:.4f}" if r["rmse"] is not None else "N/A",
                f"{r['mae']:.4f}" if r["mae"] is not None else "N/A",
                f"{r['mape']:.4f}" if r["mape"] is not None else "N/A",
                f"{r['r2']:.4f}" if r["r2"] is not None else "N/A",
                f"{r['nse']:.4f}" if r["nse"] is not None else "N/A",
                f"{r['kge']:.4f}" if r["kge"] is not None else "N/A",
                f"{r['time']:.4f}" if r["time"] is not None else "N/A",
                emis_str,
                r["n_train"],
                r["n_val"],
                r["n_test"],
                r["train_range"],
                r["valid_range"],
                r["test_range"],
                r["total_samples"],
                r["n_original"]
            ])

        table_output = tabulate(
            tabla_individual,
            headers=RESULT_TABLE_HEADERS,
            tablefmt="grid"
        )
        print(table_output)
        safe_asset = sanitize_run_component(asset_name)
        safe_model = sanitize_run_component(model)
        atomic_write_text(
            f"table_{safe_asset}__{safe_model}.txt",
            table_output,
        )


def result_checkpoint_paths(checkpoint_dir, combination_id):
    base_dir = os.path.join(checkpoint_dir, "results")
    return (
        os.path.join(base_dir, f"{combination_id}.json"),
        os.path.join(base_dir, f"{combination_id}__predictions.npz"),
    )


def save_result_checkpoint(
    checkpoint_dir,
    combination_id,
    run_fingerprint,
    result,
):
    """Save metadata atomically and keep large prediction arrays separate."""
    if not checkpoint_dir:
        return
    metadata_path, predictions_path = result_checkpoint_paths(
        checkpoint_dir, combination_id
    )
    predictions, targets = result["preds_list"][0]
    atomic_save_npz(
        predictions_path,
        predictions=np.asarray(predictions, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        raw_target_indices=np.asarray(result["idx_test"], dtype=np.int64),
        test_dates=np.asarray(result["test_dates"], dtype="datetime64[D]"),
    )
    metadata_result = dict(result)
    metadata_result.pop("preds_list", None)
    metadata_result.pop("idx_test", None)
    metadata_result.pop("test_dates", None)
    atomic_write_json(metadata_path, {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "combination_id": combination_id,
        "prediction_file": os.path.relpath(
            predictions_path, os.path.dirname(metadata_path)
        ),
        "result": metadata_result,
    })


def load_result_checkpoint(
    checkpoint_dir,
    combination_id,
    run_fingerprint,
):
    if not checkpoint_dir:
        return None
    metadata_path, _ = result_checkpoint_paths(checkpoint_dir, combination_id)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported result checkpoint: {metadata_path}")
    if payload.get("status") != "completed":
        raise RuntimeError(f"Incomplete result checkpoint: {metadata_path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(
            f"Result checkpoint belongs to another experiment: {metadata_path}"
        )
    if payload.get("combination_id") != combination_id:
        raise RuntimeError(f"Result checkpoint key mismatch: {metadata_path}")
    predictions_path = os.path.normpath(os.path.join(
        os.path.dirname(metadata_path), payload["prediction_file"]
    ))
    if not os.path.exists(predictions_path):
        raise RuntimeError(
            f"Prediction array is missing for checkpoint: {metadata_path}"
        )
    with np.load(predictions_path, allow_pickle=False) as arrays:
        result = payload["result"]
        predictions = arrays["predictions"].astype(np.float64, copy=True)
        targets = arrays["targets"].astype(np.float64, copy=True)
        result["idx_test"] = arrays["raw_target_indices"].astype(
            int, copy=True
        )
        result["test_dates"] = arrays["test_dates"].astype(
            "datetime64[D]", copy=True
        )
        result["preds_list"] = [(predictions, targets)]
    return result


def build_partition_cache(series_to_analyze, methods_to_analyze):
    """Build and validate each asset/method protocol once."""
    partition_cache = {}
    for asset_name in series_to_analyze:
        partition_cache[asset_name] = {}
        seen_partition_signatures = {}
        for metodo in methods_to_analyze:
            try:
                partition = obtener_particion(
                    series_dict[asset_name], WINDOW_SIZE, metodo
                )
                validate_causal_partition(
                    series_dict[asset_name], partition, metodo
                )
            except Exception as error:
                raise RuntimeError(
                    f"Partition construction failed for "
                    f"{asset_name}/{metodo}: {error}"
                ) from error
            signature = tuple(
                (
                    fold["label"],
                    tuple(np.asarray(fold["train_pos"], dtype=int)),
                    tuple(np.asarray(fold["val_pos"], dtype=int)),
                    tuple(np.asarray(fold["test_pos"], dtype=int)),
                    tuple(np.asarray(fold.get("purge_pos", []), dtype=int)),
                    tuple(np.asarray(fold.get("embargo_pos", []), dtype=int)),
                )
                for fold in partition["folds"]
            )
            if signature in seen_partition_signatures:
                print(
                    "[WARNING] Exact partition-index equivalence for "
                    f"{asset_name}: '{metodo}' and "
                    f"'{seen_partition_signatures[signature]}'."
                )
            else:
                seen_partition_signatures[signature] = metodo
            partition_cache[asset_name][metodo] = partition
    return partition_cache


def run_evaluation(
    series_to_analyze,
    models_to_analyze,
    methods_to_analyze,
    run_seeds,
    epochs_override=None,
    batch_size=None,
    early_stopping_patience=None,
    track_emissions=True,
    carbon_config=None,
    checkpoint_dir=None,
    run_fingerprint=None,
):
    print("[INFO] Starting all asset/model/method/seed evaluations")
    neural_models = [
        model for model in models_to_analyze if model in NEURAL_MODELS
    ]
    include_naive = "Naive Persistence" in models_to_analyze
    total_comb = (
        len(series_to_analyze)
        * len(neural_models)
        * len(methods_to_analyze)
        * len(run_seeds)
        + len(series_to_analyze)
        * len(methods_to_analyze)
        * int(include_naive)
    )
    current_comb = 0
    global_results_by_asset = {}
    partition_cache = build_partition_cache(
        series_to_analyze, methods_to_analyze
    )

    for asset_name in series_to_analyze:
        global_results_by_asset[asset_name] = {
            model_name: [] for model_name in models_to_analyze
        }
        naive_by_method = {}
        if include_naive:
            for metodo in methods_to_analyze:
                current_comb += 1
                combination_id = combination_identifier(
                    asset_name, "Naive Persistence", metodo, 0
                )
                print(
                    f"[INFO] Combination {current_comb}/{total_comb}: Asset "
                    f"'{asset_name}', Naive Persistence, method '{metodo}'"
                )
                result = load_result_checkpoint(
                    checkpoint_dir, combination_id, run_fingerprint
                )
                if result is None:
                    result = evaluate_naive_asset(
                        asset_name,
                        series_dict[asset_name],
                        metodo,
                        partition_plan=partition_cache[asset_name][metodo],
                        carbon_config=carbon_config,
                    )
                    save_result_checkpoint(
                        checkpoint_dir, combination_id,
                        run_fingerprint, result
                    )
                else:
                    print("[INFO] Restored completed Naive result from checkpoint")
                naive_by_method[metodo] = result
                global_results_by_asset[asset_name][
                    "Naive Persistence"
                ].append(result)

        for model_name in neural_models:
            for metodo in methods_to_analyze:
                partition_plan = partition_cache[asset_name][metodo]
                if partition_plan is None:
                    raise RuntimeError(
                        f"Unavailable partition for {asset_name}/{metodo}."
                    )
                for run_seed in run_seeds:
                    current_comb += 1
                    combination_id = combination_identifier(
                        asset_name, model_name, metodo, run_seed
                    )
                    print(
                        f"[INFO] Combination {current_comb}/{total_comb}: "
                        f"Asset '{asset_name}', Model '{model_name}', "
                        f"method '{metodo}', seed {run_seed}"
                    )
                    result = load_result_checkpoint(
                        checkpoint_dir, combination_id, run_fingerprint
                    )
                    restored = result is not None
                    if restored:
                        print("[INFO] Restored completed combination from checkpoint")
                    else:
                        set_reproducible_seed(run_seed)
                        result = evaluate_model_asset(
                            model_name,
                            modelos[model_name],
                            asset_name,
                            series_dict[asset_name],
                            metodo,
                            EPOCHS_DICT[model_name],
                            partition_plan=partition_plan,
                            epochs_override=epochs_override,
                            batch_size=batch_size,
                            early_stopping_patience=early_stopping_patience,
                            track_emissions=track_emissions,
                            carbon_config=carbon_config,
                            run_seed=run_seed,
                            checkpoint_dir=checkpoint_dir,
                            run_fingerprint=run_fingerprint,
                        )

                    if include_naive:
                        naive = naive_by_method[metodo]
                        if not (
                            np.array_equal(result["idx_test"], naive["idx_test"])
                            and np.array_equal(
                                result["test_dates"], naive["test_dates"]
                            )
                        ):
                            raise RuntimeError(
                                f"Naive and neural test targets differ for "
                                f"{asset_name}/{metodo}."
                            )
                        for metric in ("mse", "rmse", "mae", "mape"):
                            result[f"naive_{metric}"] = float(naive[metric])
                        result["mse_skill_vs_naive"] = (
                            1.0 - float(result["mse"]) / float(naive["mse"])
                            if float(naive["mse"]) != 0.0 else np.nan
                        )
                        result["mae_skill_vs_naive"] = (
                            1.0 - float(result["mae"]) / float(naive["mae"])
                            if float(naive["mae"]) != 0.0 else np.nan
                        )
                    if not restored:
                        save_result_checkpoint(
                            checkpoint_dir, combination_id,
                            run_fingerprint, result
                        )
                    global_results_by_asset[asset_name][model_name].append(result)
                    print(
                        f"[INFO] Combination {current_comb}/{total_comb} completed. "
                        f"Remaining: {total_comb - current_comb}.\n"
                    )

        write_individual_tables(asset_name, global_results_by_asset[asset_name])
        print("\n======================================================\n")

    validate_result_completeness(
        global_results_by_asset,
        series_to_analyze,
        models_to_analyze,
        methods_to_analyze,
        run_seeds,
    )
    return global_results_by_asset


def validate_result_completeness(
    global_results_by_asset,
    series_to_analyze,
    models_to_analyze,
    methods_to_analyze,
    run_seeds,
):
    """Refuse publication output when a result is missing, duplicated, or misaligned."""
    expected = set()
    for asset in series_to_analyze:
        for model in models_to_analyze:
            for method in methods_to_analyze:
                if model == "Naive Persistence":
                    expected.add((asset, model, method, 0))
                else:
                    for run_seed in run_seeds:
                        expected.add((asset, model, method, int(run_seed)))

    observed = [
        (asset, model, result["method"], int(result["seed"]))
        for asset, model_results in global_results_by_asset.items()
        for model, results in model_results.items()
        for result in results
    ]
    observed_set = set(observed)
    duplicates = sorted(
        key for key in observed_set if observed.count(key) != 1
    )
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    if missing or duplicates or unexpected:
        raise RuntimeError(
            "Result completeness check failed. "
            f"Missing={missing[:10]}, duplicates={duplicates[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    for asset in series_to_analyze:
        method_references = {}
        asset_results = [
            result
            for results in global_results_by_asset[asset].values()
            for result in results
        ]
        for result in asset_results:
            method = result["method"]
            indices = np.asarray(result["idx_test"], dtype=int)
            test_dates = np.asarray(result["test_dates"], dtype="datetime64[D]")
            predictions, targets = result["preds_list"][0]
            predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
            targets = np.asarray(targets, dtype=np.float64).reshape(-1)
            if not (
                len(indices) == len(test_dates) == len(predictions)
                == len(targets) == int(result["n_test"])
            ):
                raise RuntimeError(
                    f"Test-array length mismatch for {asset}/"
                    f"{result['model']}/{method}."
                )
            if len(np.unique(indices)) != len(indices):
                raise RuntimeError(
                    f"Duplicate pooled test target for {asset}/"
                    f"{result['model']}/{method}."
                )
            if not np.all(np.diff(indices) > 0):
                raise RuntimeError(
                    f"Pooled test targets are not strictly chronological for "
                    f"{asset}/{result['model']}/{method}."
                )
            if not (
                np.all(np.isfinite(predictions))
                and np.all(np.isfinite(targets))
            ):
                raise RuntimeError(
                    f"Non-finite test value for {asset}/"
                    f"{result['model']}/{method}."
                )
            for metric in (
                "mse", "rmse", "mae", "mape", "mse_scaled",
                "rmse_scaled", "mae_scaled", "r2", "nse", "kge",
            ):
                if result[metric] is None or not np.isfinite(result[metric]):
                    raise RuntimeError(
                        f"Non-finite {metric} for {asset}/"
                        f"{result['model']}/{method}."
                    )
            recomputed_mse = float(mean_squared_error(targets, predictions))
            if not np.isclose(
                recomputed_mse, float(result["mse"]),
                rtol=1e-12, atol=1e-9
            ):
                raise RuntimeError(
                    f"Pooled MSE mismatch for {asset}/"
                    f"{result['model']}/{method}."
                )

            if result["model"] in NEURAL_MODELS:
                expected_stages = int(result["fold_count"])
                if result["time"] is None or float(result["time"]) <= 0.0:
                    raise RuntimeError(
                        f"Invalid total fitting time for {asset}/"
                        f"{result['model']}/{method}."
                    )
                if len(result["fold_best_epochs"]) != expected_stages:
                    raise RuntimeError(
                        f"Missing fold early-stopping epochs for {asset}/"
                        f"{result['model']}/{method}."
                    )
                if result["carbon_metadata"].get("tracking_enabled"):
                    if (
                        result["emissions"] is None
                        or not np.isfinite(result["emissions"])
                        or len(result.get("carbon_run_ids", []))
                        != expected_stages
                    ):
                        raise RuntimeError(
                            f"Incomplete CodeCarbon stages for {asset}/"
                            f"{result['model']}/{method}."
                        )
            else:
                if float(result["time"]) != 0.0 or result["emissions"] is not None:
                    raise RuntimeError(
                        f"Naive training time/emissions boundary is inconsistent "
                        f"for {asset}/{method}."
                    )

            reference = method_references.get(method)
            identity = (indices, test_dates, targets)
            if reference is None:
                method_references[method] = identity
            elif not all(
                np.array_equal(first, second)
                for first, second in zip(reference, identity)
            ):
                raise RuntimeError(
                    f"Within-method test identity check failed for "
                    f"{asset}/{method}."
                )

    print(
        f"[INFO] Completeness check passed: {len(expected)} unique results "
        "with method-specific test identity preserved"
    )
    return {
        "expected_results": len(expected),
        "observed_results": len(observed),
        "complete": True,
    }


def build_results_by_model(global_results_by_asset):
    global_results_by_model = {}
    for asset, models_results in global_results_by_asset.items():
        for model, res_list in models_results.items():
            if model not in global_results_by_model:
                global_results_by_model[model] = {}
            global_results_by_model[model][asset] = res_list
    return global_results_by_model


def write_global_visualizations(
    global_results_by_model,
    show_plots=False,
    output_dir="paper_figures",
):
    """Create a compact, non-redundant set of publication figures."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model_name, asset_results in global_results_by_model.items():
        for asset_name, results in asset_results.items():
            for result in results:
                rows.append({
                    "Model": model_name,
                    "Asset": asset_name,
                    "Method": METODOS_MAP[result["method"]],
                    "method_key": result["method"],
                    "Seed": result["seed"],
                    "Scaled RMSE": result["rmse_scaled"],
                    "MAPE (%)": result["mape"],
                    "R2": result["r2"],
                    "Fitting time (s)": result["time"],
                    "Emissions (kgCO2eq)": result["emissions"],
                })
    if not rows:
        print("[INFO] No results are available for publication figures.")
        return
    frame = pd.DataFrame(rows)
    for numeric_column in (
        "Scaled RMSE", "MAPE (%)", "R2", "Fitting time (s)",
        "Emissions (kgCO2eq)",
    ):
        frame[numeric_column] = pd.to_numeric(
            frame[numeric_column], errors="coerce"
        )
    sns.set_theme(style="whitegrid", context="paper")
    generated_files = [
        os.path.join(output_dir, filename)
        for filename in (
            "temporal_partition_protocol.png",
            "temporal_partition_protocol.pdf",
            "temporal_partition_protocol.json",
        )
        if os.path.exists(os.path.join(output_dir, filename))
    ]

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    sns.boxplot(
        data=frame,
        x="Model",
        y="Scaled RMSE",
        color="#8ecae6",
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="Model",
        y="Scaled RMSE",
        color="#1d3557",
        size=2.2,
        alpha=0.32,
        ax=axis,
    )
    axis.set_title("Pooled method-specific test error by forecasting model")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"model_scaled_rmse.{extension}")
        fig.savefig(path, dpi=300 if extension == "png" else None)
        generated_files.append(path)
    finish_plot(fig, show_plots)

    neural_frame = frame[
        frame["Model"].isin(NEURAL_MODELS)
        & frame["method_key"].isin(METODOS_PARTICION)
    ].copy()
    if not neural_frame.empty:
        method_order = [
            METODOS_MAP[method] for method in METODOS_PARTICION
            if method in set(neural_frame["method_key"])
        ]
        fig, axis = plt.subplots(figsize=(11.5, 5.6))
        sns.boxplot(
            data=neural_frame,
            x="Method",
            y="Scaled RMSE",
            order=method_order,
            color="#ffbf69",
            showfliers=False,
            ax=axis,
        )
        sns.stripplot(
            data=neural_frame,
            x="Method",
            y="Scaled RMSE",
            order=method_order,
            color="#6d3b00",
            size=2.0,
            alpha=0.25,
            ax=axis,
        )
        axis.set_title("Pooled test error by temporal partitioning method")
        axis.tick_params(axis="x", rotation=35)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = os.path.join(
                output_dir, f"partition_scaled_rmse.{extension}"
            )
            fig.savefig(path, dpi=300 if extension == "png" else None)
            generated_files.append(path)
        finish_plot(fig, show_plots)

    cost_frame = (
        frame.groupby("Model", as_index=False)[
            ["Fitting time (s)", "Emissions (kgCO2eq)"]
        ]
        .mean(numeric_only=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.barplot(
        data=cost_frame,
        x="Model",
        y="Fitting time (s)",
        color="#90be6d",
        ax=axes[0],
    )
    axes[0].set_title("Mean total fitting time")
    axes[0].tick_params(axis="x", rotation=35)
    emissions_frame = cost_frame.dropna(subset=["Emissions (kgCO2eq)"])
    if emissions_frame.empty:
        axes[1].text(
            0.5, 0.5, "Emissions tracking disabled", ha="center", va="center"
        )
        axes[1].set_axis_off()
    else:
        sns.barplot(
            data=emissions_frame,
            x="Model",
            y="Emissions (kgCO2eq)",
            color="#f9844a",
            ax=axes[1],
        )
        axes[1].set_title("Mean total fitting emissions")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.7f"))
    fig.suptitle("Complete computational cost: all method-specific fold fits")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"computational_cost.{extension}")
        fig.savefig(path, dpi=300 if extension == "png" else None)
        generated_files.append(path)
    finish_plot(fig, show_plots)

    atomic_write_json(os.path.join(output_dir, "figure_manifest.json"), {
        "generated_files": generated_files,
        "design": (
            "Compact paper set; redundant per-model global line, violin, box, "
            "and strip plot families are intentionally not generated."
        ),
        "cross_asset_error_metric": "RMSE on development-scaled values",
        "cost_boundary": CARBON_ACCOUNTING_BOUNDARY,
    })
    print(f"[INFO] Publication figures saved in '{output_dir}'")


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit='D')
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def serialize_for_csv(value):
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit='D')
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_dataframe_csv_atomic(frame, output_filename):
    absolute_path = os.path.abspath(output_filename)
    parent = os.path.dirname(absolute_path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp_table_", suffix=".csv", dir=parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        os.replace(temporary_path, absolute_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def save_prediction_artifacts(
    global_results_by_asset,
    output_dir="predictions",
):
    """Export pooled method-specific test arrays separately from metadata."""
    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    for asset, model_results in global_results_by_asset.items():
        for model, results in model_results.items():
            for result in results:
                combination_id = combination_identifier(
                    asset,
                    model,
                    result["method"],
                    result["seed"],
                )
                filename = f"{combination_id}.npz"
                path = os.path.join(output_dir, filename)
                predictions, targets = result["preds_list"][0]
                atomic_save_npz(
                    path,
                    predictions=np.asarray(predictions, dtype=np.float64),
                    targets=np.asarray(targets, dtype=np.float64),
                    raw_target_indices=np.asarray(
                        result["idx_test"], dtype=np.int64
                    ),
                    test_dates=np.asarray(
                        result["test_dates"], dtype="datetime64[D]"
                    ),
                )
                result["prediction_file"] = os.path.abspath(path)
                index_rows.append({
                    "asset": asset,
                    "model": model,
                    "method": result["method"],
                    "seed": result["seed"],
                    "prediction_file": os.path.abspath(path),
                    "sha256": file_sha256(path),
                    "n_test": int(len(targets)),
                    "test_range": result["test_range"],
                })
    index_frame = pd.DataFrame(index_rows)
    index_path = os.path.join(output_dir, "prediction_index.csv")
    save_dataframe_csv_atomic(index_frame, index_path)
    print(f"[INFO] Pooled test prediction arrays saved in '{output_dir}'")
    return index_frame


def save_global_results_csv(global_results_by_asset, output_filename="global_results.csv"):
    all_results_flat = []
    for _, models_results in global_results_by_asset.items():
        for _, res_list in models_results.items():
            for result in res_list:
                row = {
                    key: serialize_for_csv(value)
                    for key, value in result.items()
                    if key not in {"preds_list", "idx_test", "test_dates"}
                }
                all_results_flat.append(row)

    if all_results_flat:
        df_global = pd.DataFrame(all_results_flat)
        ordered_columns = [col for col in GLOBAL_CSV_COLUMNS if col in df_global.columns]
        extra_columns = [col for col in df_global.columns if col not in ordered_columns]
        df_global = df_global[ordered_columns + extra_columns]
    else:
        df_global = pd.DataFrame(columns=GLOBAL_CSV_COLUMNS)

    save_dataframe_csv_atomic(df_global, output_filename)
    print(f"[INFO] Global CSV saved as '{output_filename}'")
    return df_global


def save_metric_summary_tables(df_global, output_dir="metric_tables"):
    """Export reviewer-ready per-asset and cross-asset metric summaries."""
    required = {"asset", "model", "method", "seed"}
    missing = sorted(required.difference(df_global.columns))
    if missing:
        raise ValueError("Metric summaries are missing columns: " + ", ".join(missing))

    metrics = [
        name for name in (
            "mse", "rmse", "mae", "mape", "mse_scaled", "rmse_scaled",
            "mae_scaled", "r2", "nse", "kge", "time", "emissions",
            "mse_skill_vs_naive", "mae_skill_vs_naive",
        )
        if name in df_global.columns
    ]
    os.makedirs(output_dir, exist_ok=True)

    def aggregate_mean_std(frame, grouping):
        grouped = frame.groupby(grouping, dropna=False)
        summary = grouped[metrics].agg(["mean", "std"]).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple) else str(column)
            for column in summary.columns
        ]
        repetitions = grouped["seed"].nunique().reset_index(name="n_repetitions")
        return summary.merge(repetitions, on=grouping, how="left")

    per_asset = aggregate_mean_std(
        df_global,
        ["asset", "model", "method"],
    )
    per_asset.insert(
        per_asset.columns.get_loc("method") + 1,
        "method_label",
        per_asset["method"].map(METODOS_MAP),
    )
    per_asset_path = os.path.join(output_dir, "per_asset_metrics.csv")
    save_dataframe_csv_atomic(per_asset, per_asset_path)

    # First average repeated seeds within each asset/configuration. Each asset
    # then contributes exactly once to the unweighted cross-asset mean.
    seed_means = (
        df_global
        .groupby(["asset", "model", "method"], as_index=False)[metrics]
        .mean()
    )
    cross_grouped = seed_means.groupby(["model", "method"], dropna=False)
    cross_asset = cross_grouped[metrics].agg(["mean", "std"]).reset_index()
    cross_asset.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple) else str(column)
        for column in cross_asset.columns
    ]
    asset_counts = cross_grouped["asset"].nunique().reset_index(name="n_assets")
    cross_asset = cross_asset.merge(asset_counts, on=["model", "method"], how="left")
    cross_asset.insert(
        cross_asset.columns.get_loc("method") + 1,
        "method_label",
        cross_asset["method"].map(METODOS_MAP),
    )
    cross_asset_path = os.path.join(output_dir, "cross_asset_metrics.csv")
    save_dataframe_csv_atomic(cross_asset, cross_asset_path)

    protocol_columns = [
        column for column in (
            "model", "architecture_protocol", "architecture_config", "optimizer",
            "learning_rate", "training_loss", "batch_size",
            "early_stopping_patience", "trainable_parameters",
            "hyperparameter_provenance", "hyperparameters_validated",
            "hyperparameter_validation_record",
        )
        if column in df_global.columns
    ]
    architecture_protocol = (
        df_global[protocol_columns]
        .drop_duplicates()
        .sort_values(["model"])
        .reset_index(drop=True)
    )
    architecture_path = os.path.join(output_dir, "architecture_protocol.csv")
    save_dataframe_csv_atomic(architecture_protocol, architecture_path)

    metric_decimals = {
        "mse": 2,
        "rmse": 2,
        "mae": 2,
        "mape": 2,
        "mse_scaled": 5,
        "rmse_scaled": 5,
        "mae_scaled": 5,
        "r2": 4,
        "nse": 4,
        "kge": 4,
        "time": 2,
        "emissions": 7,
        "mse_skill_vs_naive": 4,
        "mae_skill_vs_naive": 4,
    }

    def publication_format(summary, identity_columns, count_column):
        publication = summary[identity_columns + [count_column]].copy()
        for metric in metrics:
            mean_column = f"{metric}_mean"
            std_column = f"{metric}_std"
            decimals = metric_decimals[metric]

            def render(row):
                mean_value = row[mean_column]
                std_value = row[std_column]
                if pd.isna(mean_value):
                    return "N/A"
                mean_text = f"{mean_value:.{decimals}f}"
                if pd.isna(std_value):
                    return mean_text
                return f"{mean_text} ± {std_value:.{decimals}f}"

            publication[metric] = summary.apply(render, axis=1)
        return publication

    paper_per_asset = publication_format(
        per_asset,
        ["asset", "model", "method", "method_label"],
        "n_repetitions",
    )
    paper_cross_asset = publication_format(
        cross_asset,
        ["model", "method", "method_label"],
        "n_assets",
    )
    paper_per_asset_path = os.path.join(
        output_dir, "paper_per_asset_mean_sd.csv"
    )
    paper_cross_asset_path = os.path.join(
        output_dir, "paper_cross_asset_mean_sd.csv"
    )
    save_dataframe_csv_atomic(paper_per_asset, paper_per_asset_path)
    save_dataframe_csv_atomic(paper_cross_asset, paper_cross_asset_path)

    metadata = {
        "per_asset_file": os.path.abspath(per_asset_path),
        "cross_asset_file": os.path.abspath(cross_asset_path),
        "architecture_protocol_file": os.path.abspath(architecture_path),
        "paper_per_asset_file": os.path.abspath(paper_per_asset_path),
        "paper_cross_asset_file": os.path.abspath(paper_cross_asset_path),
        "publication_number_format": (
            "mean ± sample standard deviation; a single deterministic Naive "
            "run is shown without a standard deviation"
        ),
        "per_asset_aggregation": (
            "Arithmetic mean and sample standard deviation across independent seeds "
            "for each asset/model/partition configuration."
        ),
        "cross_asset_aggregation": (
            "Unweighted arithmetic mean and sample standard deviation across assets "
            "after averaging seeds within each asset/model/partition configuration."
        ),
        "error_units": {
            "mse": "USD^2",
            "rmse": "USD",
            "mae": "USD",
            "mape": "percent",
            "mse_scaled": "squared training-range-scaled units",
            "rmse_scaled": "training-range-scaled units",
            "mae_scaled": "training-range-scaled units",
            "r2_nse_kge": "dimensionless",
        },
        "cross_asset_comparability_note": (
            "Use rmse_scaled/mae_scaled or within-asset ranks for comparisons across "
            "assets with different price levels; use skill versus the same-method "
            "Naive baseline when test horizons differ by partition protocol; raw "
            "USD errors remain asset-scale dependent."
        ),
        "method_specific_test_note": (
            "Partition methods use the test blocks declared by their own protocol. "
            "A partition comparison therefore compares complete forecasting "
            "protocols, not predictions on one identical target set."
        ),
    }
    atomic_write_json(
        os.path.join(output_dir, "metric_aggregation_metadata.json"),
        metadata,
    )

    print(f"[INFO] Metric summary files saved in '{output_dir}'")
    save_configuration_rank_tables(df_global, output_dir=output_dir)
    return per_asset, cross_asset


def save_configuration_rank_tables(df_global, output_dir="metric_tables"):
    """Regenerate the paper's 5-by-8 configuration-rank tables."""
    metrics = {
        "mse": "lower",
        "rmse": "lower",
        "mae": "lower",
        "mape": "lower",
        "r2": "higher",
        "nse": "higher",
        "kge": "higher",
    }
    neural = df_global[df_global["model"].isin(NEURAL_MODELS)].copy()
    if neural.empty:
        print("[INFO] No neural results available for configuration-rank tables.")
        return {}
    seed_means = (
        neural
        .groupby(["asset", "model", "method"], as_index=False)[list(metrics)]
        .mean()
    )
    rank_frames = {}
    long_rows = []
    for metric, direction in metrics.items():
        ranked_assets = []
        for asset, asset_frame in seed_means.groupby("asset", sort=True):
            ranked = asset_frame[["asset", "model", "method", metric]].copy()
            ranked["rank"] = ranked[metric].rank(
                method="average",
                ascending=(direction == "lower"),
            )
            ranked_assets.append(ranked)
            for row in ranked.itertuples(index=False):
                long_rows.append({
                    "asset": row.asset,
                    "model": row.model,
                    "method": row.method,
                    "method_label": METODOS_MAP[row.method],
                    "metric": metric,
                    "direction": direction,
                    "value_after_seed_mean": getattr(row, metric),
                    "within_asset_configuration_rank": row.rank,
                })
        ranked_all = pd.concat(ranked_assets, ignore_index=True)
        average_ranks = (
            ranked_all
            .groupby(["method", "model"], as_index=False)["rank"]
            .mean()
        )
        pivot = average_ranks.pivot(
            index="method", columns="model", values="rank"
        ).reindex(index=[
            method for method in METODOS_PARTICION
            if method in set(average_ranks["method"])
        ])
        model_order = [
            model for model in AVAILABLE_MODELS
            if model in NEURAL_MODELS and model in pivot.columns
        ]
        pivot = pivot.reindex(columns=model_order)
        pivot["Mean rank across models"] = pivot.mean(axis=1)
        model_means = pivot[model_order].mean(axis=0)
        model_means["Mean rank across models"] = float(
            pivot["Mean rank across models"].mean()
        )
        pivot.loc["__model_mean__"] = model_means
        table = pivot.reset_index().rename(columns={"method": "Method"})
        table["Method"] = table["Method"].map(
            lambda value: (
                "Mean rank across methods"
                if value == "__model_mean__" else METODOS_MAP[value]
            )
        )
        numeric_columns = [
            column for column in table.columns if column != "Method"
        ]
        table[numeric_columns] = table[numeric_columns].round(2)
        path = os.path.join(
            output_dir, f"configuration_average_ranks_{metric}.csv"
        )
        save_dataframe_csv_atomic(table, path)
        rank_frames[metric] = ranked_all

    if "mse" in rank_frames and "rmse" in rank_frames:
        if not np.array_equal(
            rank_frames["mse"]["rank"].to_numpy(),
            rank_frames["rmse"]["rank"].to_numpy(),
        ):
            raise RuntimeError("MSE and RMSE configuration ranks must be identical.")
    if "r2" in rank_frames and "nse" in rank_frames:
        if not np.array_equal(
            rank_frames["r2"]["rank"].to_numpy(),
            rank_frames["nse"]["rank"].to_numpy(),
        ):
            raise RuntimeError("R2 and NSE configuration ranks must be identical.")
    save_dataframe_csv_atomic(
        pd.DataFrame(long_rows),
        os.path.join(output_dir, "configuration_ranks_long.csv"),
    )
    atomic_write_json(
        os.path.join(output_dir, "configuration_rank_metadata.json"),
        {
            "seed_handling": "mean within asset/model/method before ranking",
            "ranking_block": "asset",
            "configuration_count_full_run": (
                len(NEURAL_MODELS) * len(METODOS_PARTICION)
            ),
            "rank_1_is_best": True,
            "mse_rmse_rank_identity_checked": True,
            "r2_nse_rank_identity_checked": True,
            "naive_persistence_excluded_reason": (
                "partition choice cannot alter a model fit for Naive Persistence "
                "because the baseline has no fitted parameters; it remains reported "
                "for every asset/method as a target-matched reference and is "
                "tested separately as a horizon-difficulty diagnostic"
            ),
            "method_specific_test_interpretation": (
                "ranks compare complete partition protocols; their test dates "
                "are method-specific rather than one shared horizon"
            ),
        },
    )
    print(f"[INFO] Configuration-rank tables saved in '{output_dir}'")
    return rank_frames


def paired_rank_biserial(first, second):
    """Matched-pairs rank-biserial effect size for first minus second."""
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    differences = differences[np.isfinite(differences) & (differences != 0.0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    return (positive - negative) / (positive + negative)


def holm_adjust(raw_p_values, alpha=0.05):
    """Holm step-down adjusted p-values without an extra dependency."""
    p_values = np.asarray(raw_p_values, dtype=np.float64)
    if p_values.ndim != 1 or p_values.size == 0:
        raise ValueError("Holm correction requires a non-empty one-dimensional p-value array.")
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("Holm correction received an invalid p-value.")

    order = np.argsort(p_values)
    ordered = p_values[order]
    adjusted_ordered = np.empty_like(ordered)
    running_max = 0.0
    total = len(ordered)
    for position, p_value in enumerate(ordered):
        candidate = (total - position) * p_value
        running_max = max(running_max, candidate)
        adjusted_ordered[position] = min(1.0, running_max)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted <= alpha, adjusted


def analyze_friedman_family(
    frame,
    block_column,
    treatment_column,
    family,
    metric_directions,
    alpha=0.05,
):
    omnibus_rows = []
    posthoc_rows = []
    rank_rows = []

    for metric, direction in metric_directions.items():
        matrix = frame.pivot(index=block_column, columns=treatment_column, values=metric)
        matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            print(
                f"[WARNING] Skipping {family}/{metric}: Friedman requires at least "
                "three complete blocks and three treatments."
            )
            continue

        treatments = list(matrix.columns)
        samples = [matrix[name].to_numpy(dtype=np.float64) for name in treatments]
        try:
            statistic, p_value = friedmanchisquare(*samples)
        except ValueError as error:
            print(f"[WARNING] Skipping {family}/{metric}: {error}")
            continue

        omnibus_rows.append({
            "family": family,
            "metric": metric,
            "n_blocks": int(matrix.shape[0]),
            "n_treatments": int(matrix.shape[1]),
            "friedman_chi_square": float(statistic),
            "p_value": float(p_value),
            "alpha": float(alpha),
            "significant": bool(p_value < alpha),
        })

        rank_matrix = matrix.rank(
            axis=1,
            method="average",
            ascending=(direction == "lower"),
        )
        for treatment, average_rank in rank_matrix.mean(axis=0).items():
            rank_rows.append({
                "family": family,
                "metric": metric,
                "treatment": treatment,
                "average_rank": float(average_rank),
                "rank_1_is_best": True,
                "n_blocks": int(matrix.shape[0]),
            })

        metric_pairs = []
        raw_p_values = []
        for first_name, second_name in itertools.combinations(treatments, 2):
            first = matrix[first_name].to_numpy(dtype=np.float64)
            second = matrix[second_name].to_numpy(dtype=np.float64)
            if np.allclose(first, second, rtol=0.0, atol=0.0):
                wilcoxon_statistic, pair_p_value = 0.0, 1.0
            else:
                try:
                    test = wilcoxon(
                        first,
                        second,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                    wilcoxon_statistic = float(test.statistic)
                    pair_p_value = float(test.pvalue)
                except ValueError:
                    wilcoxon_statistic, pair_p_value = 0.0, 1.0
            metric_pairs.append({
                "family": family,
                "metric": metric,
                "first": first_name,
                "second": second_name,
                "n_blocks": int(matrix.shape[0]),
                "wilcoxon_statistic": wilcoxon_statistic,
                "p_value_raw": pair_p_value,
                "rank_biserial_first_minus_second": float(
                    paired_rank_biserial(first, second)
                ),
                "better_direction": direction,
                "omnibus_p_value": float(p_value),
                "omnibus_significant": bool(p_value < alpha),
            })
            raw_p_values.append(pair_p_value)

        rejected, adjusted = holm_adjust(raw_p_values, alpha=alpha)
        for row, reject, adjusted_p in zip(metric_pairs, rejected, adjusted):
            row["p_value_holm"] = float(adjusted_p)
            row["significant_holm"] = bool(reject)
            row["significant_after_omnibus_gate"] = bool(
                reject and row["omnibus_significant"]
            )
            row["alpha"] = float(alpha)
            posthoc_rows.append(row)

    return omnibus_rows, posthoc_rows, rank_rows


def run_statistical_analysis(df_global, output_dir="statistical_tests", alpha=0.05):
    """Friedman omnibus tests and paired Wilcoxon-Holm post-hoc tests."""
    required = {
        "asset", "model", "method", "seed",
        "mse_scaled", "rmse_scaled", "mae_scaled",
        "mape", "r2", "nse", "kge",
        "mse_skill_vs_naive", "mae_skill_vs_naive",
    }
    missing = sorted(required.difference(df_global.columns))
    if missing:
        raise ValueError(
            "Statistical analysis is missing columns: "
            + ", ".join(missing)
        )

    os.makedirs(output_dir, exist_ok=True)

    # Cross-asset inferential comparisons use training-range-normalized
    # error metrics because cryptocurrency prices differ substantially
    # in scale. MSE and RMSE are both retained for reporting completeness,
    # even though they produce identical within-block rankings. NSE is also
    # retained, although it is algebraically identical to R² in this
    # unweighted, single-output implementation.
    metrics = {
        "mse_scaled": "lower",
        "rmse_scaled": "lower",
        "mae_scaled": "lower",
        "mape": "lower",
        "r2": "higher",
        "nse": "higher",
        "kge": "higher",
        "mse_skill_vs_naive": "higher",
        "mae_skill_vs_naive": "higher",
    }

    # Seeds are repeated measurements, not independent experimental units.
    # Average them before treating cryptocurrency assets as Friedman blocks.
    seed_aggregated = (
        df_global
        .groupby(["asset", "model", "method"], as_index=False)[list(metrics)]
        .mean()
    )

    architecture_frame = (
        seed_aggregated
        .groupby(["asset", "model"], as_index=False)[list(metrics)]
        .mean()
    )
    partition_frame = (
        seed_aggregated[seed_aggregated["model"].isin(NEURAL_MODELS)]
        .groupby(["asset", "method"], as_index=False)[list(metrics)]
        .mean()
    )
    partition_frame["method"] = partition_frame["method"].map(METODOS_MAP)
    naive_horizon_frame = seed_aggregated[
        seed_aggregated["model"] == "Naive Persistence"
    ].copy()
    if not naive_horizon_frame.empty:
        naive_horizon_frame["method"] = naive_horizon_frame["method"].map(
            METODOS_MAP
        )

    all_omnibus = []
    all_posthoc = []
    all_ranks = []
    for family_frame, block, treatment, family in (
        (architecture_frame, "asset", "model", "forecasting_model"),
        (partition_frame, "asset", "method", "partition_method"),
    ):
        omnibus, posthoc, ranks = analyze_friedman_family(
            family_frame,
            block,
            treatment,
            family,
            metrics,
            alpha=alpha,
        )
        all_omnibus.extend(omnibus)
        all_posthoc.extend(posthoc)
        all_ranks.extend(ranks)

    # Naive is not mixed into the trained-model partition family. It receives
    # its own diagnostic comparison: variation here reflects differing test
    # horizon difficulty, not a training benefit from the partition method.
    if not naive_horizon_frame.empty:
        naive_metrics = {
            metric: direction for metric, direction in metrics.items()
            if not metric.endswith("_skill_vs_naive")
        }
        omnibus, posthoc, ranks = analyze_friedman_family(
            naive_horizon_frame,
            "asset",
            "method",
            "naive_test_horizon_difficulty",
            naive_metrics,
            alpha=alpha,
        )
        all_omnibus.extend(omnibus)
        all_posthoc.extend(posthoc)
        all_ranks.extend(ranks)

    omnibus_df = pd.DataFrame(all_omnibus)
    posthoc_df = pd.DataFrame(all_posthoc)
    ranks_df = pd.DataFrame(all_ranks)
    save_dataframe_csv_atomic(
        omnibus_df, os.path.join(output_dir, "friedman_omnibus.csv")
    )
    save_dataframe_csv_atomic(
        posthoc_df, os.path.join(output_dir, "wilcoxon_holm_posthoc.csv")
    )
    save_dataframe_csv_atomic(
        ranks_df, os.path.join(output_dir, "average_ranks.csv")
    )

    metadata = {
        "alpha": alpha,
        "independent_block": "cryptocurrency asset",
        "seed_aggregation": "arithmetic mean before inferential testing",
        "model_aggregation": "mean across partitioning methods within each asset",
        "partition_aggregation": "mean across the five neural architectures within each asset",
        "naive_in_partition_family": (
            "not mixed into the trained-model partition family because it has "
            "no fitted parameters. It is evaluated on every asset/method test "
            "set, used for target-matched skill, and analyzed separately in the "
            "naive_test_horizon_difficulty family"
        ),
        "method_specific_test_interpretation": (
            "Partition-method tests compare complete resampling/forecasting "
            "protocols whose declared test dates may differ. They are not paired "
            "prediction-level comparisons on one common horizon. Skill relative "
            "to the same-method Naive baseline is therefore reported alongside "
            "scaled and rank-based measures."
        ),
        "omnibus_test": "Friedman",
        "posthoc_test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity_correction": "Holm within each metric and comparison family",
        "posthoc_interpretation": (
            "A pairwise result is interpreted as confirmatory only when both the "
            "corresponding Friedman omnibus test and Holm-adjusted comparison are significant."
        ),
        "effect_size": "matched-pairs rank-biserial correlation (first minus second)",
        "inferential_error_scale": (
            "MSE, RMSE, and MAE are tested on the training-range-normalized "
            "scale to prevent cryptocurrency price levels from determining "
            "the cross-asset comparisons."
        ),
        "retained_metric_redundancy": {
            "mse_vs_rmse": (
                "MSE is the square of RMSE for a fixed test sample. Therefore, "
                "both metrics produce identical within-asset rankings and "
                "identical Friedman average ranks. Both are retained for "
                "reporting completeness."
            ),
            "nse_vs_r2": (
                "In this unweighted, single-output regression implementation, "
                "standard NSE and R² are algebraically identical because both "
                "equal one minus the residual sum of squares divided by the "
                "total sum of squares. Consequently, their values, rankings, "
                "Friedman statistics, pairwise Wilcoxon results, and effect "
                "sizes are expected to be identical. Both are retained to "
                "match the manuscript's declared evaluation metrics."
            ),
        },
    }
    atomic_write_json(
        os.path.join(output_dir, "statistical_analysis_metadata.json"),
        metadata,
    )

    print(f"[INFO] Statistical comparison files saved in '{output_dir}'")
    return omnibus_df, posthoc_df, ranks_df


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate time-series partitioning experiments.")
    parser.add_argument(
        "--series",
        default=None,
        help="Comma-separated assets to analyze, or 'all'. If omitted, the script prompts interactively."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated models to analyze, or 'all'. If omitted, the script prompts interactively."
    )
    parser.add_argument(
        "--architecture-protocol",
        choices=["individually_tuned", "controlled"],
        default="individually_tuned",
        help=(
            "Use independently validated asset/model configurations for the "
            "primary experiment. 'controlled' retains the matched-parameter "
            "architecture as a separately labelled sensitivity analysis."
        )
    )
    parser.add_argument(
        "--validated-hyperparameters",
        default=None,
        help=(
            "Completed JSON manifest produced by --tune-hyperparameters. "
            "Required for a confirmatory individually_tuned neural run."
        ),
    )
    parser.add_argument(
        "--evaluation-start",
        default=None,
        help=(
            "First experimental date (YYYY-MM-DD). It must be later than the "
            "calibration cutoff when using a prefix-validation manifest."
        ),
    )
    parser.add_argument(
        "--allow-unvalidated-hyperparameters",
        action="store_true",
        help=(
            "Permit expert/supplied candidates without a completed validation "
            "manifest. Exploratory only; never label such a run as optimized."
        ),
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help=(
            "Run the separate temporal calibration stage and exit. Every selected "
            "asset/neural-model pair is optimized with identical search rigor."
        ),
    )
    parser.add_argument(
        "--tuning-calibration-end",
        default=None,
        help=(
            "Inclusive final date (YYYY-MM-DD) of the calibration-only prefix; "
            "required with --tune-hyperparameters."
        ),
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=12,
        help="Deterministic local-search candidates per asset/model pair (default: 12).",
    )
    parser.add_argument(
        "--tuning-seeds",
        default="17,29,43",
        help="Independent seeds used for every tuning candidate (default: 17,29,43).",
    )
    parser.add_argument(
        "--tuning-output",
        default="validated_hyperparameters.json",
        help=(
            "Atomic, resumable tuning manifest and final selected configurations."
        ),
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated partition method keys to analyze, or 'all'."
    )
    parser.add_argument(
        "--seeds",
        default="123,456,789,101112,131415",
        help=(
            "Comma-separated unique integer seeds. Repetitions are averaged "
            "before treating cryptocurrency assets as independent blocks."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for every model. Useful for faster test runs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Keras batch size (default: 32)."
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early-stopping patience on validation loss (default: 15 epochs)."
    )
    parser.add_argument(
        "--no-emissions",
        action="store_true",
        help="Skip CodeCarbon emissions tracking to reduce per-run overhead."
    )
    parser.add_argument(
        "--carbon-country",
        default=os.environ.get("CODECARBON_COUNTRY_ISO_CODE"),
        help=(
            "Real three-letter ISO country code of the compute "
            "location, for example USA, NLD, or ESP."
        )
    )
    parser.add_argument(
        "--carbon-region",
        default=os.environ.get("CODECARBON_REGION"),
        help=(
            "Real state/province supported by CodeCarbon; "
            "omit it when it does not apply."
        )
    )
    parser.add_argument(
        "--carbon-tracking-mode",
        choices=["machine", "process"],
        default="machine",
        help=(
            "CodeCarbon attribution mode. Use 'machine' only "
            "on an otherwise idle dedicated instance."
        )
    )
    parser.add_argument(
        "--carbon-measure-power-secs",
        type=float,
        default=1.0,
        help=(
            "CodeCarbon hardware-power sampling interval "
            "in seconds."
        )
    )
    parser.add_argument(
        "--carbon-output-dir",
        default="codecarbon_results",
        help="Directory for CodeCarbon's detailed CSV output."
    )
    parser.add_argument(
        "--carbon-output-file",
        default="emissions_detailed.csv",
        help=(
            "Base filename for CodeCarbon CSV files; every fitting stage and "
            "retry receives a unique suffix."
        )
    )
    parser.add_argument(
        "--environment-output",
        default="run_environment.json",
        help=(
            "JSON file with the real hardware, software, "
            "and experimental configuration."
        )
    )
    parser.add_argument(
        "--software-output",
        default="software_environment.txt",
        help="Text file with the exact output of pip freeze."
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/yfinance",
        help="Directory for cached yfinance Close series. Use an empty value to disable caching."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Ignore cached market data and download it again."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set TensorFlow intra/inter op thread count before training."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Allow neural models to run without a visible GPU. Omit this for "
            "the paper run so the preflight fails instead of silently using CPU."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for quick runs: skip plots/emissions, use 25 epochs, batch size 128, and early stopping unless overridden."
    )
    parser.add_argument(
        "--skip-partition-plots",
        action="store_true",
        help="Skip the revised eight-method partition protocol diagram."
    )
    parser.add_argument(
        "--skip-global-plots",
        action="store_true",
        help="Skip the compact publication figure set."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving them. By default plots are closed after saving."
    )
    parser.add_argument(
        "--output-csv",
        default="global_results.csv",
        help="Path for the flattened global results CSV."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="experiment_checkpoints",
        help=(
            "Root directory for atomic fold/result checkpoints. Re-running the "
            "same command resumes the matching experiment automatically."
        ),
    )
    parser.add_argument(
        "--prediction-output-dir",
        default="predictions",
        help="Directory for pooled method-specific test arrays and their index.",
    )
    parser.add_argument(
        "--figure-output-dir",
        default="paper_figures",
        help="Directory for the revised diagram and compact paper figures.",
    )
    parser.add_argument(
        "--run-summary-output",
        default="execution_summary.json",
        help="JSON file containing wall time, fitting time, totals, and hashes.",
    )
    parser.add_argument(
        "--statistical-output-dir",
        default="statistical_tests",
        help="Directory for Friedman, Wilcoxon-Holm, rank, and metadata files."
    )
    parser.add_argument(
        "--metric-table-output-dir",
        default="metric_tables",
        help="Directory for per-asset, cross-asset, and aggregation-metadata tables."
    )
    parser.add_argument(
        "--statistical-alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level used by the inferential analysis."
    )
    return parser


def collect_artifact_hashes(paths):
    artifacts = []
    seen = set()
    for requested_path in paths:
        if not requested_path:
            continue
        absolute = os.path.abspath(requested_path)
        candidates = []
        if os.path.isfile(absolute):
            candidates = [absolute]
        elif os.path.isdir(absolute):
            for root, directories, filenames in os.walk(absolute):
                directories.sort()
                for filename in sorted(filenames):
                    candidates.append(os.path.join(root, filename))
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            artifacts.append({
                "path": candidate,
                "size_bytes": int(os.path.getsize(candidate)),
                "sha256": file_sha256(candidate),
            })
    return artifacts


def main(argv=None):
    global series_dict, dates_dict, ARCHITECTURE_PROTOCOL
    invocation_start = PROCESS_START_PERF_COUNTER
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ARCHITECTURE_PROTOCOL = args.architecture_protocol
    if args.tune_hyperparameters:
        if args.fast:
            parser.error(
                "--fast cannot be combined with --tune-hyperparameters; "
                "temporal validation must use the declared full protocol."
        )
        ARCHITECTURE_PROTOCOL = "individually_tuned"
        args.skip_partition_plots = True
        args.skip_global_plots = True
    if args.fast:
        args.skip_partition_plots = True
        args.skip_global_plots = True
        args.no_emissions = True
        args.seeds = "123"
        if args.epochs is None:
            args.epochs = 25
        args.batch_size = 128
        args.early_stopping_patience = 8
        args.allow_unvalidated_hyperparameters = True
    if ARCHITECTURE_PROTOCOL == "controlled":
        if args.epochs is not None:
            CONTROLLED_ARCHITECTURE_CONFIG["max_epochs"] = int(args.epochs)
        CONTROLLED_ARCHITECTURE_CONFIG["batch_size"] = int(args.batch_size)
        CONTROLLED_ARCHITECTURE_CONFIG["early_stopping_patience"] = int(
            args.early_stopping_patience
        )
    if args.threads is not None:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)
            tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        except RuntimeError as e:
            print(f"[WARNING] TensorFlow thread settings could not be changed: {e}")
    available_series = list(activos.keys())
    available_models = AVAILABLE_MODELS

    series_to_analyze = parse_selection(args.series, available_series, "series")
    if series_to_analyze is None:
        series_to_analyze = prompt_selection("series", available_series)

    models_to_analyze = parse_selection(args.models, available_models, "algorithms")
    if models_to_analyze is None:
        models_to_analyze = prompt_selection("algorithms", available_models)
    methods_to_analyze = parse_selection(
        args.methods,
        METODOS_PARTICION,
        "partition methods"
    )
    if len(METODOS_PARTICION) != 8:
        raise RuntimeError(
            f"The primary protocol must expose exactly eight methods; "
            f"found {len(METODOS_PARTICION)}."
        )
    try:
        run_seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 < args.statistical_alpha < 1.0:
        parser.error("--statistical-alpha must lie strictly between 0 and 1.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.early_stopping_patience <= 0:
        parser.error(
            "--early-stopping-patience must be greater than zero because "
            "every neural development fold requires validation-based selection."
        )
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be greater than zero when supplied.")
    try:
        tuning_seeds = parse_seeds(args.tuning_seeds)
    except ValueError as error:
        parser.error(f"Invalid --tuning-seeds: {error}")
    if args.tuning_trials <= 0:
        parser.error("--tuning-trials must be greater than zero.")
    if args.tune_hyperparameters and not args.tuning_calibration_end:
        parser.error(
            "--tuning-calibration-end is required with "
            "--tune-hyperparameters."
        )
    for option_name, date_value in (
        ("--tuning-calibration-end", args.tuning_calibration_end),
        ("--evaluation-start", args.evaluation_start),
    ):
        if date_value:
            try:
                pd.Timestamp(date_value)
            except (TypeError, ValueError) as error:
                parser.error(f"{option_name} is not a valid date: {error}")

    carbon_country = (
        args.carbon_country.upper()
        if args.carbon_country
        else None
    )

    if not args.no_emissions:
        if not carbon_country:
            parser.error(
                "--carbon-country is required when emissions "
                "tracking is enabled; use the real three-letter "
                "ISO code of the machine location."
            )

        if (
            len(carbon_country) != 3
            or not carbon_country.isalpha()
        ):
            parser.error(
                "--carbon-country must be a three-letter "
                "alphabetic ISO code."
            )

        if args.carbon_measure_power_secs <= 0:
            parser.error(
                "--carbon-measure-power-secs must be "
                "greater than zero."
            )

    carbon_config = {
        "enabled": not args.no_emissions,
        "country_iso_code": carbon_country,
        "region": args.carbon_region,
        "tracking_mode": args.carbon_tracking_mode,
        "measure_power_secs": args.carbon_measure_power_secs,
        "output_dir": args.carbon_output_dir,
        "output_file": args.carbon_output_file,
        "accounting_boundary": CARBON_ACCOUNTING_BOUNDARY,
    }

    print(f"[INFO] Definitive deliverable: {DELIVERABLE_ID}")
    print(f"[INFO] Selected series: {series_to_analyze}")
    print(f"[INFO] Selected algorithms: {models_to_analyze}")
    print(f"[INFO] Selected partition methods: {methods_to_analyze}")
    print(f"[INFO] Architecture protocol: {ARCHITECTURE_PROTOCOL}")
    print(f"[INFO] Repetition seeds: {run_seeds}")

    cache_dir = args.cache_dir if args.cache_dir else None
    series_dict, dates_dict = download_series(
        series_to_analyze, cache_dir, args.refresh_data
    )

    neural_selected = any(
        model in NEURAL_MODELS for model in models_to_analyze
    )
    if args.tune_hyperparameters:
        if not neural_selected:
            parser.error(
                "--tune-hyperparameters requires at least one neural model."
            )
        preflight_runtime(
            require_gpu=bool(not args.allow_cpu),
            carbon_config=carbon_config,
        )
        tuning_manifest = run_temporal_hyperparameter_optimization(
            series_to_analyze,
            models_to_analyze,
            args.tuning_calibration_end,
            args.tuning_trials,
            tuning_seeds,
            args.tuning_output,
            args.batch_size,
            args.early_stopping_patience,
            not args.no_emissions,
            carbon_config,
        )
        tuning_wall_time = float(time.perf_counter() - invocation_start)
        tuning_summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "deliverable_id": DELIVERABLE_ID,
            "status": "hyperparameter_validation_completed",
            "completed_at_utc": dt.datetime.now(
                dt.timezone.utc
            ).isoformat(),
            "wall_time_seconds": tuning_wall_time,
            "validated_pair_count": int(len(tuning_manifest["pairs"])),
            "tuning_output": os.path.abspath(args.tuning_output),
            "tuning_output_sha256": file_sha256(args.tuning_output),
            "calibration_end_date_inclusive": args.tuning_calibration_end,
            "total_tuning_fitting_time_seconds": float(sum(
                float(trial["fit_time_seconds"])
                for record in tuning_manifest["pairs"].values()
                for trial in record.get("trials", [])
                if trial.get("status") == "completed"
            )),
            "total_tuning_emissions_kgco2eq": (
                float(sum(
                    float(trial["emissions_kgco2eq"])
                    for record in tuning_manifest["pairs"].values()
                    for trial in record.get("trials", [])
                    if trial.get("emissions_kgco2eq") is not None
                ))
                if not args.no_emissions else None
            ),
            "carbon_accounting_boundary": TUNING_CARBON_ACCOUNTING_BOUNDARY,
            "required_experimental_constraint": (
                "the subsequent --evaluation-start must be strictly later "
                "than the calibration cutoff"
            ),
        }
        atomic_write_json(args.run_summary_output, tuning_summary)
        print(
            "[INFO] HYPERPARAMETER VALIDATION WALL TIME: "
            f"{tuning_wall_time:.6f} seconds"
        )
        return tuning_summary

    if ARCHITECTURE_PROTOCOL == "controlled" and args.validated_hyperparameters:
        parser.error(
            "--validated-hyperparameters applies only to "
            "--architecture-protocol individually_tuned."
        )
    if ARCHITECTURE_PROTOCOL == "individually_tuned" and neural_selected:
        if args.validated_hyperparameters:
            load_validated_hyperparameters(
                args.validated_hyperparameters,
                series_to_analyze,
                models_to_analyze,
                evaluation_start=args.evaluation_start,
            )
        elif not args.allow_unvalidated_hyperparameters:
            parser.error(
                "A confirmatory individually_tuned neural run requires "
                "--validated-hyperparameters. First run the temporal tuning "
                "stage, or use --allow-unvalidated-hyperparameters only for "
                "an explicitly exploratory execution."
            )
        else:
            print(
                "[WARNING] Unvalidated hyperparameter candidates are enabled; "
                "this run is exploratory and must not be described as optimized."
            )
    if args.evaluation_start:
        restrict_to_evaluation_period(args.evaluation_start)

    dataset_summary = build_dataset_summary(series_to_analyze)
    identity, run_fingerprint = build_experiment_identity(
        args,
        series_to_analyze,
        models_to_analyze,
        methods_to_analyze,
        run_seeds,
        carbon_config,
        dataset_summary,
    )
    checkpoint_root = args.checkpoint_dir or "experiment_checkpoints"
    checkpoint_dir = os.path.join(
        checkpoint_root, run_fingerprint[:16]
    )
    manifest_path, manifest = initialize_run_manifest(
        checkpoint_dir,
        identity,
        run_fingerprint,
    )

    try:
        preflight = preflight_runtime(
            require_gpu=bool(neural_selected and not args.allow_cpu),
            carbon_config=carbon_config,
        )
        write_environment_artifacts(
            args,
            series_to_analyze,
            models_to_analyze,
            methods_to_analyze,
            run_seeds,
            carbon_config,
            dataset_summary,
            run_fingerprint=run_fingerprint,
            preflight=preflight,
        )

        if not args.skip_partition_plots:
            run_partition_visualizations(
                methods_to_analyze,
                show_plots=args.show_plots,
                output_dir=args.figure_output_dir,
            )

        global_results_by_asset = run_evaluation(
            series_to_analyze,
            models_to_analyze,
            methods_to_analyze,
            run_seeds,
            epochs_override=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            track_emissions=not args.no_emissions,
            carbon_config=carbon_config,
            checkpoint_dir=checkpoint_dir,
            run_fingerprint=run_fingerprint,
        )
        global_results_by_model = build_results_by_model(
            global_results_by_asset
        )
        save_prediction_artifacts(
            global_results_by_asset,
            output_dir=args.prediction_output_dir,
        )

        if not args.skip_global_plots:
            write_global_visualizations(
                global_results_by_model,
                show_plots=args.show_plots,
                output_dir=args.figure_output_dir,
            )

        df_global = save_global_results_csv(
            global_results_by_asset, args.output_csv
        )
        save_metric_summary_tables(
            df_global,
            output_dir=args.metric_table_output_dir,
        )
        run_statistical_analysis(
            df_global,
            output_dir=args.statistical_output_dir,
            alpha=args.statistical_alpha,
        )

        results = [
            result
            for model_results in global_results_by_asset.values()
            for result_list in model_results.values()
            for result in result_list
        ]
        total_fitting_time = float(sum(
            float(result["time"])
            for result in results
            if result["model"] in NEURAL_MODELS
        ))
        tracked_emissions = [
            float(result["emissions"])
            for result in results
            if result["emissions"] is not None
        ]
        total_emissions = (
            float(sum(tracked_emissions)) if tracked_emissions else None
        )
        fitting_stage_count = int(sum(
            int(result.get("fold_count", 0))
            for result in results
            if result["model"] in NEURAL_MODELS
        ))
        expected_final_results = (
            len(series_to_analyze)
            * sum(model in NEURAL_MODELS for model in models_to_analyze)
            * len(methods_to_analyze)
            * len(run_seeds)
            + len(series_to_analyze)
            * len(methods_to_analyze)
            * int("Naive Persistence" in models_to_analyze)
        )
        individual_table_paths = [
            os.path.join(
                os.getcwd(),
                "table_"
                f"{sanitize_run_component(asset)}__"
                f"{sanitize_run_component(model)}.txt",
            )
            for asset in series_to_analyze
            for model in models_to_analyze
        ]
        artifact_hashes = collect_artifact_hashes([
            args.output_csv,
            args.environment_output,
            args.software_output,
            args.prediction_output_dir,
            args.figure_output_dir,
            args.metric_table_output_dir,
            args.statistical_output_dir,
            args.carbon_output_dir if not args.no_emissions else None,
            *individual_table_paths,
        ])
        invocation_wall_time = float(time.perf_counter() - invocation_start)
        cumulative_wall_time = float(
            manifest.get("previous_cumulative_wall_time_seconds", 0.0)
            + invocation_wall_time
        )
        execution_summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "deliverable_id": DELIVERABLE_ID,
            "status": "completed",
            "run_fingerprint": run_fingerprint,
            "completed_at_utc": dt.datetime.now(
                dt.timezone.utc
            ).isoformat(),
            "process_started_at_utc": PROCESS_STARTED_AT_UTC,
            "invocation_wall_time_seconds": invocation_wall_time,
            "cumulative_wall_time_seconds_across_resumes": cumulative_wall_time,
            "total_neural_fitting_time_seconds": total_fitting_time,
            "fitting_time_definition": CARBON_ACCOUNTING_BOUNDARY,
            "fitting_stage_count": fitting_stage_count,
            "total_reported_emissions_kgco2eq": total_emissions,
            "final_result_count": int(len(results)),
            "expected_final_result_count": int(expected_final_results),
            "completeness_check_passed": bool(
                len(results) == expected_final_results
            ),
            "test_protocol": (
                "method-specific fold tests with metrics computed once on "
                "the pooled unique out-of-sample predictions"
            ),
            "hyperparameter_validation_enforced": bool(
                ARCHITECTURE_PROTOCOL == "individually_tuned"
                and neural_selected
                and not args.allow_unvalidated_hyperparameters
            ),
            "resume_count": int(manifest["resume_count"]),
            "wall_time_note": (
                "Measured with time.perf_counter from early process startup "
                "(before heavy scientific-library imports) through all data, "
                "training, predictions, tables, figures, and statistical "
                "analyses; final summary serialization itself is excluded. On a "
                "resumed experiment, cumulative_wall_time_seconds_across_resumes "
                "adds every completed or failed invocation recorded by the same "
                "checkpoint manifest; invocation_wall_time_seconds covers only "
                "the current process."
            ),
            "artifact_hashes": artifact_hashes,
        }
        atomic_write_json(args.run_summary_output, execution_summary)
        finalize_run_manifest(
            manifest_path,
            manifest,
            status="completed",
            invocation_wall_time_seconds=invocation_wall_time,
            cumulative_wall_time_seconds=cumulative_wall_time,
            execution_summary=execution_summary,
            run_summary_file=os.path.abspath(args.run_summary_output),
        )
        print(
            "[INFO] COMPLETE EXECUTION WALL TIME: "
            f"{invocation_wall_time:.6f} seconds"
        )
        print(
            "[INFO] TOTAL NEURAL FITTING TIME (all method-specific folds): "
            f"{total_fitting_time:.6f} seconds"
        )
        print(
            f"[INFO] Execution summary saved as '{args.run_summary_output}'"
        )
        return execution_summary
    except BaseException as error:
        finalize_run_manifest(
            manifest_path,
            manifest,
            status="failed",
            failed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            invocation_wall_time_seconds=float(
                time.perf_counter() - invocation_start
            ),
            cumulative_wall_time_seconds=float(
                manifest.get("previous_cumulative_wall_time_seconds", 0.0)
                + (time.perf_counter() - invocation_start)
            ),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
