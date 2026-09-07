"""Editable model and row-local feature proposals for the OpenBiomechanics example.

Keep evaluation, athlete grouping, target, and metric output in train.py fixed.
Available estimators and parameters are documented in models.py.
"""
import numpy as np

MODEL_TYPE = "xgboost"
MODEL_PARAMS = {}
TOP_N_FEATURES = 15  # None disables supervised selection.


def engineer_features(X):
    """Return row-local features, preserving row order and excluding identifiers.

    Do not fit scalers, imputers, encoders, or supervised selectors here. Any
    learned transform belongs inside a model pipeline fitted on training folds.
    """
    X = X.copy()
    # Deterministic row-local transforms do not learn from held-out observations.
    ratios = {
        "thorax_to_elbow_transfer_ratio": ("thorax_distal_transfer_fp_br", "elbow_transfer_fp_br"),
        "shoulder_to_elbow_transfer_ratio": ("shoulder_transfer_fp_br", "elbow_transfer_fp_br"),
        "pelvis_to_thorax_transfer_ratio": ("pelvis_lumbar_transfer_fp_br", "thorax_distal_transfer_fp_br"),
        "torso_to_pelvis_rot_ratio": ("max_torso_rotational_velo", "max_pelvis_rotational_velo"),
        "grf_lead_rear_ratio": ("lead_grf_mag_max", "rear_grf_mag_max"),
        "moment_ratio": ("shoulder_internal_rotation_moment", "elbow_varus_moment"),
    }
    for name, (numerator, denominator) in ratios.items():
        if numerator in X and denominator in X:
            denom = X[denominator].to_numpy(dtype=float)
            # Near-zero denominators have an explicit finite neutral value.
            X[name] = np.divide(X[numerator].to_numpy(dtype=float), denom,
                                out=np.zeros(len(X)), where=np.abs(denom) > 1e-6)
    transfers = ["shoulder_transfer_fp_br", "elbow_transfer_fp_br",
                 "thorax_distal_transfer_fp_br", "pelvis_lumbar_transfer_fp_br"]
    if set(transfers).issubset(X.columns):
        X["total_energy_transfer"] = X[transfers].sum(axis=1)
    return X
