"""Hybrid classical + quantum SVM training pipeline for QuDrugGuard V2."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import data
import drug_db
from quantum_circuit import QuantumKernelSVM, live_circuit_bundle


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "qudrug_model.pkl"


def build_training_frame() -> pd.DataFrame:
    rows = []
    for item in data.TRAINING_PAIRS:
        engineered = drug_db.build_pair_feature_vector(item["drug_a"], item["drug_b"])
        row = {"drug_a": item["drug_a"], "drug_b": item["drug_b"], "label": item["label"], "rationale": item["rationale"]}
        row.update(engineered["feature_vector"])
        rows.append(row)
    return pd.DataFrame(rows)


def train_models() -> dict[str, Any]:
    frame = build_training_frame()
    x = frame[drug_db.INTERACTION_FEATURES].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    classical_model = SVC(kernel="rbf", probability=True, class_weight="balanced", gamma="scale", random_state=42)
    classical_model.fit(x_scaled, y)
    classical_accuracy = accuracy_score(y, classical_model.predict(x_scaled))

    pca = PCA(n_components=4, random_state=42)
    x_quantum = pca.fit_transform(x_scaled)
    quantum_model = QuantumKernelSVM(random_state=42).fit(x_quantum, y)
    quantum_accuracy = accuracy_score(y, quantum_model.predict(x_quantum))

    artifact = {
        "version": "2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(drug_db.INTERACTION_FEATURES),
        "scaler": scaler,
        "pca": pca,
        "classical_model": classical_model,
        "quantum_model": quantum_model,
        "training_frame": frame,
        "training_metrics": {"classical_accuracy": round(float(classical_accuracy), 4), "quantum_accuracy": round(float(quantum_accuracy), 4)},
    }
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)
    return artifact


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return train_models()
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def _mechanistic_risk(engineered: dict[str, Any]) -> float:
    vector = engineered["feature_vector"]
    weighted_sum = (
        vector["shared_enzyme_count"] * 0.14
        + vector["inhibitor_overlap_score"] * 0.16
        + vector["transporter_conflict"] * 0.08
        + vector["solubility_gap"] * 0.07
        + vector["qt_stack"] * 0.12
        + vector["cns_stack"] * 0.08
        + vector["serotonin_stack"] * 0.10
        + vector["bleeding_stack"] * 0.12
        + vector["nephro_stack"] * 0.05
        + vector["hepatotoxic_stack"] * 0.05
        + vector["narrow_therapeutic_penalty"] * 0.12
    )
    return float(np.clip(weighted_sum, 0, 1))


def predict_interaction(drug_a: str, drug_b: str, model_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    bundle = model_bundle or load_model()
    engineered = drug_db.build_pair_feature_vector(drug_a, drug_b)
    x = np.asarray([engineered["feature_list"]], dtype=float)
    x_scaled = bundle["scaler"].transform(x)
    x_quantum = bundle["pca"].transform(x_scaled)

    classical_probability = float(bundle["classical_model"].predict_proba(x_scaled)[0, 1])
    quantum_probability = float(bundle["quantum_model"].predict_proba(x_quantum)[0, 1])
    mechanistic_probability = _mechanistic_risk(engineered)
    final_score = float(np.clip(classical_probability * 0.42 + quantum_probability * 0.38 + mechanistic_probability * 0.20, 0, 1))
    confidence = float(np.clip(abs(final_score - 0.5) * 2, 0, 1))
    label = "Dangerous" if final_score >= 0.55 else "Safe"
    quantum_live = live_circuit_bundle(x_quantum[0], shots=1024)

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "label": label,
        "risk_score": round(final_score, 4),
        "confidence": round(confidence, 4),
        "shared_enzymes": engineered["shared_enzyme"]["shared"],
        "drivers": engineered["drivers"],
        "classical_probability": round(classical_probability, 4),
        "quantum_probability": round(quantum_probability, 4),
        "mechanistic_probability": round(mechanistic_probability, 4),
        "features": engineered["feature_vector"],
        "training_metrics": bundle["training_metrics"],
        "quantum_live": quantum_live,
    }


def main() -> None:
    artifact = train_models()
    print("Training rows:", len(artifact["training_frame"]))
    print("Classical SVM accuracy:", artifact["training_metrics"]["classical_accuracy"])
    print("Quantum SVM accuracy:", artifact["training_metrics"]["quantum_accuracy"])
    print(f"Model saved to: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
