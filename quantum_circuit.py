"""Advanced quantum circuit utilities for QuDrugGuard V2.

Upgrades over V1
----------------
- 8 qubits (was 4)
- Multi-layer variational feature map (ZZFeatureMap-style with configurable depth)
- Three entanglement topologies: linear, circular, full
- Two-qubit ZZ interactions for richer expressibility
- Multi-basis (X, Y, Z) Pauli expectation values
- Parameterised ansatz (hardware-efficient) for hybrid VQE-style use
- Quantum kernel bank: fidelity, projected, and linear-combination kernels
- Kernel caching via lru_cache for expensive statevector comparisons
- Richer SVM wrapper with cross-validation support
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

NUM_QUBITS: int = 8
SHOTS: int = 2048
FEATURE_MAP_REPS: int = 2          # depth of the feature map encoding layers
ANSATZ_REPS: int = 2               # depth of the hardware-efficient ansatz


class Entanglement(str, Enum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    FULL = "full"


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def _normalise_features(features: np.ndarray, num_qubits: int = NUM_QUBITS) -> np.ndarray:
    """Resize, normalise to [-π, π], and return a 1-D array of length *num_qubits*."""
    vector = np.asarray(features, dtype=float).reshape(-1)
    if vector.size < num_qubits:
        # Pad with repeated values (circular)
        repeats = int(np.ceil(num_qubits / vector.size))
        vector = np.tile(vector, repeats)
    vector = vector[:num_qubits]
    # Soft normalisation: map to [-π, π] via arctan scaling
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm * np.pi
    return np.clip(vector, -np.pi, np.pi)


# ---------------------------------------------------------------------------
# Entanglement helpers
# ---------------------------------------------------------------------------

def _entanglement_pairs(
    num_qubits: int, topology: Entanglement
) -> list[tuple[int, int]]:
    """Return list of (control, target) qubit pairs for the chosen topology."""
    if topology == Entanglement.LINEAR:
        return [(i, i + 1) for i in range(num_qubits - 1)]
    if topology == Entanglement.CIRCULAR:
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        pairs.append((num_qubits - 1, 0))
        return pairs
    # FULL: all unique pairs
    return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]


# ---------------------------------------------------------------------------
# Feature map (data encoding circuit)
# ---------------------------------------------------------------------------

def build_feature_map(
    features: np.ndarray,
    *,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
    measure: bool = True,
) -> QuantumCircuit:
    """Multi-layer ZZFeatureMap-style encoding circuit.

    Each repetition applies:
      1. Hadamard layer  → superposition
      2. Single-qubit Ry(θ), Rz(2θ) rotations  → data encoding
      3. ZZ-entanglement layer  → pairwise correlations Rz(2·xi·xj) on CX ladders
      4. Secondary Ry(arctan(xi)) layer  → non-linear re-encoding
    """
    vector = _normalise_features(features, num_qubits)
    pairs = _entanglement_pairs(num_qubits, topology)
    circuit = QuantumCircuit(num_qubits, num_qubits if measure else 0)

    for rep in range(reps):
        # --- Hadamard initialisation (first rep only to avoid cancellation) ---
        if rep == 0:
            circuit.h(range(num_qubits))

        # --- Single-qubit data encoding ---
        for qubit, val in enumerate(vector):
            circuit.ry(val, qubit)
            circuit.rz(2.0 * val, qubit)

        circuit.barrier(label=f"enc-{rep}")

        # --- ZZ entanglement layer ---
        for ctrl, tgt in pairs:
            phase = 2.0 * float(vector[ctrl] * vector[tgt])
            circuit.cx(ctrl, tgt)
            circuit.rz(phase, tgt)
            circuit.cx(ctrl, tgt)

        circuit.barrier(label=f"ent-{rep}")

        # --- Non-linear re-encoding (arctan squashing) ---
        for qubit, val in enumerate(vector):
            circuit.ry(float(np.arctan(val)), qubit)

        # --- Global mixing layer (last qubit cross-connects to first) ---
        circuit.cx(num_qubits - 1, 0)
        circuit.rx(float(np.mean(vector)), 0)
        circuit.ry(float(np.std(vector) + 1e-6), num_qubits // 2)
        circuit.rz(float(np.ptp(vector)), num_qubits - 1)

        if rep < reps - 1:
            circuit.barrier(label=f"rep-{rep}-end")

    if measure:
        circuit.barrier(label="meas")
        circuit.measure(range(num_qubits), range(num_qubits))

    return circuit


# ---------------------------------------------------------------------------
# Hardware-efficient variational ansatz
# ---------------------------------------------------------------------------

def build_ansatz(
    *,
    reps: int = ANSATZ_REPS,
    topology: Entanglement = Entanglement.LINEAR,
    num_qubits: int = NUM_QUBITS,
) -> QuantumCircuit:
    """Ry-Rz hardware-efficient ansatz with entanglement blocks.

    Returns a parametrised circuit; bind parameters before simulation.
    """
    params = ParameterVector("θ", length=num_qubits * 2 * reps)
    pairs = _entanglement_pairs(num_qubits, topology)
    circuit = QuantumCircuit(num_qubits)
    param_index = 0

    for _ in range(reps):
        for qubit in range(num_qubits):
            circuit.ry(params[param_index], qubit)
            param_index += 1
            circuit.rz(params[param_index], qubit)
            param_index += 1
        for ctrl, tgt in pairs:
            circuit.cx(ctrl, tgt)
        circuit.barrier()

    return circuit


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_counts(
    circuit: QuantumCircuit, shots: int = SHOTS
) -> dict[str, int]:
    backend = AerSimulator()
    compiled = transpile(circuit, backend)
    result = backend.run(compiled, shots=shots).result()
    return result.get_counts(compiled)


def statevector_embedding(
    features: np.ndarray,
    *,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> Statevector:
    circuit = build_feature_map(
        features, reps=reps, topology=topology, num_qubits=num_qubits, measure=False
    )
    return Statevector.from_instruction(circuit)


# ---------------------------------------------------------------------------
# Quantum kernel functions
# ---------------------------------------------------------------------------

def _vec_key(vec: np.ndarray) -> str:
    """Deterministic cache key for a float array."""
    return hashlib.md5(np.ascontiguousarray(vec, dtype=np.float64).tobytes()).hexdigest()


@lru_cache(maxsize=4096)
def _cached_statevector(key: str, flat_vec: tuple[float, ...], reps: int, topology: str, num_qubits: int) -> np.ndarray:
    """Statevector cached by content hash (lru_cache requires hashable args)."""
    vec = np.array(flat_vec, dtype=float)
    return statevector_embedding(vec, reps=reps, topology=Entanglement(topology), num_qubits=num_qubits).data


def _get_sv(vec: np.ndarray, reps: int, topology: Entanglement, num_qubits: int) -> np.ndarray:
    key = _vec_key(vec)
    return _cached_statevector(key, tuple(vec.tolist()), reps, topology.value, num_qubits)


def fidelity_kernel(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> float:
    """|⟨ψ(a)|ψ(b)⟩|² — quantum fidelity kernel."""
    sv_a = _get_sv(_normalise_features(vec_a, num_qubits), reps, topology, num_qubits)
    sv_b = _get_sv(_normalise_features(vec_b, num_qubits), reps, topology, num_qubits)
    return float(np.abs(np.vdot(sv_a, sv_b)) ** 2)


def projected_kernel(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    gamma: float = 1.0,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> float:
    """Projected quantum kernel: exp(-γ · ‖E(a) - E(b)‖²).

    Uses single-qubit Z-expectation values as the projected feature space,
    which is cheaper to estimate and avoids the curse of dimensionality.
    """
    sv_a = _get_sv(_normalise_features(vec_a, num_qubits), reps, topology, num_qubits)
    sv_b = _get_sv(_normalise_features(vec_b, num_qubits), reps, topology, num_qubits)
    state_a = Statevector(sv_a)
    state_b = Statevector(sv_b)

    def z_expectations(state: Statevector) -> np.ndarray:
        exps = []
        for q in range(num_qubits):
            pauli_str = "I" * (num_qubits - q - 1) + "Z" + "I" * q
            op = SparsePauliOp(pauli_str)
            exps.append(float(np.real(state.expectation_value(op))))
        return np.array(exps)

    diff = z_expectations(state_a) - z_expectations(state_b)
    return float(np.exp(-gamma * np.dot(diff, diff)))


def combo_kernel(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    alpha: float = 0.5,
    gamma: float = 1.0,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> float:
    """Convex combination of fidelity and projected kernels.

    k(a,b) = α·k_fidelity(a,b) + (1-α)·k_projected(a,b)
    """
    kw = dict(reps=reps, topology=topology, num_qubits=num_qubits)
    return alpha * fidelity_kernel(vec_a, vec_b, **kw) + (1 - alpha) * projected_kernel(vec_a, vec_b, gamma=gamma, **kw)


# ---------------------------------------------------------------------------
# Multi-basis expectation values
# ---------------------------------------------------------------------------

def multi_basis_expectations(
    features: np.ndarray,
    *,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> dict[str, list[float]]:
    """Compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for every qubit from the statevector."""
    sv = _get_sv(_normalise_features(features, num_qubits), reps, topology, num_qubits)
    state = Statevector(sv)
    result: dict[str, list[float]] = {"X": [], "Y": [], "Z": []}
    for basis, label in [("X", "X"), ("Y", "Y"), ("Z", "Z")]:
        for q in range(num_qubits):
            pad_l = "I" * (num_qubits - q - 1)
            pad_r = "I" * q
            op = SparsePauliOp(pad_l + label + pad_r)
            result[basis].append(round(float(np.real(state.expectation_value(op))), 5))
    return result


def expectation_from_counts(counts: dict[str, int], num_qubits: int = NUM_QUBITS) -> dict[str, Any]:
    """Richer summary: per-qubit Z expectations, entropy, dominant state."""
    total = max(sum(counts.values()), 1)
    z_expectations = []
    for qubit in range(num_qubits):
        exp = sum(
            (1 if bitstring[::-1][qubit] == "0" else -1) * freq / total
            for bitstring, freq in counts.items()
        )
        z_expectations.append(round(exp, 4))

    probs = np.array(list(counts.values()), dtype=float) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    weighted_score = float(np.clip((1 - np.mean(z_expectations)) / 2, 0, 1))
    dominant_state = max(counts, key=counts.__getitem__) if counts else "0" * num_qubits

    return {
        "z_expectations": z_expectations,
        "weighted_score": round(weighted_score, 4),
        "entropy_bits": round(entropy, 4),
        "dominant_state": dominant_state,
        "num_distinct_states": len(counts),
    }


# ---------------------------------------------------------------------------
# Plotly visualisation helpers
# ---------------------------------------------------------------------------

def build_circuit_figure(circuit: QuantumCircuit):
    import plotly.graph_objects as go

    num_qubits = circuit.num_qubits
    figure = go.Figure()

    # Wire lines
    for q in range(num_qubits):
        figure.add_trace(go.Scatter(
            x=[0, max(1, len(circuit.data) + 1)],
            y=[num_qubits - q] * 2,
            mode="lines",
            line={"color": "#334155", "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        ))

    GATE_COLORS = {
        "h": "#6366f1", "ry": "#0ea5e9", "rz": "#14b8a6",
        "rx": "#8b5cf6", "cx": "#f59e0b", "measure": "#f43f5e",
        "barrier": None,
    }

    for idx, instruction in enumerate(circuit.data, start=1):
        name = instruction.operation.name.lower()
        if name == "barrier":
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        y_pts = [num_qubits - q for q in qubits]
        color = GATE_COLORS.get(name, "#64748b")
        label = name.upper()

        if len(y_pts) > 1:
            figure.add_trace(go.Scatter(
                x=[idx, idx], y=[min(y_pts), max(y_pts)],
                mode="lines", line={"color": color, "width": 2},
                hoverinfo="skip", showlegend=False,
            ))
        figure.add_trace(go.Scatter(
            x=[idx] * len(y_pts), y=y_pts,
            mode="markers+text",
            text=[label] * len(y_pts),
            textposition="middle center",
            marker={"size": 28, "color": color, "symbol": "square",
                    "line": {"color": "#e2e8f0", "width": 1}},
            hovertemplate=f"Gate: {name}<extra></extra>",
            showlegend=False,
        ))

    figure.update_layout(
        height=max(320, num_qubits * 40),
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False, "range": [0.5, num_qubits + 0.5]},
    )
    return figure


def build_counts_figure(counts: dict[str, int]):
    import plotly.graph_objects as go

    ordered = sorted(counts.items())
    total = max(sum(v for _, v in ordered), 1)
    probs = [v / total for _, v in ordered]

    figure = go.Figure(data=[
        go.Bar(
            x=[k for k, _ in ordered],
            y=probs,
            marker={"color": "#14b8a6", "opacity": 0.85},
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    ])
    figure.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Measured bitstring",
        yaxis_title="Probability",
    )
    return figure


def build_expectation_radar(expectations: dict[str, list[float]]):
    """Radar / spider chart of per-qubit multi-basis expectations."""
    import plotly.graph_objects as go

    num_q = len(next(iter(expectations.values())))
    labels = [f"Q{i}" for i in range(num_q)]
    colors = {"X": "#6366f1", "Y": "#14b8a6", "Z": "#f59e0b"}

    figure = go.Figure()
    for basis, vals in expectations.items():
        figure.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            name=f"⟨{basis}⟩",
            line={"color": colors[basis], "width": 2},
            fill="toself",
            opacity=0.3,
        ))
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [-1, 1]}},
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h"},
    )
    return figure


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

def live_circuit_bundle(
    features: np.ndarray,
    *,
    shots: int = SHOTS,
    reps: int = FEATURE_MAP_REPS,
    topology: Entanglement = Entanglement.CIRCULAR,
    num_qubits: int = NUM_QUBITS,
) -> dict[str, Any]:
    circuit = build_feature_map(features, reps=reps, topology=topology, num_qubits=num_qubits, measure=True)
    counts = simulate_counts(circuit, shots=shots)
    expectation = expectation_from_counts(counts, num_qubits=num_qubits)
    multi_exp = multi_basis_expectations(features, reps=reps, topology=topology, num_qubits=num_qubits)
    return {
        "circuit": circuit,
        "counts": counts,
        "expectation": expectation,
        "multi_basis_expectations": multi_exp,
        "circuit_figure": build_circuit_figure(circuit),
        "counts_figure": build_counts_figure(counts),
        "expectation_radar": build_expectation_radar(multi_exp),
    }


# ---------------------------------------------------------------------------
# Quantum Kernel SVM — enhanced
# ---------------------------------------------------------------------------

KernelName = Literal["fidelity", "projected", "combo"]


@dataclass
class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible quantum kernel SVM.

    Parameters
    ----------
    kernel_name : "fidelity" | "projected" | "combo"
        Which quantum kernel to use.
    reps : int
        Feature map depth.
    topology : Entanglement
        Qubit entanglement topology.
    num_qubits : int
        Circuit width.
    alpha : float
        Combo kernel mixing weight (ignored for other kernels).
    gamma : float
        Projected kernel bandwidth (ignored for fidelity kernel).
    random_state : int
        SVC random state.
    """

    kernel_name: KernelName = "fidelity"
    reps: int = FEATURE_MAP_REPS
    topology: Entanglement = Entanglement.CIRCULAR
    num_qubits: int = NUM_QUBITS
    alpha: float = 0.5
    gamma: float = 1.0
    random_state: int = 42

    def __post_init__(self):
        self.training_vectors_: np.ndarray | None = None
        self.svc_: SVC = SVC(
            kernel="precomputed",
            probability=True,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def _resolve_training_vectors(self) -> np.ndarray | None:
        """Backward-compatible access for older pickled model attributes."""
        if hasattr(self, "training_vectors_"):
            return self.training_vectors_
        if hasattr(self, "training_vectors"):
            return self.training_vectors
        return None

    def _resolve_svc(self) -> SVC | None:
        """Backward-compatible access for older pickled classifier attributes."""
        if hasattr(self, "svc_"):
            return self.svc_
        if hasattr(self, "svc"):
            return self.svc
        if hasattr(self, "model"):
            return self.model
        return None

    # ------------------------------------------------------------------
    def _k(self, a: np.ndarray, b: np.ndarray) -> float:
        kw = dict(reps=self.reps, topology=self.topology, num_qubits=self.num_qubits)
        if self.kernel_name == "fidelity":
            return fidelity_kernel(a, b, **kw)
        if self.kernel_name == "projected":
            return projected_kernel(a, b, gamma=self.gamma, **kw)
        return combo_kernel(a, b, alpha=self.alpha, gamma=self.gamma, **kw)

    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(left), len(right)))
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                matrix[i, j] = self._k(a, b)
        return matrix

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVM":
        self.training_vectors_ = np.asarray(X, dtype=float)
        # Keep alias for compatibility with previously pickled versions.
        self.training_vectors = self.training_vectors_
        K = self._kernel_matrix(self.training_vectors_, self.training_vectors_)
        self.svc_.fit(K, np.asarray(y, dtype=int))
        # Keep aliases for compatibility with previously pickled versions.
        self.svc = self.svc_
        self.model = self.svc_
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        training_vectors = self._resolve_training_vectors()
        if training_vectors is None:
            raise RuntimeError("Model is not fitted.")
        svc = self._resolve_svc()
        if svc is None:
            raise RuntimeError("Classifier backend is missing. Retrain model with `python train.py`.")
        K = self._kernel_matrix(np.asarray(X, dtype=float), training_vectors)
        return svc.predict_proba(K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> dict[str, float]:
        """Convenience wrapper for k-fold CV using the precomputed kernel trick."""
        scores = cross_val_score(self, X, y, cv=cv, scoring="balanced_accuracy")
        return {
            "mean_balanced_accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        }
