"""Fast circuit-level Monte Carlo simulation for bivariate bicycle codes.

The decoder data are produced by ``decoder_offline.py``.  This module keeps
the noise model and success criterion of ``old_decoder_run.py``, but simulates
many trials together and applies each syndrome-extraction round as a parallel
Clifford layer instead of constructing Python gate lists trial by trial.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ldpc import BpOsdDecoder
from scipy.sparse import csc_matrix
from tqdm import tqdm

from config import (
    N_c,
    bp_osd,
    code_distance,
    code_param,
    decoder_data_dir,
    error_rate,
    noise,
    online,
    result_dir,
    sched_x,
    sched_z,
)


# Non-identity two-qubit Paulis in the order used by old_decoder_run.py.
_CNOT_PAULIS = (
    "IX",  # placeholder for index 0 (no fault); never applied
    "XI", "YI", "ZI", "IX", "IY", "IZ",
    "XX", "YY", "ZZ", "XY", "YX", "YZ", "ZY", "XZ", "ZX",
)
_CNOT_X_CONTROL = np.array([0] + [p[0] in "XY" for p in _CNOT_PAULIS[1:]], dtype=np.uint8)
_CNOT_Z_CONTROL = np.array([0] + [p[0] in "YZ" for p in _CNOT_PAULIS[1:]], dtype=np.uint8)
_CNOT_X_TARGET = np.array([0] + [p[1] in "XY" for p in _CNOT_PAULIS[1:]], dtype=np.uint8)
_CNOT_Z_TARGET = np.array([0] + [p[1] in "YZ" for p in _CNOT_PAULIS[1:]], dtype=np.uint8)

# IDLE error types: 0=I, 1=X, 2=Y, 3=Z.
_IDLE_X = np.array([0, 1, 1, 0], dtype=np.uint8)
_IDLE_Z = np.array([0, 0, 1, 1], dtype=np.uint8)


@dataclass(frozen=True)
class CompiledCycle:
    n2: int
    state_size: int
    data: np.ndarray
    xchecks: np.ndarray
    zchecks: np.ndarray
    cnot_layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    idle_layers: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class DecoderData:
    hdec_x: csc_matrix
    hdec_z: csc_matrix
    logical_map_x: csc_matrix
    logical_map_z: csc_matrix
    prob_x: np.ndarray
    prob_z: np.ndarray
    lx: np.ndarray
    lz: np.ndarray
    cycle: CompiledCycle
    k: int


def _gate_kinds(gates: Sequence[tuple]) -> tuple[str, ...]:
    return tuple(gate[0] for gate in gates)


def compile_cycle(raw: dict) -> CompiledCycle:
    """Compile the legacy named-qubit cycle into eight parallel CNOT layers."""
    cycle = raw["cycle"]
    lin_order = raw["lin_order"]
    n2 = len(raw["Xchecks"])
    n = 2 * n2

    def take(position: int, count: int, kind: str) -> tuple[list[tuple], int]:
        gates = cycle[position : position + count]
        if len(gates) != count or any(gate[0] != kind for gate in gates):
            actual = _gate_kinds(gates[:3])
            raise ValueError(f"Unexpected cycle layout at {position}: expected {kind}, got {actual}")
        return gates, position + count

    position = 0
    _, position = take(position, n2, "PrepX")
    round0, position = take(position, n2, "CNOT")
    idle0, position = take(position, n2, "IDLE")

    cnot_layers: list[tuple[np.ndarray, np.ndarray]] = []

    def indexed_layer(gates: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
        controls = np.fromiter((lin_order[g[1]] for g in gates), dtype=np.intp)
        targets = np.fromiter((lin_order[g[2]] for g in gates), dtype=np.intp)
        if len(np.unique(np.concatenate((controls, targets)))) != 2 * len(controls):
            raise ValueError("CNOT layer is not parallel")
        return controls, targets

    cnot_layers.append(indexed_layer(round0))
    for _ in range(5):
        xgates, position = take(position, n2, "CNOT")
        zgates, position = take(position, n2, "CNOT")
        cnot_layers.append(indexed_layer(xgates + zgates))

    _, position = take(position, n2, "MeasZ")
    round6, position = take(position, n2, "CNOT")
    idle6, position = take(position, n2, "IDLE")
    cnot_layers.append(indexed_layer(round6))
    idle7, position = take(position, n, "IDLE")
    _, position = take(position, n2, "MeasX")
    _, position = take(position, n2, "PrepZ")
    if position != len(cycle):
        raise ValueError(f"Cycle has {len(cycle) - position} unparsed operations")

    data = np.fromiter((lin_order[q] for q in raw["data_qubits"]), dtype=np.intp)
    xchecks = np.fromiter((lin_order[q] for q in raw["Xchecks"]), dtype=np.intp)
    zchecks = np.fromiter((lin_order[q] for q in raw["Zchecks"]), dtype=np.intp)
    idle_layers = tuple(
        np.fromiter((lin_order[g[1]] for g in gates), dtype=np.intp)
        for gates in (idle0, idle6, idle7)
    )
    return CompiledCycle(n2, 4 * n2, data, xchecks, zchecks, tuple(cnot_layers), idle_layers)


def validate_pickle(raw: dict) -> None:
    expected = {
        "num_cycles": N_c,
        "ell": code_param.ell,
        "m": code_param.m,
        "a1": code_param.a1,
        "a2": code_param.a2,
        "a3": code_param.a3,
        "b1": code_param.b1,
        "b2": code_param.b2,
        "b3": code_param.b3,
        "error_rate": error_rate,
        "sX": list(sched_x),
        "sZ": list(sched_z),
    }
    mismatches = [f"{key}: file={raw.get(key)!r}, config={value!r}" for key, value in expected.items() if raw.get(key) != value]
    expected_noise = {
        "init": noise.init,
        "idle": noise.idle,
        "cnot": noise.cnot,
        "meas": noise.meas,
    }
    file_noise = raw.get("noise")
    # Pickles from the earlier all-depolarizing format remain readable.
    if file_noise is None:
        file_noise = {key: raw.get("error_rate") for key in expected_noise}
    if file_noise != expected_noise:
        mismatches.append(f"noise: file={file_noise!r}, config={expected_noise!r}")
    if mismatches:
        raise ValueError("Offline data do not match config.py:\n  " + "\n  ".join(mismatches))


def find_decoder_data(directory: Path) -> Path:
    n = 2 * code_param.ell * code_param.m
    pattern = f"mydata_{n}_*_p_{error_rate}_cycles_{N_c}"
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one offline-data file matching {directory / pattern}, found {len(matches)}"
        )
    return matches[0]


def load_decoder_data(path: Path) -> DecoderData:
    print(f">>> Loading offline decoder data: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    validate_pickle(raw)
    k = int(np.asarray(raw["lx"].shape)[0])
    first_x = int(raw["first_logical_rowX"])
    first_z = int(raw["first_logical_rowZ"])
    hx = csc_matrix(raw["HX"], dtype=np.uint8)
    hz = csc_matrix(raw["HZ"], dtype=np.uint8)
    data = DecoderData(
        hdec_x=csc_matrix(raw["HdecX"], dtype=np.uint8),
        hdec_z=csc_matrix(raw["HdecZ"], dtype=np.uint8),
        logical_map_x=hx[first_x : first_x + k].tocsc(),
        logical_map_z=hz[first_z : first_z + k].tocsc(),
        prob_x=np.asarray(raw["probX"], dtype=float),
        prob_z=np.asarray(raw["probZ"], dtype=float),
        lx=np.asarray(raw["lx"], dtype=np.uint8),
        lz=np.asarray(raw["lz"], dtype=np.uint8),
        cycle=compile_cycle(raw),
        k=k,
    )
    if data.hdec_x.shape[1] != len(data.prob_x) or data.hdec_z.shape[1] != len(data.prob_z):
        raise ValueError("Decoder matrix and channel probability dimensions differ")
    print("<<< Offline decoder data loaded\n")
    return data


class BatchedCircuitSimulator:
    def __init__(self, compiled: CompiledCycle, rng: np.random.Generator):
        self.c = compiled
        self.rng = rng

    def _bernoulli(self, shape: tuple[int, ...], probability: float) -> np.ndarray:
        if probability <= 0:
            return np.zeros(shape, dtype=np.uint8)
        if probability >= 1:
            return np.ones(shape, dtype=np.uint8)
        return (self.rng.random(shape) < probability).astype(np.uint8)

    def _cnot_layer(
        self,
        x: np.ndarray,
        z: np.ndarray,
        layer: tuple[np.ndarray, np.ndarray],
        noisy: bool,
    ) -> None:
        controls, targets = layer
        # Ideal CNOT propagation.  Every control/target occurs once per layer.
        x[:, targets] ^= x[:, controls]
        z[:, controls] ^= z[:, targets]
        if not noisy or noise.cnot <= 0:
            return
        shape = (x.shape[0], len(controls))
        error_type = self.rng.integers(1, 16, size=shape, dtype=np.uint8)
        error_type *= self._bernoulli(shape, noise.cnot)
        x[:, controls] ^= _CNOT_X_CONTROL[error_type]
        z[:, controls] ^= _CNOT_Z_CONTROL[error_type]
        x[:, targets] ^= _CNOT_X_TARGET[error_type]
        z[:, targets] ^= _CNOT_Z_TARGET[error_type]

    def _idle_layer(self, x: np.ndarray, z: np.ndarray, qubits: np.ndarray, noisy: bool) -> None:
        if not noisy or noise.idle <= 0:
            return
        shape = (x.shape[0], len(qubits))
        error_type = self.rng.integers(1, 4, size=shape, dtype=np.uint8)
        error_type *= self._bernoulli(shape, noise.idle)
        x[:, qubits] ^= _IDLE_X[error_type]
        z[:, qubits] ^= _IDLE_Z[error_type]

    def _cycle(
        self,
        x: np.ndarray,
        z: np.ndarray,
        syndrome_x: np.ndarray,
        syndrome_z: np.ndarray,
        cycle_index: int,
        noisy: bool,
    ) -> None:
        c = self.c
        # Round 0: PrepX, Z-check CNOTs, data idles.
        x[:, c.xchecks] = 0
        z[:, c.xchecks] = 0
        if noisy:
            z[:, c.xchecks] ^= self._bernoulli((x.shape[0], c.n2), noise.init)
        self._cnot_layer(x, z, c.cnot_layers[0], noisy)
        self._idle_layer(x, z, c.idle_layers[0], noisy)

        # Rounds 1--5.
        for layer in c.cnot_layers[1:6]:
            self._cnot_layer(x, z, layer, noisy)

        # Round 6: MeasZ, X-check CNOTs, data idles.
        meas_z_error = self._bernoulli((x.shape[0], c.n2), noise.meas) if noisy else 0
        syndrome_x[:, cycle_index] = x[:, c.zchecks] ^ meas_z_error
        self._cnot_layer(x, z, c.cnot_layers[6], noisy)
        self._idle_layer(x, z, c.idle_layers[1], noisy)

        # Round 7: data idles, MeasX, PrepZ.
        self._idle_layer(x, z, c.idle_layers[2], noisy)
        meas_x_error = self._bernoulli((x.shape[0], c.n2), noise.meas) if noisy else 0
        syndrome_z[:, cycle_index] = z[:, c.xchecks] ^ meas_x_error
        x[:, c.zchecks] = 0
        z[:, c.zchecks] = 0
        if noisy:
            x[:, c.zchecks] ^= self._bernoulli((x.shape[0], c.n2), noise.init)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self.c
        total_cycles = N_c + 2
        x = np.zeros((batch_size, c.state_size), dtype=np.uint8)
        z = np.zeros_like(x)
        sx = np.empty((batch_size, total_cycles, c.n2), dtype=np.uint8)
        sz = np.empty_like(sx)
        for cycle_index in range(total_cycles):
            self._cycle(x, z, sx, sz, cycle_index, noisy=cycle_index < N_c)

        # Detection events are differences of consecutive syndrome rounds.
        sx[:, 1:] ^= sx[:, :-1].copy()
        sz[:, 1:] ^= sz[:, :-1].copy()
        return (
            sx.reshape(batch_size, -1),
            sz.reshape(batch_size, -1),
            x[:, c.data],
            z[:, c.data],
        )


def make_decoder(pcm: csc_matrix, probabilities: np.ndarray) -> BpOsdDecoder:
    return BpOsdDecoder(
        pcm,
        # ldpc 2.4.1's Cython constructor currently requires a Python list
        # even though the public documentation also advertises ndarray.
        error_channel=probabilities.tolist(),
        max_iter=bp_osd.max_iter,
        bp_method=bp_osd.bp_method,
        ms_scaling_factor=bp_osd.ms_scaling_factor,
        schedule=bp_osd.schedule,
        omp_thread_count=bp_osd.omp_thread_count,
        osd_method=bp_osd.osd_method,
        osd_order=bp_osd.osd_order,
    )


def logical_from_recovery(logical_map: csc_matrix, recovery: np.ndarray) -> np.ndarray:
    return np.asarray(logical_map @ recovery, dtype=np.uint8).reshape(-1) & 1


def run_monte_carlo(data: DecoderData, num_trials: int, batch_size: int, seed: int | None) -> int:
    if num_trials <= 0 or batch_size <= 0:
        raise ValueError("num_trials and batch_size must be positive")
    print(">>> Start Monte-Carlo")
    decoder_x = make_decoder(data.hdec_x, data.prob_x)
    decoder_z = make_decoder(data.hdec_z, data.prob_z)
    simulator = BatchedCircuitSimulator(data.cycle, np.random.default_rng(seed))
    failures = 0
    completed = 0
    batches = (num_trials + batch_size - 1) // batch_size
    progress = tqdm(total=num_trials, unit="trial", disable=not online.progress)
    try:
        for _ in range(batches):
            current = min(batch_size, num_trials - completed)
            syndrome_x, syndrome_z, final_x, final_z = simulator.sample(current)
            actual_x = (final_x @ data.lz.T) & 1
            actual_z = (final_z @ data.lx.T) & 1

            for i in range(current):
                # Preserve old_decoder_run.py's short-circuit order: decode Z
                # errors first, and decode X errors only if Z correction passed.
                recovery_z = decoder_z.decode(syndrome_z[i])
                guessed_z = logical_from_recovery(data.logical_map_z, recovery_z)
                if not np.array_equal(guessed_z, actual_z[i]):
                    failures += 1
                    continue
                recovery_x = decoder_x.decode(syndrome_x[i])
                guessed_x = logical_from_recovery(data.logical_map_x, recovery_x)
                if not np.array_equal(guessed_x, actual_x[i]):
                    failures += 1

            completed += current
            progress.update(current)
            progress.set_postfix(failures=failures, refresh=False)
    finally:
        progress.close()
    print("<<< Finish Monte-Carlo\n")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="offline pickle (auto-detected by default)")
    parser.add_argument("--trials", type=int, default=None, help="override config.online.num_trials")
    parser.add_argument("--batch-size", type=int, default=None, help="override config.online.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="override config.online.random_seed")
    parser.add_argument("--output", type=Path, default=None, help="result TSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or find_decoder_data(decoder_data_dir)
    data = load_decoder_data(input_path)
    trials = args.trials if args.trials is not None else online.num_trials
    batch_size = args.batch_size if args.batch_size is not None else online.batch_size
    seed = args.seed if args.seed is not None else online.random_seed

    start = time.perf_counter()
    failures = run_monte_carlo(data, trials, batch_size, seed)
    elapsed = time.perf_counter() - start
    logical_error_rate = failures / (trials * N_c)
    print(
        f"p={error_rate}\tcycles={N_c}\ttrials={trials}\tfailures={failures}\t"
        f"logical_error_rate={logical_error_rate:.8e}\telapsed={elapsed:.1f}s"
    )

    n = 2 * code_param.ell * code_param.m
    output = args.output or result_dir / f"code_{n}_{data.k}_{code_distance}.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists()
    with output.open("a", encoding="utf-8") as stream:
        if new_file:
            stream.write("error_rate\tcycles\ttrials\tfailures\tlogical_error_rate\telapsed_seconds\n")
        stream.write(
            f"{error_rate}\t{N_c}\t{trials}\t{failures}\t{logical_error_rate:.12g}\t{elapsed:.6f}\n"
        )
    print(f"Result appended to {output}")


if __name__ == "__main__":
    main()
