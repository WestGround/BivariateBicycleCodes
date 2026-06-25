"""Stim-backed circuit-level Monte Carlo decoder for BB quantum codes."""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
from scipy.sparse import csc_matrix
from tqdm import tqdm

from code_parameters import normalized_a_params, normalized_b_params
from config import (
    N_c,
    beam_search,
    bp_osd,
    code_distance,
    code_param,
    decoder_data_dir,
    decoder_backend,
    error_rate,
    noise,
    online,
    result_dir,
)
from decoder_backends import DecoderBuildContext, make_recovery_decoder
from stim_backend import StimBatchSimulator, StimCircuitData, build_stim_circuit


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
    stim_data: StimCircuitData
    k: int


def validate_pickle(raw: dict) -> None:
    def comparable_params(specs):
        return [tuple(item) if isinstance(item, (list, tuple)) else item for item in specs]

    expected = {
        "format_version": 2,
        "num_cycles": N_c,
        "ell": code_param.ell,
        "m": code_param.m,
        "a_params": list(normalized_a_params(code_param)),
        "b_params": list(normalized_b_params(code_param)),
        "cycle_depth": len(code_param.sched_x),
        "error_rate": error_rate,
        "sX": list(code_param.sched_x),
        "sZ": list(code_param.sched_z),
    }
    mismatches = [
        (
            f"{key}: file="
            f"{comparable_params(raw.get(key, [])) if key in {'a_params', 'b_params'} else raw.get(key)!r}, "
            f"config={value!r}"
        )
        for key, value in expected.items()
        if (
            comparable_params(raw.get(key, []))
            if key in {"a_params", "b_params"}
            else raw.get(key)
        )
        != value
    ]
    expected_noise = {
        "init": noise.init,
        "idle": noise.idle,
        "cnot": noise.cnot,
        "meas": noise.meas,
    }
    file_noise = raw.get("noise")
    if file_noise is None:  # Compatibility with older all-depolarizing pickle.
        file_noise = {key: raw.get("error_rate") for key in expected_noise}
    if file_noise != expected_noise:
        mismatches.append(f"noise: file={file_noise!r}, config={expected_noise!r}")
    if mismatches:
        raise ValueError(
            "Offline data do not match config.py:\n  " + "\n  ".join(mismatches)
        )


def find_decoder_data(directory: Path) -> Path:
    n = 2 * code_param.ell * code_param.m
    check_weight = len(normalized_a_params(code_param)) + len(normalized_b_params(code_param))
    pattern = (
        f"decoder_data_n{n}_*_p_{error_rate}_cycles_{N_c}_"
        f"w{check_weight}_d{len(code_param.sched_x)}.pkl"
    )
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one offline-data file matching "
            f"{directory / pattern}, found {len(matches)}"
        )
    return matches[0]


def load_decoder_data(path: Path) -> DecoderData:
    print(f">>> Loading offline decoder data: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    validate_pickle(raw)

    stim_data = build_stim_circuit(
        cycle=raw["cycle"],
        lin_order=raw["lin_order"],
        data_qubits=raw["data_qubits"],
        xchecks=raw["Xchecks"],
        zchecks=raw["Zchecks"],
        num_cycles=N_c,
        noise=noise,
        hx=raw.get("hx"),
        hz=raw.get("hz"),
        lz=raw.get("lz") if "hx" in raw and "hz" in raw else None,
    )
    if "stim_circuit" in raw:
        stored = stim.Circuit(raw["stim_circuit"])
        if stored != stim_data.circuit:
            raise ValueError("Stored Stim circuit differs from current config/circuit")

    k = int(raw["lx"].shape[0])
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
        stim_data=stim_data,
        k=k,
    )
    if data.hdec_x.shape[1] != len(data.prob_x):
        raise ValueError("X decoder matrix and channel probabilities differ")
    if data.hdec_z.shape[1] != len(data.prob_z):
        raise ValueError("Z decoder matrix and channel probabilities differ")
    print(
        f"<<< Loaded Stim {stim.__version__} circuit: "
        f"{stim_data.circuit.num_qubits} qubits, "
        f"{stim_data.circuit.num_detectors} detectors\n"
    )
    return data


def logical_from_recovery(
    logical_map: csc_matrix, recovery: np.ndarray
) -> np.ndarray:
    return np.asarray(logical_map @ recovery, dtype=np.uint8).reshape(-1) & 1


def run_monte_carlo(
    data: DecoderData,
    num_trials: int,
    batch_size: int,
    seed: int | None,
    backend_name: str,
) -> int:
    if num_trials <= 0 or batch_size <= 0:
        raise ValueError("num_trials and batch_size must be positive")
    print(f">>> Start Stim Monte Carlo ({backend_name})")
    decoder_x = make_recovery_decoder(
        backend_name,
        DecoderBuildContext(data.hdec_x, data.prob_x, "X"),
        bp_osd=bp_osd,
        beam_search=beam_search,
    )
    decoder_z = make_recovery_decoder(
        backend_name,
        DecoderBuildContext(data.hdec_z, data.prob_z, "Z"),
        bp_osd=bp_osd,
        beam_search=beam_search,
    )
    simulator = StimBatchSimulator(
        data.stim_data, batch_size=batch_size, seed=seed
    )
    failures = 0
    completed = 0
    progress = tqdm(total=num_trials, unit="trial", disable=not online.progress)
    progress.set_postfix(failures=failures, refresh=True)
    try:
        while completed < num_trials:
            current = min(batch_size, num_trials - completed)
            syndrome_x, syndrome_z, final_x, final_z = simulator.sample(current)
            actual_x = (final_x @ data.lz.T) & 1
            actual_z = (final_z @ data.lx.T) & 1

            for i in range(current):
                recovery_z = decoder_z.decode(syndrome_z[i])
                guessed_z = logical_from_recovery(data.logical_map_z, recovery_z)
                failed = not np.array_equal(guessed_z, actual_z[i])
                if not failed:
                    recovery_x = decoder_x.decode(syndrome_x[i])
                    guessed_x = logical_from_recovery(data.logical_map_x, recovery_x)
                    failed = not np.array_equal(guessed_x, actual_x[i])
                if failed:
                    failures += 1
                    # Failures are rare, so force an immediate redraw when one
                    # occurs instead of waiting for tqdm's next timed refresh.
                    progress.set_postfix(failures=failures, refresh=True)
                completed += 1
                progress.update(1)
    finally:
        progress.close()
    print("<<< Finish Stim Monte Carlo\n")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=None, help="offline pickle (auto-detected)"
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--decoder",
        choices=("bp_osd", "beam_search"),
        default=None,
        help="override config.decoder_backend",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--write-stim-circuit",
        type=Path,
        default=None,
        help="write the generated detector-annotated .stim circuit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or find_decoder_data(decoder_data_dir)
    data = load_decoder_data(input_path)
    if args.write_stim_circuit is not None:
        args.write_stim_circuit.parent.mkdir(parents=True, exist_ok=True)
        args.write_stim_circuit.write_text(str(data.stim_data.circuit), encoding="utf-8")

    trials = args.trials if args.trials is not None else online.num_trials
    batch_size = args.batch_size if args.batch_size is not None else online.batch_size
    seed = args.seed if args.seed is not None else online.random_seed
    backend_name = args.decoder if args.decoder is not None else decoder_backend

    # Main Monte-Carlo execution
    start = time.perf_counter()
    failures = run_monte_carlo(data, trials, batch_size, seed, backend_name)
    elapsed = time.perf_counter() - start

    logical_error_rate = failures / (trials * N_c)
    print(
        f"decoder={backend_name}\tp={error_rate}\tcycles={N_c}\t"
        f"trials={trials}\tfailures={failures}\t"
        f"logical_error_rate={logical_error_rate:.8e}\telapsed={elapsed:.1f}s"
    )

    # Write result
    n = 2 * code_param.ell * code_param.m
    output = args.output or result_dir / f"code_{n}_{data.k}_{code_distance}.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists()
    with output.open("a", encoding="utf-8") as stream:
        if new_file:
            stream.write(
                "decoder\terror_rate\tcycles\ttrials\tfailures\t"
                "logical_error_rate\telapsed_seconds\n"
            )
        stream.write(
            f"{backend_name}\t{error_rate}\t{N_c}\t{trials}\t{failures}\t"
            f"{logical_error_rate:.12g}\t{elapsed:.6f}\n"
        )
    print(f"Result appended to {output}")


if __name__ == "__main__":
    main()
