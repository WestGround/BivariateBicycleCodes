"""Build the circuit-level effective noise model for a BB code.

The main optimization is to avoid constructing and simulating one complete
copy of the circuit per single fault.  A Clifford circuit is linear over each
of its X and Z components.  We therefore sweep the circuit backwards once and
read the response columns at every possible fault-insertion boundary.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from bposd.css import css_code
from scipy.sparse import csc_matrix, issparse
from tqdm import tqdm

from config import BBParameters, NoiseParameters, normalized_a_params, normalized_b_params
from config import code_param, sched_x, sched_z, N_c, error_rate, noise
from stim_backend import build_stim_circuit


Gate = tuple
FaultQubits = tuple[int, ...]


def gf2_rank(matrix: np.ndarray) -> int:
    """Return the rank of a binary matrix without constructing its nullspace."""
    a = np.asarray(matrix, dtype=np.uint8).copy()
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        other = np.flatnonzero(a[:, col])
        other = other[other != rank]
        a[other] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def shift_matrices(size: int) -> list[np.ndarray]:
    identity = np.eye(size, dtype=np.uint8)
    return [np.roll(identity, shift, axis=1) for shift in range(size)]


def monomial_matrix(
    spec: tuple[str, int],
    *,
    x_shifts: Sequence[np.ndarray],
    y_shifts: Sequence[np.ndarray],
) -> np.ndarray:
    axis, shift = spec
    if axis == "x":
        return x_shifts[shift % len(x_shifts)]
    if axis == "y":
        return y_shifts[shift % len(y_shifts)]
    raise AssertionError(f"Unexpected monomial axis {axis!r}")


def build_code(params: BBParameters):
    """ Construct PCM (hx, hz) and css_code instance from l, m and A/B terms.

        Return each permutation-matrix term list and code dimension k as well.
    """
    print(">>> Start building PCM from parameters")
    ell, m = params.ell, params.m
    i_ell = np.eye(ell, dtype=np.uint8)
    i_m = np.eye(m, dtype=np.uint8)
    x = [np.kron(s, i_m) for s in shift_matrices(ell)]
    y = [np.kron(i_ell, s) for s in shift_matrices(m)]

    a_specs = normalized_a_params(params)
    b_specs = normalized_b_params(params)
    a_terms = tuple(monomial_matrix(spec, x_shifts=x, y_shifts=y) for spec in a_specs)
    b_terms = tuple(monomial_matrix(spec, x_shifts=x, y_shifts=y) for spec in b_specs)
    a = np.bitwise_xor.reduce(a_terms)
    b = np.bitwise_xor.reduce(b_terms)
    hx = np.hstack((a, b)).astype(np.uint8, copy=False)
    hz = np.hstack((b.T, a.T)).astype(np.uint8, copy=False)

    n = 2 * ell * m
    k = n - gf2_rank(hx) - gf2_rank(hz)
    code = css_code(hx, hz)

    print(">>> Testing CSS code")
    code.test()
    print(f"<<< Done ([[{n}, {k}]], A terms={a_specs}, B terms={b_specs})")
    print("<<< Complete building PCM from parameter\n")
    return hx, hz, code, a_terms, b_terms, k, a_specs, b_specs


def one_position(row: np.ndarray) -> int:
    positions = np.flatnonzero(row)
    if len(positions) != 1:
        raise ValueError("Each BB monomial must have exactly one nonzero per row")
    return int(positions[0])


def build_tanner_graph(
    a_terms: Sequence[np.ndarray], b_terms: Sequence[np.ndarray]
) -> tuple[list[list[int]], list[list[int]]]:
    """ Returns the tanner graph, which is the list of list.
        tanner_x/z[i] is the list of indexes of data qubits checked by ith x/z check. 
    """
    print(">>> Start building tanner graph")
    n2 = a_terms[0].shape[0]
    tanner_x: list[list[int]] = []
    tanner_z: list[list[int]] = []
    for check in range(n2):
        # oneposition(term[check]) denotes the index of 1 in each row of term(=ai or bi).
        # Accordingly, tanner_x[i] is the list of "indexes of data qubits" checked by ith X-check.
        # Similar for tanner_z.
        tanner_x.append(
            [one_position(term[check]) for term in a_terms]
            + [n2 + one_position(term[check]) for term in b_terms]
        )
        tanner_z.append(
            [one_position(term[:, check]) for term in b_terms]
            + [n2 + one_position(term[:, check]) for term in a_terms]
        )
    print("<<< Complete building tanner graph\n")
    return tanner_x, tanner_z


def build_named_qubits(n2: int):
    """ Return lists of (name, index) for X chcecks, data, Z checks,
        as well as dict of {name: linear_index} for entire qubits.

        x/zchecks, data_qubits: lists of (name, i_from_0) for each type
        lin_order: dict of (name, lin_order)
    """
    xchecks = [("Xcheck", i) for i in range(n2)]
    data_qubits = [("data_left", i) for i in range(n2)] + [
        ("data_right", i) for i in range(n2)
    ]
    zchecks = [("Zcheck", i) for i in range(n2)]
    all_qubits = xchecks + data_qubits + zchecks
    return xchecks, data_qubits, zchecks, {q: i for i, q in enumerate(all_qubits)}


def build_cycle(
    n2: int,
    tanner_x: Sequence[Sequence[int]],
    tanner_z: Sequence[Sequence[int]],
    sched_x: Sequence[int | str],
    sched_z: Sequence[int | str],
) -> tuple[list[Gate], dict, list, list, list]:
    """Construct one scheduled syndrome-extraction cycle.

    ``sched_x`` and ``sched_z`` are explicit timestep schedules.  At each
    timestep, the X side may be one of ``PrepX``, ``MeasX``, ``idle``, or an
    integer Tanner-edge index.  The Z side is analogous with ``PrepZ`` and
    ``MeasZ``.  The two schedules must have the same length; that length is
    the circuit depth of one QEC cycle.
    """
    print(">>> Start build scheduled syndrome extraction cycle")
    check_weight = len(tanner_x[0])
    if check_weight != len(tanner_z[0]):
        raise ValueError("X and Z Tanner graph check weights differ")
    if len(sched_x) != len(sched_z):
        raise ValueError(
            f"X/Z schedule depths differ: len(sched_x)={len(sched_x)}, "
            f"len(sched_z)={len(sched_z)}"
        )
    xchecks, data, zchecks, lin_order = build_named_qubits(n2)
    cycle: list[Gate] = []

    x_special = {"PrepX", "MeasX", "idle"}
    z_special = {"PrepZ", "MeasZ", "idle"}
    x_counts = {"PrepX": 0, "MeasX": 0}
    z_counts = {"PrepZ": 0, "MeasZ": 0}

    def normalize_op(op: int | str, allowed: set[str], side: str, t: int) -> int | str:
        if isinstance(op, str):
            value = op.strip()
            if value not in allowed:
                raise ValueError(f"Invalid {side} schedule op {op!r} at timestep {t}")
            return value
        direction = int(op)
        if not 0 <= direction < check_weight:
            raise ValueError(
                f"{side} CNOT direction {direction} at timestep {t} is outside "
                f"[0, {check_weight})"
            )
        return direction

    for timestep, (x_op_raw, z_op_raw) in enumerate(zip(sched_x, sched_z)):
        x_op = normalize_op(x_op_raw, x_special, "X", timestep)
        z_op = normalize_op(z_op_raw, z_special, "Z", timestep)
        data_used: set = set()
        timestep_ops: list[Gate] = []

        # Resets and measurements are scheduled explicitly.
        if x_op == "PrepX":
            timestep_ops.extend(("PrepX", q) for q in xchecks)
            x_counts["PrepX"] += 1
        elif x_op == "MeasX":
            timestep_ops.extend(("MeasX", q) for q in xchecks)
            x_counts["MeasX"] += 1

        if z_op == "PrepZ":
            timestep_ops.extend(("PrepZ", q) for q in zchecks)
            z_counts["PrepZ"] += 1
        elif z_op == "MeasZ":
            timestep_ops.extend(("MeasZ", q) for q in zchecks)
            z_counts["MeasZ"] += 1

        # CNOTs in the same timestep must be a parallel layer.  In particular,
        # an X-check and a Z-check cannot touch the same data qubit here.
        if isinstance(x_op, int):
            for check, control in enumerate(xchecks):
                target = data[tanner_x[check][x_op]]
                if target in data_used:
                    raise ValueError(
                        f"Non-parallel X CNOT layer at timestep {timestep}: "
                        f"data qubit {target} is reused"
                    )
                data_used.add(target)
                timestep_ops.append(("CNOT", control, target))
        if isinstance(z_op, int):
            for check, target in enumerate(zchecks):
                control = data[tanner_z[check][z_op]]
                if control in data_used:
                    raise ValueError(
                        f"Non-parallel X/Z CNOT schedule at timestep {timestep}: "
                        f"data qubit {control} is reused"
                    )
                data_used.add(control)
                timestep_ops.append(("CNOT", control, target))

        timestep_ops.extend(("IDLE", q) for q in data if q not in data_used)
        cycle.extend(timestep_ops)
        cycle.append(("TICK",))

    expected_counts = {
        "PrepX": x_counts["PrepX"],
        "MeasX": x_counts["MeasX"],
        "PrepZ": z_counts["PrepZ"],
        "MeasZ": z_counts["MeasZ"],
    }
    bad = {name: count for name, count in expected_counts.items() if count != 1}
    if bad:
        raise ValueError(
            "Each QEC cycle must contain exactly one PrepX, MeasX, PrepZ, and "
            f"MeasZ layer; got {expected_counts}"
        )
    print("<<< Complete build scheduled syndrome extraction cycle\n")
    return cycle, lin_order, data, xchecks, zchecks


def indexed_cycle(cycle: Iterable[Gate], lin_order: dict) -> list[Gate]:
    return [(gate[0], *(lin_order[q] for q in gate[1:])) for gate in cycle]


def enumerate_faults(
    operations: Sequence[Gate], component: str, noise: NoiseParameters
) -> dict[int, list[tuple[FaultQubits, float]]]:
    """Group single-component faults by their insertion boundary."""
    faults: dict[int, list[tuple[FaultQubits, float]]] = defaultdict(list)
    meas, prep = (("MeasZ", "PrepZ") if component == "X" else ("MeasX", "PrepX"))
    for i, gate in enumerate(operations):
        kind = gate[0]
        if kind == meas:  # Measurement error occurs immediately before measurement.
            faults[i].append(((gate[1],), noise.meas))
        if kind == prep:
            faults[i + 1].append(((gate[1],), noise.init))
        elif kind == "IDLE":
            faults[i + 1].append(((gate[1],), 2.0 * noise.idle / 3.0))
        elif kind == "CNOT":
            probability = 4.0 * noise.cnot / 15.0
            faults[i + 1].extend(
                [((gate[1],), probability), ((gate[2],), probability), ((gate[1], gate[2]), probability)]
            )
    return faults


def sparsify_response(response: np.ndarray, cycles: int, n2: int) -> np.ndarray:
    result = response.copy()
    raw = response[: cycles * n2].reshape(cycles, n2)
    sparse = result[: cycles * n2].reshape(cycles, n2)
    sparse[1:] ^= raw[:-1]
    return result


def effective_noise_model(
    component: str,
    cycle_indexed: Sequence[Gate],
    num_cycles: int,
    n2: int,
    logicals: np.ndarray,
    data_indices: np.ndarray,
    noise: NoiseParameters,
):
    """Build augmented and decoding matrices for X-only or Z-only faults."""
    noisy_operations = list(cycle_indexed) * num_cycles
    operations = noisy_operations + list(cycle_indexed) * 2
    faults = enumerate_faults(noisy_operations, component, noise)
    total_faults = sum(map(len, faults.values()))
    measured_kind = "MeasZ" if component == "X" else "MeasX"
    reset_kind = "PrepZ" if component == "X" else "PrepX"
    total_cycles = num_cycles + 2
    syndrome_rows = total_cycles * n2
    logicals = np.asarray(logicals, dtype=np.uint8)
    k = logicals.shape[0]
    output_rows = syndrome_rows + k
    state_size = 4 * n2

    # Attach chronological syndrome-output positions to measurement operations.
    measured_positions: dict[int, int] = {}
    measurement_count = 0
    for i, gate in enumerate(operations):
        if gate[0] == measured_kind:
            measured_positions[i] = measurement_count
            measurement_count += 1
    if measurement_count != syndrome_rows:
        raise AssertionError("Unexpected number of syndrome measurements")

    # response[:, q] is the future output caused by toggling state bit q at
    # the current circuit boundary.
    response = np.zeros((output_rows, state_size), dtype=np.uint8)
    response[syndrome_rows:, data_indices] = logicals
    columns: list[np.ndarray] = []
    probabilities: list[float] = []
    fault_specs: list[tuple[int, FaultQubits]] = []

    def collect(boundary: int) -> None:
        for qubits, probability in faults.get(boundary, ()):
            vector = response[:, qubits[0]].copy()
            for q in qubits[1:]:
                vector ^= response[:, q]
            columns.append(sparsify_response(vector, total_cycles, n2))
            probabilities.append(probability)
            fault_specs.append((boundary, qubits))

    collect(len(operations))
    for i in tqdm(
        range(len(operations) - 1, -1, -1),
        desc=f"Backward propagation ({component})",
        unit="gate",
    ):
        gate = operations[i]
        kind = gate[0]
        if kind == measured_kind:
            response[measured_positions[i], gate[1]] ^= 1
        elif kind == reset_kind:
            response[:, gate[1]] = 0
        elif kind == "CNOT":
            control, target = gate[1], gate[2]
            if component == "X":
                # Forward: target ^= control.  Backward observable update:
                # response(control) ^= response(target).
                response[:, control] ^= response[:, target]
            else:
                # Forward: control ^= target for the Z component.
                response[:, target] ^= response[:, control]
        collect(i)

    if len(columns) != total_faults:
        raise AssertionError(f"Collected {len(columns)} of {total_faults} faults")

    # Cheap regression guard: compare representative response columns with a
    # literal forward simulation.  This catches boundary/preparation/CNOT
    # convention mistakes without reviving the original all-fault bottleneck.
    sample_indices = np.linspace(0, total_faults - 1, min(12, total_faults), dtype=int)
    for sample in np.unique(sample_indices):
        boundary, qubits = fault_specs[int(sample)]
        state = np.zeros(state_size, dtype=np.uint8)
        state[list(qubits)] ^= 1
        direct = np.zeros(output_rows, dtype=np.uint8)
        for i in range(boundary, len(operations)):
            gate = operations[i]
            kind = gate[0]
            if kind == measured_kind:
                direct[measured_positions[i]] = state[gate[1]]
            elif kind == reset_kind:
                state[gate[1]] = 0
            elif kind == "CNOT":
                control, target = gate[1], gate[2]
                if component == "X":
                    state[target] ^= state[control]
                else:
                    state[control] ^= state[target]
        direct[syndrome_rows:] = (logicals @ state[data_indices]) & 1
        direct = sparsify_response(direct, total_cycles, n2)
        if not np.array_equal(direct, columns[int(sample)]):
            raise AssertionError(
                f"Backward response failed direct simulation for {component} fault {sample}"
            )

    # Preserve first-occurrence ordering while merging identical columns.
    merged: dict[bytes, tuple[np.ndarray, float]] = {}
    for column, probability in zip(columns, probabilities):
        key = np.packbits(column, bitorder="little").tobytes()
        if key in merged:
            old_column, old_probability = merged[key]
            merged[key] = old_column, old_probability + probability
        else:
            merged[key] = column, probability

    dense = np.column_stack([entry[0] for entry in merged.values()]).astype(np.uint8)
    augmented = csc_matrix(dense)
    decoding = augmented[:syndrome_rows].tocsc()
    channel_probabilities = [entry[1] for entry in merged.values()]
    print(
        f"{component}: {total_faults} faults -> {augmented.shape[1]} distinct columns; "
        f"max column weight={int(decoding.getnnz(axis=0).max())}, "
        f"max row weight={int(decoding.getnnz(axis=1).max())}"
    )
    return augmented, decoding, channel_probabilities, syndrome_rows


def build_decoder_data(
    params: BBParameters,
    num_cycles: int,
    nominal_error_rate: float,
    noise_params: NoiseParameters,
    sched_x: Sequence[int | str],
    sched_z: Sequence[int | str],
) -> dict:
    hx, hz, code, a_terms, b_terms, k, a_specs, b_specs = build_code(params)
    n2 = params.ell * params.m
    tanner_x, tanner_z = build_tanner_graph(a_terms, b_terms)
    cycle, lin_order, data, xchecks, zchecks = build_cycle(
        n2, tanner_x, tanner_z, sched_x, sched_z
    )
    cycle_i = indexed_cycle(cycle, lin_order)
    data_indices = np.fromiter((lin_order[q] for q in data), dtype=np.int64)
    lx = np.asarray(code.lx.toarray() if issparse(code.lx) else code.lx, dtype=np.uint8)
    lz = np.asarray(code.lz.toarray() if issparse(code.lz) else code.lz, dtype=np.uint8)
    # cycle: List of ('CNOT', ('data_right', 18), ('Zcheck', 0))
    # cycle_i: List of ('CNOT', 90, 108) (=('CNOT', 36+36+18, 36+36+36+0))
    # lx, lz: numpy.ndarray type

    hx_aug, hdec_x, prob_x, first_x = effective_noise_model(
        "X", cycle_i, num_cycles, n2, lz, data_indices, noise_params
    )
    hz_aug, hdec_z, prob_z, first_z = effective_noise_model(
        "Z", cycle_i, num_cycles, n2, lx, data_indices, noise_params
    )
    stim_data = build_stim_circuit(
        cycle=cycle,
        lin_order=lin_order,
        data_qubits=data,
        xchecks=xchecks,
        zchecks=zchecks,
        num_cycles=num_cycles,
        noise=noise_params,
        hx=hx,
        hz=hz,
        lz=lz,
    )

    return {
        "HdecX": hdec_x,
        "HdecZ": hdec_z,
        "probX": prob_x,
        "probZ": prob_z,
        "cycle": cycle,
        "lin_order": lin_order,
        "num_cycles": num_cycles,
        "data_qubits": data,
        "Xchecks": xchecks,
        "Zchecks": zchecks,
        "HX": hx_aug,
        "HZ": hz_aug,
        "lx": lx,
        "lz": lz,
        "first_logical_rowZ": first_z,
        "first_logical_rowX": first_x,
        "ell": params.ell,
        "m": params.m,
        "format_version": 2,
        "a_params": list(a_specs),
        "b_params": list(b_specs),
        "check_weight": len(a_terms) + len(b_terms),
        "cycle_depth": len(sched_x),
        "error_rate": nominal_error_rate,
        "noise": {
            "init": noise_params.init,
            "idle": noise_params.idle,
            "cnot": noise_params.cnot,
            "meas": noise_params.meas,
        },
        "sX": list(sched_x),
        "sZ": list(sched_z),
        "k": k,
        "hx": hx,
        "hz": hz,
        # Text is used instead of pickling a version-specific extension type.
        "stim_circuit": str(stim_data.circuit),
        "stim_version": __import__("stim").__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    params = code_param
    data = build_decoder_data(params, N_c, error_rate, noise, sched_x, sched_z)
    n = 2 * params.ell * params.m

    args = parse_args()
    check_weight = data["check_weight"]
    output = args.output or Path("TMP") / (
        f"decoder_data_n{n}_k{data['k']}_p_{error_rate}_cycles_{N_c}_w{check_weight}_d{data['cycle_depth']}.pkl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done")


if __name__ == "__main__":
    main()
