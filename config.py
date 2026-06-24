from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MonomialSpec = int | tuple[str, int]


def normalize_monomial_specs(
    specs: Sequence[MonomialSpec],
    *,
    default_first: str,
    default_rest: str,
) -> tuple[tuple[str, int], ...]:
    """Return explicit ("x"|"y", shift) specs.

    Integer-only specs keep the convention used by the original BB scripts:
    A = x^a1 + y^a2 + ..., B = y^b1 + x^b2 + ...
    """
    normalized: list[tuple[str, int]] = []
    for index, spec in enumerate(specs):
        if isinstance(spec, int):
            axis = default_first if index == 0 else default_rest
            shift = spec
        else:
            axis, shift = spec
        axis = str(axis).lower()
        if axis not in {"x", "y"}:
            raise ValueError(f"Invalid BB monomial axis {axis!r}; use 'x' or 'y'")
        normalized.append((axis, int(shift)))
    if not normalized:
        raise ValueError("Each BB side must contain at least one monomial")
    return tuple(normalized)


def normalized_a_params(params: "BBParameters") -> tuple[tuple[str, int], ...]:
    return normalize_monomial_specs(
        params.a_params, default_first="x", default_rest="y"
    )


def normalized_b_params(params: "BBParameters") -> tuple[tuple[str, int], ...]:
    return normalize_monomial_specs(
        params.b_params, default_first="y", default_rest="x"
    )


@dataclass(frozen=True)
class BBParameters:
    ell: int
    m: int
    # Each entry can be either:
    #   - ("x", shift) or ("y", shift), explicitly selecting the monomial.
    #   - an int, using the old BB convention for convenience:
    #       A: first term is x^shift, remaining terms are y^shift
    #       B: first term is y^shift, remaining terms are x^shift
    a_params: list[MonomialSpec]
    b_params: list[MonomialSpec]


@dataclass(frozen=True)
class NoiseParameters:
    init: float
    idle: float
    cnot: float
    meas: float

    @classmethod
    def depolarizing(cls, p: float) -> "NoiseParameters":
        return cls(init=p, idle=p, cnot=p, meas=p)


@dataclass(frozen=True)
class BpOsdParameters:
    bp_method: str
    max_iter: int
    ms_scaling_factor: float
    schedule: str
    omp_thread_count: int
    osd_method: str
    osd_order: int


@dataclass(frozen=True)
class OnlineParameters:
    num_trials: int
    batch_size: int
    random_seed: int | None
    progress: bool


# [[144, 12, 12]]
GrossParam = BBParameters(
    ell = 12,
    m = 6,
    a_params = [("x", 3), ("y", 1), ("y", 2)],
    b_params = [("y", 3), ("x", 1), ("x", 2)],
)

# [[72, 12, 6]]
HalfGrossParam = BBParameters(
    ell = 6,
    m = 6,
    a_params = [("x", 3), ("y", 1), ("y", 2)],
    b_params = [("y", 3), ("x", 1), ("x", 2)],
)

# [[288, 12, 18]]
TwoGrossParam = BBParameters(
    ell = 12,
    m = 12,
    a_params = [("x", 3), ("y", 2), ("y", 7)],
    b_params = [("y", 3), ("x", 1), ("x", 2)],
)

# [[108, 22, 9]]
Q102Param = BBParameters(
    ell = 54,
    m = 1,
    a_params = [("x", 22), ("x", 26), ("x", 37), ("x", 50)],
    b_params = [("x", 19), ("x", 28), ("x", 29), ("x", 35)]
)

#sched_x: list[int | str] = ["PrepX", 1, 4, 3, 5, 0, 2, "MeasX"]
#sched_z: list[int | str] = [3, 5, 0, 1, 2, 4, "MeasZ", "PrepZ"]
sched_x: list[int | str] = ["PrepX", 4, 0, 2, 5, 6, 3, 1, 7, "MeasX"]
sched_z: list[int | str] = ["PrepZ", 3, 5, 7, 2, 1, 6, 4, 0, "MeasZ"]
N_c = 9

error_rate = 0.001
noise = NoiseParameters.depolarizing(error_rate)

# Select the code used by both offline and online scripts.
code_param = Q102Param
code_distance = 9

# BP-OSD parameters use the ldpc v2 spelling.
bp_osd = BpOsdParameters(
    bp_method="minimum_sum",
    max_iter=30,
    ms_scaling_factor=1.0,
    schedule="parallel",
    omp_thread_count=1,
    osd_method="osd_cs",
    osd_order=10,
)

online = OnlineParameters(
    num_trials=10_000,
    batch_size=2_048,
    random_seed=None,
    progress=True,
)

decoder_data_dir = Path("TMP")
result_dir = Path("results")
