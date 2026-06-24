"""Bivariate-bicycle code parameter presets.

This module intentionally contains only code-dependent data: the BB polynomial
terms and the syndrome-extraction schedule for that code.  Runtime choices
such as physical error rate, number of QEC cycles, BP-OSD settings, and output
paths stay in ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


MonomialSpec = int | tuple[str, int]
ScheduleOp = int | str


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
    # Explicit one-cycle schedules.  Each entry is a timestep operation:
    #   X side: "PrepX", "MeasX", "idle", or an integer Tanner-edge index.
    #   Z side: "PrepZ", "MeasZ", "idle", or an integer Tanner-edge index.
    sched_x: list[ScheduleOp]
    sched_z: list[ScheduleOp]

    def __post_init__(self) -> None:
        if len(self.sched_x) != len(self.sched_z):
            raise ValueError(
                f"X/Z schedule depths differ: {len(self.sched_x)} != {len(self.sched_z)}"
            )


def normalized_a_params(params: BBParameters) -> tuple[tuple[str, int], ...]:
    return normalize_monomial_specs(
        params.a_params, default_first="x", default_rest="y"
    )


def normalized_b_params(params: BBParameters) -> tuple[tuple[str, int], ...]:
    return normalize_monomial_specs(
        params.b_params, default_first="y", default_rest="x"
    )


GrossScheduleX: list[ScheduleOp] = ["PrepX", 1, 4, 3, 5, 0, 2, "MeasX"]
GrossScheduleZ: list[ScheduleOp] = [3, 5, 0, 1, 2, 4, "MeasZ", "PrepZ"]

### 1. IBM Papers
# [[144, 12, 12]]
GrossParam = BBParameters(
    ell=12,
    m=6,
    a_params=[("x", 3), ("y", 1), ("y", 2)],
    b_params=[("y", 3), ("x", 1), ("x", 2)],
    sched_x=GrossScheduleX,
    sched_z=GrossScheduleZ,
)

# [[72, 12, 6]]
HalfGrossParam = BBParameters(
    ell=6,
    m=6,
    a_params=[("x", 3), ("y", 1), ("y", 2)],
    b_params=[("y", 3), ("x", 1), ("x", 2)],
    sched_x=GrossScheduleX,
    sched_z=GrossScheduleZ,
)

# [[288, 12, 18]]
TwoGrossParam = BBParameters(
    ell=12,
    m=12,
    a_params=[("x", 3), ("y", 2), ("y", 7)],
    b_params=[("y", 3), ("x", 1), ("x", 2)],
    sched_x=GrossScheduleX,
    sched_z=GrossScheduleZ,
)

### 2. Walking cat papers
# [[108, 22, 9]]
Q102Param = BBParameters(
    ell=51,
    m=1,
    a_params=[("x", 22), ("x", 26), ("x", 37), ("x", 50)],
    b_params=[("x", 19), ("x", 28), ("x", 29), ("x", 35)],
    sched_x=["PrepX", 4, 0, 2, 5, 6, 3, 1, 7, "MeasX"],
    sched_z=["PrepZ", 3, 5, 7, 2, 1, 6, 4, 0, "MeasZ"],
)
