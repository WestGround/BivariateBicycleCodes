"""Pluggable recovery decoders used by ``decoder_online.py``.

The Monte-Carlo code only needs a decoder that maps a binary syndrome to a
binary recovery vector over the columns of the offline effective-noise model.
Keeping that interface small makes it easy to add new algorithms without
touching the simulator loop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from ldpc import BpOsdDecoder
from scipy.sparse import csc_matrix

from config import BeamSearchParameters, BpOsdParameters


class RecoveryDecoder(Protocol):
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Return a binary recovery vector for ``syndrome``."""


@dataclass(frozen=True)
class DecoderBuildContext:
    pcm: csc_matrix
    probabilities: np.ndarray
    component: str


class BpOsdRecoveryDecoder:
    def __init__(self, context: DecoderBuildContext, params: BpOsdParameters):
        self._decoder = BpOsdDecoder(
            context.pcm,
            # ldpc 2.4.1's Cython constructor requires list despite its ndarray docs.
            error_channel=context.probabilities.tolist(),
            max_iter=params.max_iter,
            bp_method=params.bp_method,
            ms_scaling_factor=params.ms_scaling_factor,
            schedule=params.schedule,
            omp_thread_count=params.omp_thread_count,
            osd_method=params.osd_method,
            osd_order=params.osd_order,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return np.asarray(self._decoder.decode(syndrome), dtype=np.uint8)


def _import_beam_search_decoder():
    try:
        from beam_search_decoder import BeamSearchDecoder

        return BeamSearchDecoder
    except ImportError as first_error:
        vendor_decoder_dir = (
            Path(__file__).resolve().parent / "vendor" / "BeamSearchDecoder" / "decoder"
        )
        if str(vendor_decoder_dir) not in sys.path:
            sys.path.insert(0, str(vendor_decoder_dir))
        try:
            from beam_search_decoder import BeamSearchDecoder

            return BeamSearchDecoder
        except ImportError as second_error:
            raise ImportError(
                "Beam search decoder is selected, but its Cython extension is not "
                "importable. Build the vendored extension with:\n"
                "  cd BivariateBicycleCodes/vendor/BeamSearchDecoder/decoder\n"
                "  python setup.py build_ext --inplace\n"
                "or install an importable `beam_search_decoder` package."
            ) from second_error or first_error


class BeamSearchRecoveryDecoder:
    def __init__(self, context: DecoderBuildContext, params: BeamSearchParameters):
        BeamSearchDecoder = _import_beam_search_decoder()
        self._decoder = BeamSearchDecoder(
            pcm=context.pcm,
            error_channel=context.probabilities.tolist(),
            max_rounds=params.max_rounds,
            beam_width=params.beam_width,
            num_results=params.num_results,
            initial_iters=params.initial_iters,
            iters_per_round=params.iters_per_round,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return np.asarray(self._decoder.decode(syndrome), dtype=np.uint8)


def make_recovery_decoder(
    name: str,
    context: DecoderBuildContext,
    *,
    bp_osd: BpOsdParameters,
    beam_search: BeamSearchParameters,
) -> RecoveryDecoder:
    normalized = name.lower().replace("-", "_")
    if normalized in {"bp_osd", "bposd"}:
        return BpOsdRecoveryDecoder(context, bp_osd)
    if normalized in {"beam_search", "beamsearch", "beam"}:
        return BeamSearchRecoveryDecoder(context, beam_search)
    raise ValueError(
        f"Unknown decoder backend {name!r}. Supported values are 'bp_osd' and "
        "'beam_search'."
    )
