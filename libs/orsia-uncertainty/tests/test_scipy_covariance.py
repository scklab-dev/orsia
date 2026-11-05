"""Selected spectra and opt-in Cholesky retain covariance contracts."""

from unittest.mock import patch

import numpy as np
import pytest
from orsia_uncertainty import (
    combine_contributions,
    validate_covariance,
)
from orsia_uncertainty import covariance as covariance_module


def test_selected_minimum_validates_psd_without_a_full_spectrum() -> None:
    rng = np.random.default_rng(63)
    for size in (1, 4, 32, 127, 128):
        basis, _ = np.linalg.qr(rng.standard_normal((size, size)))
        spectrum = np.linspace(0.01, 1.0, size)
        matrix = (basis * spectrum) @ basis.T
        saved = matrix.copy()
        with patch.object(covariance_module, "eigvalsh", wraps=covariance_module.eigvalsh) as eigen:
            law = validate_covariance(matrix)
        if size >= 128:
            assert eigen.call_args.kwargs["subset_by_index"] == [0, 0]
        else:
            eigen.assert_not_called()
        np.testing.assert_array_equal(matrix, saved)
        assert law.flags.writeable
        np.testing.assert_allclose(np.linalg.eigvalsh(law), spectrum, atol=3e-15)
        if size > 1:
            spectrum[0] = -1e-3
            with pytest.raises(ValueError, match="positive semidefinite"):
                validate_covariance((basis * spectrum) @ basis.T)
    # The PSD tolerance and singular-law support are not replaced by Cholesky.
    validate_covariance(np.diag([-0.5e-12, 1.0]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance(np.diag([-2e-12, 1.0]))


def test_partial_spectrum_checks_large_joint_law_not_only_the_sum() -> None:
    covariance = validate_covariance(np.ones(64))
    sources = tuple((name, covariance) for name in ("first", "second"))
    cross = {("first", "second"): np.eye(64) * 0.25}
    with patch.object(covariance_module, "eigvalsh", wraps=covariance_module.eigvalsh) as eigen:
        combined = combine_contributions(dict(sources), cross)
    assert eigen.call_count == 1
    assert eigen.call_args.args[0].shape == (128, 128)
    assert eigen.call_args.kwargs["subset_by_index"] == [0, 0]
    np.testing.assert_array_equal(combined, 2.5 * np.eye(64))
    # The sum remains positive while the joint law has negative eigenvalues.
    with pytest.raises(ValueError, match="joint source covariance"):
        combine_contributions(dict(sources), {("first", "second"): 1.1 * np.eye(64)})
