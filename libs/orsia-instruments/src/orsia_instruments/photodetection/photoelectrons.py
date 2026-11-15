"""Calculate expected photoelectrons and their tangents from physical radiance."""

import numpy as np
from numpy.typing import NDArray


def integrate_photoelectrons(
    coefficients: NDArray[np.float64],
    radiance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate expected photoelectrons from a physical response and radiance.

    Args:
        coefficients: Finite nonnegative (read,...integration axes) coefficients,
            with at least one integration axis. Include all physical and quadrature
            factors exactly once, in electrons per unit of radiance density.
        radiance: Finite nonnegative radiance with exactly the coefficient shape,
            in the density units used by the coefficients. Broadcasting is explicit
            at the caller. Empty integration axes contribute zero electrons.

    Returns:
        Fresh float64 (read,) electron means, summed over all integration axes.
        Inputs are borrowed and unchanged. Detector noise and dark current are separate.

    Raises:
        ValueError: Shapes, physical values or integrated counts are invalid.
    """
    if (
        coefficients.ndim < 2
        or radiance.shape != coefficients.shape
        or not np.isfinite(coefficients).all()
        or not np.isfinite(radiance).all()
        or np.any(coefficients < 0)
        or np.any(radiance < 0)
    ):
        raise ValueError("aligned finite nonnegative coefficients and radiance required")
    values = np.sum(coefficients * radiance, axis=tuple(range(1, coefficients.ndim)))
    if not np.isfinite(values).all():
        raise ValueError("photoelectron integration exceeds finite numerical domain")
    return values


def integrate_photoelectron_tangents(
    coefficients: NDArray[np.float64],
    radiance: NDArray[np.float64],
    radiance_tangent: NDArray[np.float64],
    response_tangent: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate expected photoelectrons and signed tangents with a shared response.

    Args:
        coefficients: Finite nonnegative (read,...integration axes) coefficients,
            including all physical and quadrature factors exactly once.
        radiance: Matching finite nonnegative radiance in coefficient density units.
        radiance_tangent: Matching axes plus a final signed direction axis.
        response_tangent: Optional coefficient derivatives with the tangent shape.

    Returns:
        Fresh (read,) electron means and (read,direction) derivatives of K*L,
        including K*dL and dK*L. Inputs are borrowed and unchanged.

    Raises:
        ValueError: Physical domains, shapes or finite products fail.
    """
    if (
        radiance_tangent.ndim != coefficients.ndim + 1
        or radiance_tangent.shape[:-1] != coefficients.shape
        or not np.isfinite(radiance_tangent).all()
    ):
        raise ValueError("aligned physical coefficients, radiance and signed directions required")
    axes = tuple(range(1, coefficients.ndim))
    values = integrate_photoelectrons(coefficients, radiance)
    derivative = np.sum(coefficients[..., None] * radiance_tangent, axis=axes)
    if response_tangent is not None:
        if (
            response_tangent.shape != radiance_tangent.shape
            or not np.isfinite(response_tangent).all()
        ):
            raise ValueError("response directions must match radiance direction axes")
        derivative += np.sum(response_tangent * radiance[..., None], axis=axes)
    if not np.isfinite(derivative).all():
        raise ValueError("electron contraction exceeds finite numerical domain")
    return values, derivative
