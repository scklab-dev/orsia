"""Component-level contraction for declared small columns, not a SOS-2 error bound.

The tolerance floor below accommodates invariant analytic paths. We compare each
last increment to the preceding increment, not to a presumed exact fine grid.
Independent continuous-ray references remain in test_sos_numerics.py.
"""

import numpy as np
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)


@pytest.mark.parametrize("g", [None, -0.8, 0.8])
@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_separate_and_combined_refinement_contracts_each_retained_path(
    g: float | None, rho: float
) -> None:
    axis = np.array([12000.0], dtype=np.float64)
    scene = RadiativeScene(
        Atmosphere(
            np.array([0.0, 1000.0]),
            np.array([1e5, 9e4]),
            np.array([9.5e4]),
            np.array([290.0]),
            np.array([2e29]),
            np.array([1.9e29]),
            np.array([1e28]),
        ),
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([rho])),
        ObservationGeometry(20.0),
    )
    sun = solar_source(
        SpectralIrradiance(wavenumber_grid(axis), np.array([8.0])), direction=(30.0, 70.0)
    )
    scattering = (
        RayleighComponent(wavenumber_grid(axis), np.array([[0.2]]))
        if g is None
        else AerosolComponent(
            wavenumber_grid(axis), np.array([[0.2]]), np.array([[0.1]]), np.array([[g]])
        )
    )
    sequences = {
        "angular": [(4, 64), (8, 64), (16, 64)],
        "vertical": [(16, 4), (16, 16), (16, 64)],
        "combined": [(4, 4), (8, 16), (16, 64)],
    }
    results = {
        grid: terms_support.reflected_radiance(
            scene,
            source=sun,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                sun,
                rayleigh=scattering if isinstance(scattering, RayleighComponent) else None,
                aerosol=scattering if isinstance(scattering, AerosolComponent) else None,
            ),
            angular_order=grid[0],
            layer_subdivisions=grid[1],
        )
        for grid in dict.fromkeys(grid for sequence in sequences.values() for grid in sequence)
    }
    for result in results.values():
        assert all(c.values[0] >= 0 for c in result.components)
        np.testing.assert_allclose(
            result.values, np.sum([c.values for c in result.components], axis=0)
        )
        for component in result.components:
            if component.name == "atmospheric_order_0" or (
                rho == 0 and component.name == "atmospheric_order_1"
            ):
                other = next(
                    c
                    for c in results[16, 64].components
                    if c.name == component.name and c.role == component.role
                )
                np.testing.assert_allclose(component.values, other.values, rtol=3e-13, atol=1e-15)
            if rho == 0 and component.name == "atmospheric_order_0":
                assert component.values == (0.0,)
    for axis_name, sequence in sequences.items():
        a, b, c = (results[grid] for grid in sequence)
        coarse = np.asarray([a.values, *(t.values for t in a.components)])
        middle = np.asarray([b.values, *(t.values for t in b.components)])
        fine = np.asarray([c.values, *(t.values for t in c.components)])
        assert np.all(np.abs(fine - middle) <= np.abs(middle - coarse) + 1e-12), axis_name
