"""Physical composition contracts independent of the synthetic camera example."""

from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import GaussianPuffField, GaussianPuffs, puff_path_columns
from orsia_geometry import segment_layer_paths
from orsia_scene import puff_molecular_columns, surface_radiance, surface_radiance_tangent
from scipy.constants import Avogadro


def field_inputs(k=3) -> dict[str, Any]:
    plume = GaussianPuffs(
        0.0,
        np.array([0.0]),
        np.array([2.0]),
        np.array([[30.0, 10.0, 600.0]]),
        np.array([np.diag([100.0**2, 120.0**2, 150.0**2])]),
    )
    return dict(
        height_edges_m=np.array([0.0, 400.0, 1000.0]),
        background_columns_molecules_m2=np.array([[1e23, 2e22], [3e23, 5e22]]),
        plumes=(GaussianPuffField(plume, "free"),) * 3,
        species_indices=np.array([0, 1, 0], dtype=np.int64),
        molar_masses_kg_mol=np.array([0.044, 0.030, 0.044]),
        scales=np.array([0.4, 0.7, 0.2]),
        scale_tangent=np.eye(3)[:, :k],
    )


def transfer_inputs(mode="heterogeneous_direct_solar", k=3) -> dict[str, Any]:
    field = field_inputs(k)
    return dict(
        height_m=250.0,
        slopes=np.zeros(2),
        height_edges_m=field["height_edges_m"],
        solar_direction=np.array([0.6, 0.0, 0.8]),
        columns=partial(puff_molecular_columns, **field),
        cross_sections_m2=np.array(
            [
                [[1e-24, 2e-24], [3e-24, 1e-24]],
                [[2e-24, 3e-24], [4e-24, 2e-24]],
            ]
        ),
        reflectance=np.full((2, 2), 0.3),
        reflectance_tangent=np.full((2, 2, k), 0.02),
        beam_normal_irradiance=np.full((2, 2), 0.04),
        mode=mode,
        minimum_upward_cosine=0.2,
        horizontal_bounds_m=(-5000.0, 5000.0, -5000.0, 5000.0),
    )


def queries():
    return np.array([[0.0, 0.0], [50.0, -20.0]]), np.array([[0.0, 0.6, 0.8], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("empty", [False, True])
def test_amounts_only_plume_columns_equal_tangent_primal(empty):
    field = field_inputs()
    starts = np.array([[0.0, 0.0, 250.0], [10.0, 20.0, 400.0]])
    ends = np.array([[0.0, 0.0, 400.0], [30.0, 40.0, 1000.0]])
    layers = np.array([0, 1], dtype=np.int64)
    if empty:
        starts, ends, layers = starts[:0], ends[:0], layers[:0]
    value, _ = puff_molecular_columns(starts, ends, layers, **field)
    field.pop("scale_tangent")
    direct = puff_molecular_columns(starts, ends, layers, **field)
    assert_array_equal(direct, value)
    assert direct.shape == (len(starts), 2) and direct.flags.writeable
    assert_array_equal(
        puff_molecular_columns(starts, ends, layers, **field, scale_tangent=None), value
    )


def radiance_inputs(options):
    physical = options.copy()
    physical.pop("reflectance_tangent")
    physical["columns"] = lambda starts, ends, layers: options["columns"](starts, ends, layers)[0]
    return physical


def test_multispecies_partial_layers_duplicate_species_and_signed_directions():
    field = field_inputs()
    surface = np.array([[0.0, 0.0, 250.0]])
    starts, ends, _, layers = segment_layer_paths(
        surface, np.array([[0.6, 0.0, 0.8]]), field["height_edges_m"], minimum_upward_cosine=0.2
    )
    field["scale_tangent"] = np.array([[1.0, -2.0, 0.0], [0.0, 3.0, -1.0], [1.0, 0.0, 0.5]])
    value, tangent = puff_molecular_columns(starts, ends, layers, **field)
    lengths = np.linalg.norm(ends - starts, axis=1)
    baseline = field["background_columns_molecules_m2"] / np.diff(field["height_edges_m"])[:, None]
    excess = puff_path_columns(field["plumes"][0].state, starts, ends, boundary="free") * Avogadro
    expected = baseline[layers] * lengths[:, None]
    expected[:, 0] += excess / 0.044 * 0.6
    expected[:, 1] += excess / 0.030 * 0.7
    assert_allclose(value, expected, rtol=2e-15)
    assert_allclose(tangent[:, 0], excess[:, None] / 0.044 * np.array([2.0, -2.0, 0.5]))
    assert_allclose(tangent[:, 1], excess[:, None] / 0.030 * np.array([0.0, 3.0, -1.0]))
    assert not np.shares_memory(value, field["background_columns_molecules_m2"])
    assert value.flags.writeable and tangent.flags.writeable
    for k in range(3):
        step = 1e-4
        plus = field | {"scales": field["scales"] + step * field["scale_tangent"][:, k]}
        minus = field | {"scales": field["scales"] - step * field["scale_tangent"][:, k]}
        finite = (
            puff_molecular_columns(starts, ends, layers, **plus)[0]
            - puff_molecular_columns(starts, ends, layers, **minus)[0]
        ) / (2 * step)
        assert_allclose(finite, tangent[..., k], rtol=2e-8, atol=1e10)


@pytest.mark.parametrize("mode", ["independent_column", "heterogeneous_direct_solar"])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_transfer_arbitrary_parameter_axes_finite_difference_and_permutations(mode, k):
    xy, view = queries()
    options = transfer_inputs(mode, k)
    original = options["reflectance"].copy()
    value, tangent = surface_radiance_tangent(xy, view, **options)
    assert_array_equal(surface_radiance(xy, view, **radiance_inputs(options)), value)
    assert tangent.shape == (2, 2, k)
    assert_allclose(options["reflectance"], original, rtol=0, atol=0)
    permutation = options | {
        "reflectance": options["reflectance"][::-1],
        "reflectance_tangent": options["reflectance_tangent"][::-1],
        "beam_normal_irradiance": options["beam_normal_irradiance"][::-1],
    }
    shuffled, derivative = surface_radiance_tangent(xy[::-1], view[::-1], **permutation)
    assert_allclose(shuffled, value[::-1], rtol=1e-14)
    assert_allclose(derivative, tangent[::-1], rtol=1e-14)
    for direction in range(k):
        field = field_inputs(k)
        step = 1e-3
        results = []
        for sign in (1, -1):
            perturbed = field | {
                "scales": field["scales"] + sign * step * field["scale_tangent"][:, direction]
            }
            current = options | {
                "columns": partial(puff_molecular_columns, **perturbed),
                "reflectance": original
                + sign * step * options["reflectance_tangent"][..., direction],
            }
            results.append(surface_radiance(xy, view, **radiance_inputs(current)))
        assert_allclose((results[0] - results[1]) / (2 * step), tangent[..., direction], rtol=2e-8)


def test_homogeneous_formula_empty_plumes_and_empty_queries():
    field = field_inputs(0)
    field.update(
        plumes=(),
        species_indices=np.empty(0, dtype=np.int64),
        molar_masses_kg_mol=np.empty(0),
        scales=np.empty(0),
        scale_tangent=np.empty((0, 0)),
    )
    xy, view = queries()
    options = transfer_inputs(k=0) | {"columns": partial(puff_molecular_columns, **field)}
    amounts = field["background_columns_molecules_m2"] * np.array([150 / 400, 1.0])[:, None]
    vertical = np.einsum("lg,lgs->s", amounts, options["cross_sections_m2"])
    expected = 0.3 * 0.04 * 0.8 / np.pi * np.exp(-vertical * (1 / 0.8 + 1 / view[:, 2, None]))
    for mode in ("independent_column", "heterogeneous_direct_solar"):
        value, tangent = surface_radiance_tangent(xy, view, **(options | {"mode": mode}))
        assert_allclose(
            surface_radiance(xy, view, **radiance_inputs(options | {"mode": mode})),
            expected,
            rtol=2e-15,
        )
        assert_allclose(value, expected, rtol=2e-15)
        assert tangent.shape == (2, 2, 0)
        empty = options | {
            "mode": mode,
            "reflectance": np.empty((0, 2)),
            "reflectance_tangent": np.empty((0, 2, 0)),
            "beam_normal_irradiance": np.empty((0, 2)),
        }
        value, tangent = surface_radiance_tangent(np.empty((0, 2)), np.empty((0, 3)), **empty)
        assert value.shape == (0, 2) and tangent.shape == (0, 2, 0)
        direct = surface_radiance(np.empty((0, 2)), np.empty((0, 3)), **radiance_inputs(empty))
        assert direct.shape == (0, 2)


@pytest.mark.parametrize(
    "mode,slopes",
    [("independent_column", [0.0, 0.0]), ("heterogeneous_direct_solar", [0.07, -0.03])],
)
def test_radiance_only_columns_partial_layers_geometry_and_ownership(mode, slopes):
    xy, view = queries()
    options = radiance_inputs(transfer_inputs(mode)) | {"slopes": np.array(slopes)}
    density = np.array([[1e20, 2e19], [3e20, 5e19]])
    sampled = []

    def columns(starts, ends, layers):
        sampled.append((starts.copy(), ends.copy(), layers.copy()))
        return density[layers] * np.linalg.norm(ends - starts, axis=1)[:, None]

    options["columns"] = columns
    arrays = [xy, view, *(value for value in options.values() if isinstance(value, np.ndarray))]
    saved = [value.copy() for value in arrays]
    value = surface_radiance(xy, view, **options)
    assert len(sampled) == (1 if mode == "independent_column" else 2)
    # Integrate the homogeneous layers independently of the sampled segment output.
    heights = options["height_m"] + xy @ options["slopes"]
    thickness = np.column_stack((400.0 - heights, np.full(2, 600.0)))
    vertical = np.einsum("ql,lg,lgs->qs", thickness, density, options["cross_sections_m2"])
    normal = np.r_[-options["slopes"], 1.0]
    normal /= np.linalg.norm(normal)
    expected = (
        options["reflectance"]
        * options["beam_normal_irradiance"]
        * (normal @ options["solar_direction"])
        / np.pi
        * np.exp(-vertical * (1 / options["solar_direction"][2] + 1 / view[:, 2, None]))
    )
    assert_allclose(value, expected, rtol=3e-15)
    assert value.dtype == np.float64 and value.flags.writeable
    for array, original in zip(arrays, saved, strict=True):
        assert_array_equal(array, original)
        assert not np.shares_memory(value, array)
    permutation = options | {
        "reflectance": options["reflectance"][::-1],
        "beam_normal_irradiance": options["beam_normal_irradiance"][::-1],
    }
    assert_array_equal(surface_radiance(xy[::-1], view[::-1], **permutation), value[::-1])
    tangent_options = options | {
        "columns": lambda a, b, layers: (columns(a, b, layers), np.zeros((len(a), 2, 3))),
        "reflectance_tangent": np.ones((2, 2, 3)),
    }
    assert_array_equal(surface_radiance_tangent(xy, view, **tangent_options)[0], value)


@pytest.mark.parametrize("mode", ["independent_column", "heterogeneous_direct_solar"])
@pytest.mark.parametrize("kind", ["transparent", "opaque", "zero_reflectance"])
def test_direct_radiance_physical_limits(mode, kind):
    options = radiance_inputs(transfer_inputs(mode))
    options["columns"] = lambda a, b, layers: np.full(
        (len(a), 2), 1e30 if kind == "opaque" else 0.0
    )
    if kind == "zero_reflectance":
        options["reflectance"] = np.zeros((2, 2))
    value = surface_radiance(*queries(), **options)
    expected = options["reflectance"] * options["beam_normal_irradiance"] * 0.8 / np.pi
    assert_allclose(value, 0 if kind == "opaque" else expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
@pytest.mark.parametrize("tangent", [False, True])
def test_invalid_column_values(bad, tangent):
    options = transfer_inputs()
    options["columns"] = lambda a, b, layers: (np.full((len(a), 2), bad), np.zeros((len(a), 2, 3)))
    with pytest.raises(ValueError):
        if tangent:
            surface_radiance_tangent(*queries(), **options)
        else:
            surface_radiance(*queries(), **radiance_inputs(options))


def test_direct_callback_errors_propagate():
    def unavailable(starts, ends, layers):
        raise RuntimeError("field unavailable")

    with pytest.raises(RuntimeError, match="field unavailable"):
        surface_radiance(
            *queries(), **(radiance_inputs(transfer_inputs()) | {"columns": unavailable})
        )


@pytest.mark.parametrize(
    "change",
    [
        {"height_edges_m": np.array([0.0, 0.0, 1000.0])},
        {"background_columns_molecules_m2": np.array([[-1.0]])},
        {"species_indices": np.array([0, 2, 0], dtype=np.int64)},
        {"molar_masses_kg_mol": np.array([0.0, 1.0, 1.0])},
        {"scales": np.array([-1.0, 1.0, 1.0])},
        {"scale_tangent": np.ones((3,))},
        {"scale_tangent": np.full((3, 1), np.nan)},
        {"molar_masses_kg_mol": np.full(3, 1e-300)},
    ],
)
def test_invalid_molecular_inputs(change):
    field = field_inputs() | change
    with pytest.raises(ValueError):
        puff_molecular_columns(
            np.array([[0.0, 0.0, 250.0]]),
            np.array([[0.0, 0.0, 400.0]]),
            np.array([0], dtype=np.int64),
            **field,
        )


@pytest.mark.parametrize(
    "starts,ends,layers",
    [
        (np.zeros((1, 2)), np.ones((1, 2)), np.array([0], dtype=np.int64)),
        (np.zeros((1, 3)), np.ones((1, 3)), np.array([2], dtype=np.int64)),
        (np.zeros((1, 3)), np.array([[0.0, 0.0, 600.0]]), np.array([0], dtype=np.int64)),
        (np.array([[0.0, 0.0, -1.0]]), np.zeros((1, 3)), np.array([0], dtype=np.int64)),
    ],
)
def test_invalid_segment_assignments(starts, ends, layers):
    with pytest.raises(ValueError):
        puff_molecular_columns(starts, ends, layers, **field_inputs())


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "thermal"},
        {"mode": "independent_column", "slopes": np.array([0.1, 0.0])},
        {"solar_direction": np.array([0.0, 1.0])},
        {"solar_direction": np.array([0.0, 0.0, -1.0])},
        {"minimum_upward_cosine": 0.0},
        {"minimum_upward_cosine": 0.9},
        {"cross_sections_m2": np.full((2, 2, 2), -1.0)},
        {"cross_sections_m2": np.ones((1, 2, 2))},
        {"height_m": 1000.0},
        {"height_m": -1.0},
        {"horizontal_bounds_m": (-100.0, 100.0, -100.0, 100.0)},
        {"horizontal_bounds_m": (100.0, -100.0, -100.0, 100.0)},
        {"reflectance_tangent": np.zeros((2, 2, 2))},
        {"reflectance": np.full((2, 2), 1.1)},
        {"reflectance": np.ones((2, 1))},
        {"beam_normal_irradiance": np.full((2, 2), np.nan)},
        {"beam_normal_irradiance": np.full((2, 2), -1.0)},
    ],
)
def test_invalid_transfer_inputs(change):
    with pytest.raises(ValueError):
        surface_radiance_tangent(*queries(), **(transfer_inputs() | change))
    if "reflectance_tangent" not in change:
        with pytest.raises(ValueError):
            surface_radiance(*queries(), **radiance_inputs(transfer_inputs() | change))


def test_unsupported_visibility_queries_and_column_callable_axes():
    xy, view = queries()
    with pytest.raises(ValueError, match="queries,2"):
        surface_radiance_tangent(xy[None], view[None], **transfer_inputs())
    with pytest.raises(ValueError):
        surface_radiance_tangent(xy + 1e9, view, **transfer_inputs())
    with pytest.raises(ValueError, match="visibility"):
        surface_radiance_tangent(xy, view, **(transfer_inputs() | {"slopes": np.array([2.0, 0.0])}))
    with pytest.raises(ValueError, match="segment and species"):
        surface_radiance_tangent(
            *queries(),
            **(
                transfer_inputs()
                | {
                    "columns": lambda a, b, layers: (
                        np.zeros((len(a), 1)),
                        np.zeros((len(a), 1, 3)),
                    )
                }
            ),
        )


def test_direct_unsupported_visibility_queries_and_column_callable_axes():
    xy, view = queries()
    options = radiance_inputs(transfer_inputs())
    with pytest.raises(ValueError, match="queries,2"):
        surface_radiance(xy[None], view[None], **options)
    with pytest.raises(ValueError):
        surface_radiance(xy + 1e9, view, **options)
    with pytest.raises(ValueError, match="visibility"):
        surface_radiance(xy, view, **(options | {"slopes": np.array([2.0, 0.0])}))
    with pytest.raises(ValueError, match="segment and species"):
        surface_radiance(
            xy, view, **(options | {"columns": lambda a, b, layers: np.zeros((len(a), 1))})
        )


def curved_inputs() -> dict[str, Any]:
    options = transfer_inputs()
    options.pop("height_m")
    options.pop("slopes")
    return options | dict(
        height=lambda xy: 250 + 0.002 * xy[:, 0] ** 2 + 0.001 * xy[:, 1] ** 2,
        gradient=lambda xy: xy * [0.004, 0.002],
        visibility="assume_clear",
    )


def test_curved_terrain_local_incidence_direct_tangent_and_finite_difference():
    xy, view = queries()
    options = curved_inputs()
    value, tangent = surface_radiance_tangent(xy, view, **options)
    assert_array_equal(surface_radiance(xy, view, **radiance_inputs(options)), value)
    density = np.array([[1e20, 2e19], [3e20, 5e19]])
    options["columns"] = lambda a, b, layers: (
        density[layers] * np.linalg.norm(b - a, axis=1)[:, None]
    )
    heights = options["height"](xy)
    thickness = np.column_stack((400 - heights, np.full(2, 600.0)))
    vertical = np.einsum("ql,lg,lgs->qs", thickness, density, options["cross_sections_m2"])
    normals = np.column_stack((-options["gradient"](xy), np.ones(2)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    incidence = normals @ options["solar_direction"]
    expected = (
        options["reflectance"]
        * options["beam_normal_irradiance"]
        * incidence[:, None]
        / np.pi
        * np.exp(-vertical * (1 / 0.8 + 1 / view[:, 2, None]))
    )
    physical = options.copy()
    physical.pop("reflectance_tangent")
    actual = surface_radiance(xy, view, **physical)
    assert_allclose(actual, expected, rtol=3e-15)
    assert incidence[0] != incidence[1]
    # Gas and reflectance derivatives share fixed curved geometry.
    options = curved_inputs()
    field = field_inputs()
    for k in range(3):
        step = 1e-3
        values = []
        for sign in (1, -1):
            modified = field | {
                "scales": field["scales"] + sign * step * field["scale_tangent"][:, k]
            }
            current = options | {
                "columns": partial(puff_molecular_columns, **modified),
                "reflectance": options["reflectance"]
                + sign * step * options["reflectance_tangent"][..., k],
            }
            values.append(surface_radiance(xy, view, **radiance_inputs(current)))
        assert_allclose((values[0] - values[1]) / (2 * step), tangent[..., k], rtol=2e-8)


def test_callable_affine_scene_agreement_and_empty_queries():
    xy, view = queries()
    options = transfer_inputs() | {"slopes": np.array([0.07, -0.03])}
    expected, expected_tangent = surface_radiance_tangent(xy, view, **options)
    height, slopes = options.pop("height_m"), options.pop("slopes")
    options.update(
        height=lambda xy: height + xy @ slopes,
        gradient=lambda xy: np.broadcast_to(slopes, xy.shape),
        visibility="assume_clear",
    )
    value, tangent = surface_radiance_tangent(xy, view, **options)
    assert_allclose(value, expected, rtol=3e-15)
    assert_allclose(tangent, expected_tangent, rtol=3e-15)
    options.update(
        reflectance=np.empty((0, 2)),
        reflectance_tangent=np.empty((0, 2, 3)),
        beam_normal_irradiance=np.empty((0, 2)),
    )
    value, tangent = surface_radiance_tangent(np.empty((0, 2)), np.empty((0, 3)), **options)
    assert value.shape == (0, 2) and tangent.shape == (0, 2, 3)


@pytest.mark.parametrize("tangent", [False, True])
@pytest.mark.parametrize("blocked", ["solar", "viewing", None])
def test_full_ray_visibility_policy_rejects_occlusion_before_columns(tangent, blocked):
    options = curved_inputs()
    xy, view = queries()
    sampled = []

    def visibility(points, directions):
        assert_allclose(points[:, :2], xy)
        assert_allclose(points[:, 2], options["height"](xy))
        assert_allclose(np.linalg.norm(directions, axis=1), 1)
        solar = np.array_equal(directions, np.broadcast_to(options["solar_direction"], view.shape))
        sampled.append("solar" if solar else "viewing")
        return np.full(len(points), sampled[-1] != blocked, dtype=bool)

    options["visibility"] = visibility
    if blocked is not None:

        def unavailable(a, b, layers):
            raise AssertionError("unsupported views must fail before atmospheric evaluation")

        options["columns"] = unavailable
    call = surface_radiance_tangent if tangent else surface_radiance
    args = options if tangent else radiance_inputs(options)
    if blocked is not None:
        with pytest.raises(ValueError, match="visibility"):
            call(xy, view, **args)
    else:
        call(xy, view, **args)
    assert sampled == (["solar"] if blocked == "solar" else ["solar", "viewing"])


@pytest.mark.parametrize(
    "change",
    [
        dict(visibility=None),
        dict(visibility="local_only"),
        dict(visibility=lambda p, d: np.ones(len(p))),
        dict(visibility=lambda p, d: np.ones((len(p), 1), dtype=bool)),
        dict(gradient=None),
        dict(height=None),
        dict(height_m=250.0),
        dict(slopes=np.zeros(2)),
        dict(mode="independent_column"),
        dict(gradient=lambda xy: np.broadcast_to([2.0, 0.0], xy.shape)),
        dict(gradient=lambda xy: np.broadcast_to([0.0, 2.0], xy.shape)),
        dict(height=lambda xy: np.full(len(xy), 1100.0)),
    ],
)
@pytest.mark.parametrize("tangent", [False, True])
def test_callable_terrain_invalid_inputs(change, tangent):
    options = curved_inputs() | change
    with pytest.raises(ValueError):
        if tangent:
            surface_radiance_tangent(*queries(), **options)
        else:
            surface_radiance(*queries(), **radiance_inputs(options))


def test_missing_affine_parameters_and_callback_errors():
    options = transfer_inputs()
    options.pop("slopes")
    with pytest.raises(ValueError, match="supply affine"):
        surface_radiance_tangent(*queries(), **options)
    options = curved_inputs()

    def unavailable(points, directions):
        raise RuntimeError("visibility unavailable")

    options["visibility"] = unavailable
    with pytest.raises(RuntimeError, match="visibility unavailable"):
        surface_radiance_tangent(*queries(), **options)


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
@pytest.mark.parametrize("cutoff", [None, 3.0])
def test_molecular_adapter_retains_reflection_and_compact_support(boundary, cutoff):
    state = GaussianPuffs(
        0.0, np.zeros(1), np.ones(1), np.array([[0.0, 0.0, 0.0]]), np.array([np.eye(3)])
    )
    plume = GaussianPuffField(state, boundary, kernel_cutoff_radius=cutoff)
    starts = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    ends = starts + [0.0, 0.0, 10.0]
    options = dict(
        height_edges_m=np.array([0.0, 10.0]),
        background_columns_molecules_m2=np.zeros((1, 1)),
        plumes=(plume,),
        species_indices=np.zeros(1, dtype=np.int64),
        molar_masses_kg_mol=np.array([0.016]),
        scales=np.array([2.0]),
    )
    value, tangent = puff_molecular_columns(
        starts, ends, np.zeros(2, dtype=np.int64), **options, scale_tangent=np.array([[-1.0]])
    )
    expected = (
        puff_path_columns(state, starts, ends, boundary=boundary, kernel_cutoff_radius=cutoff)
        * Avogadro
        / 0.016
    )
    assert_allclose(value[:, 0], expected * 2, rtol=2e-15)
    assert_allclose(tangent[:, 0, 0], -expected, rtol=2e-15)
    if cutoff is not None:
        assert value[-1, 0] == 0
