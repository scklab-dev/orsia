"""Direct NumPy domain checks at scalar, array-like and callback boundaries."""

import numpy as np
import pytest
from orsia_instruments import (
    DetectorElectronics,
    DetectorMaps,
    DetectorOperatingPoints,
    ReadSchedule,
    cavity_transmission,
    gaussian_filter,
    image_charge,
    image_readout,
    longpass_filter,
    lorentzian_filter,
    lvf_centre,
    notch_filter,
    pixel_quantum_efficiency,
    rectangular_filter,
    regular_pixel_bounds,
    super_gaussian_filter,
    two_beam_ports,
)
from orsia_instruments.optics.pixels import modulated_pixel_response


def test_pixel_qe_constructs_float_array_in_read_order():
    reads = ReadSchedule(
        ("a", "b", "c"),
        ("q", "p", "q"),
        ("s", "s", "s"),
        ("+", "+", "+"),
        np.array([[0.0, 1.0]] * 3),
    )
    values = pixel_quantum_efficiency(reads, {"p": 0, "q": 1})
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [[1], [0], [1]])
    with pytest.raises(ValueError, match="pixel QE must be finite"):
        pixel_quantum_efficiency(reads, {"p": np.nan, "q": 1})


def test_cavity_converts_array_like_inputs_without_mutation():
    wavelengths = [1, 2]
    spacing = np.array([1, 2], dtype=np.float32)
    spacing.flags.writeable = False
    transmission = cavity_transmission(
        wavelengths,
        spacing,
        refractive_index=[1, 1],
        internal_angle_rad=[0, 0],
        reflectivity=[0, 0],
        peak=[0.5, 1],
        order_transmission=[1, 0.25],
    )
    assert transmission.dtype == np.float64
    np.testing.assert_array_equal(transmission, [0.5, 0.25])
    np.testing.assert_array_equal(spacing, [1, 2])
    assert spacing.dtype == np.float32 and not spacing.flags.writeable
    assert wavelengths == [1, 2]


def test_scalar_lvf_centre_returns_zero_dimensional_array():
    centre = lvf_centre(0, reference_m=1.0, gradient=0)
    assert isinstance(centre, np.ndarray)
    assert centre.shape == () and centre.dtype == np.float64
    assert centre.item() == 1.0


@pytest.mark.parametrize(
    "make_maps",
    [
        lambda value: DetectorMaps(quantum_efficiency=value),
        lambda value: DetectorMaps(dark_electrons_per_s=value),
        lambda value: DetectorMaps(read_noise_electrons=value),
        lambda value: DetectorMaps(full_well_electrons=value),
        lambda value: DetectorMaps(electrons_per_dn=value),
        lambda value: DetectorMaps(bias_dn=value),
        lambda value: DetectorMaps(nonlinearity=value),
    ],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_detector_maps_reject_nonfinite_converted_values(make_maps, bad):
    with pytest.raises(ValueError, match="finite"):
        make_maps([[bad]])


@pytest.mark.parametrize(
    "field",
    ["dark_electrons_per_s", "read_noise_electrons", "full_well_electrons", "electrons_per_dn"],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_scalar_electronics_keep_finiteness_checks(field, bad):
    with pytest.raises(ValueError, match="finite"):
        DetectorElectronics(**{field: bad})


@pytest.mark.parametrize(
    "field", ["row_noise_electrons", "column_noise_electrons", "exposure_noise_electrons"]
)
@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
def test_common_noise_is_validated_before_readout_iteration(field, bad):
    charge = image_charge({"zero": np.zeros((1, 1, 1))}, exposure_bounds_s=np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="common noise must be finite"):
        image_readout(charge, **{field: bad})


@pytest.mark.parametrize(
    "law,parameters,field",
    [
        (gaussian_filter, dict(wavelength_m=1, centre_m=1, width_m=1), "width_m"),
        (lorentzian_filter, dict(wavelength_m=1, centre_m=1, fwhm_m=1), "fwhm_m"),
        (super_gaussian_filter, dict(wavelength_m=1, centre_m=1, fwhm_m=1), "fwhm_m"),
        (rectangular_filter, dict(wavelength_m=1, lower_m=1, upper_m=2), "lower_m"),
        (longpass_filter, dict(wavelength_m=1, cutoff_m=1, transition_m=1), "transition_m"),
        (cavity_transmission, dict(wavelength_m=1, spacing_m=1, order_transmission=1), "spacing_m"),
        (two_beam_ports, dict(wavelength_m=1, opd_m=0), "wavelength_m"),
    ],
)
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("array_input", [False, True])
def test_optical_laws_require_finite_strictly_positive_inputs(
    law, parameters, field, bad, array_input
):
    values = [1.0, bad] if array_input else bad
    with pytest.raises(ValueError, match="finite and positive"):
        law(**(parameters | {field: values}))


@pytest.mark.parametrize(
    "law,parameters,field",
    [
        (gaussian_filter, dict(wavelength_m=1, centre_m=1, width_m=1), "peak"),
        (lorentzian_filter, dict(wavelength_m=1, centre_m=1, fwhm_m=1), "peak"),
        (super_gaussian_filter, dict(wavelength_m=1, centre_m=1, fwhm_m=1), "peak"),
        (rectangular_filter, dict(wavelength_m=1, lower_m=1, upper_m=2), "peak"),
        (longpass_filter, dict(wavelength_m=1, cutoff_m=1, transition_m=1), "pass_transmission"),
        (notch_filter, dict(wavelength_m=1, centre_m=1, fwhm_m=1), "depth"),
        (cavity_transmission, dict(wavelength_m=1, spacing_m=1, order_transmission=1), "peak"),
        (cavity_transmission, dict(wavelength_m=1, spacing_m=1), "order_transmission"),
        (two_beam_ports, dict(wavelength_m=1, opd_m=0), "visibility"),
        (two_beam_ports, dict(wavelength_m=1, opd_m=0), "transmission"),
    ],
)
@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_optical_fractions_reject_out_of_range_and_nonfinite_values(law, parameters, field, bad):
    with pytest.raises(ValueError, match="finite and in"):
        law(**(parameters | {field: [0.5, bad]}))


def test_optical_fraction_endpoints_remain_inclusive_without_mutation():
    source = np.array([0, 99, 1, 99], dtype=np.float32)
    fractions = source[::2]
    fractions.flags.writeable = False
    np.testing.assert_array_equal(gaussian_filter(1, 1, 1, peak=fractions), [0, 1])
    plus, minus = two_beam_ports(1, 0, visibility=fractions, transmission=fractions)
    np.testing.assert_array_equal(plus, [0, 1])
    np.testing.assert_array_equal(minus, [0, 0])
    np.testing.assert_array_equal(source, [0, 99, 1, 99])
    assert fractions.dtype == np.float32 and not fractions.flags.writeable


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_local_transmission_callback_is_checked_once_before_integration(bad):
    values = np.array([0.5, bad])
    before = values.copy()
    values.flags.writeable = False
    calls = 0

    def transmission(x, y, wave):
        nonlocal calls
        calls += 1
        return values

    with pytest.raises(ValueError, match="local response must be finite and in"):
        modulated_pixel_response(
            np.array([1.0, 2.0]),
            [-1, 1, -1, 1],
            np.zeros((1, 2)),
            sigma_m=1,
            transmission=transmission,
            pixel_order=2,
        )
    assert calls == 1
    np.testing.assert_array_equal(values, before)
    assert not values.flags.writeable


@pytest.mark.parametrize(
    "field",
    [
        "dark_electrons_per_s",
        "read_noise_electrons",
        "full_well_electrons",
        "electrons_per_dn",
        "nonlinearity",
    ],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_per_read_detector_checks_nonfinite_values(field, bad):
    parameters = dict(
        dark_electrons_per_s=np.zeros(2),
        read_noise_electrons=np.zeros(2),
        full_well_electrons=np.ones(2),
        electrons_per_dn=np.ones(2),
        bias_dn=np.zeros(2),
        nonlinearity=np.array([-0.5, 0.5]),
    )
    DetectorOperatingPoints(**parameters)
    parameters[field] = np.array([0.5, bad])
    with pytest.raises(ValueError, match="finite"):
        DetectorOperatingPoints(**parameters)


@pytest.mark.parametrize("field", ["pitch_m", "active_size_m"])
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_pixel_dimensions_keep_strict_positivity_checks(field, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        regular_pixel_bounds((1, 1), **(dict(pitch_m=1, active_size_m=0.5) | {field: [1, bad]}))
