/* Absorption-only direct reflected solar value and directional tangents.
 * The binding exports input buffers and owns output storage. Serial computation
 * uses call-local scratch while detached; inputs must stay stable throughout.
 */
#include "tangent_bindings.h"
#include "solar_tangent_kernel.h"

static py::tuple reflected_radiance_tangent(
        py::buffer depth, py::buffer depth_tangent, py::buffer irradiance,
        py::buffer reflectance, py::buffer reflectance_tangent,
        double zenith, double view_zenith) {
    std::array<py::buffer_info, 5> b = {depth.request(), depth_tangent.request(),
        irradiance.request(), reflectance.request(), reflectance_tangent.request()};
    const int ranks[] = {2, 3, 1, 1, 2};
    for (int i = 0; i < 5; ++i) {
        if (b[i].itemsize != sizeof(double) ||
            (b[i].format != "d" && b[i].format != "=d" && b[i].format != "@d"))
            throw py::value_error("solar tangent inputs must be native float64");
        if (b[i].ndim != ranks[i])
            throw py::value_error("solar tangent input rank mismatch");
    }
    const auto layers = b[0].shape[0], points = b[0].shape[1], k = b[1].shape[2];
    if (layers < 1 || points < 1 || b[1].shape[0] != layers ||
        b[1].shape[1] != points || b[2].shape[0] != points ||
        b[3].shape[0] != points || b[4].shape[0] != points || b[4].shape[1] != k)
        throw py::value_error("solar tangent layer, spectral or direction support mismatch");
    if (!std::isfinite(zenith) || zenith < 0 || zenith >= 90 ||
        !std::isfinite(view_zenith) || view_zenith < 0 || view_zenith >= 90)
        throw py::value_error("solar tangent geometry outside supported hemisphere");
    checked_product(points, k);
    py::array_t<double> values(points), tangent({points, k});
    auto *v = values.mutable_data(), *dv = tangent.mutable_data();
    const auto tau = tangent_input(depth), dtau = tangent_input(depth_tangent);
    const auto sun = tangent_input(irradiance), rho = tangent_input(reflectance);
    const auto drho = tangent_input(reflectance_tangent);
    {
        py::gil_scoped_release release;
        solar_tangent_kernel(tau.data(), dtau.data(), sun.data(), rho.data(), drho.data(),
                             layers, points, k, zenith, view_zenith, v, dv);
    }
    return py::make_tuple(values, tangent);
}
