/* Fixed-cross-section absorption and directional derivatives. Inputs remain
 * exported and read-only during detached serial computation. Outputs are owned.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "absorption_kernel.h"

namespace py = pybind11;
#include "../transfer/native/tangent_bindings.h"

static void validate_absorption_buffer(const py::buffer_info &buffer, int rank) {
    if (buffer.itemsize != sizeof(double) ||
            (buffer.format != "d" && buffer.format != "=d" && buffer.format != "@d"))
        throw py::value_error("absorption inputs must be native float64");
    if (buffer.ndim != rank)
        throw py::value_error("absorption input rank mismatch");
}

static void validate_absorption_support(
        const py::buffer_info &sigma, const py::buffer_info &columns) {
    const auto layers = sigma.shape[0], points = sigma.shape[1];
    if (layers < 1 || points < 1 || columns.shape[0] != layers)
        throw py::value_error("absorption requires matching nonempty layer and point support");
    if (layers > PY_SSIZE_T_MAX / sizeof(double) / points)
        throw std::overflow_error("absorption output size overflow");
}

static py::array_t<double> cross_section_optical_depth(
        py::buffer cross_sections, py::buffer columns) {
    const auto sigma_buffer = cross_sections.request(), column_buffer = columns.request();
    validate_absorption_buffer(sigma_buffer, 2);
    validate_absorption_buffer(column_buffer, 1);
    validate_absorption_support(sigma_buffer, column_buffer);
    const auto layers = sigma_buffer.shape[0], points = sigma_buffer.shape[1];
    py::array_t<double> depth({layers, points});
    auto *out = depth.mutable_data();
    const auto sigma = tangent_input(cross_sections), n = tangent_input(columns);
    {
        py::gil_scoped_release release;
        absorption_kernel(sigma.data(), n.data(), layers, points, out);
    }
    return depth;
}

static py::tuple cross_section_optical_depth_tangent(
        py::buffer cross_sections, py::buffer columns, py::buffer column_tangent) {
    std::array<py::buffer_info, 3> b = {
        cross_sections.request(), columns.request(), column_tangent.request()};
    for (int i = 0; i < 3; ++i)
        validate_absorption_buffer(b[i], i == 1 ? 1 : 2);
    validate_absorption_support(b[0], b[1]);
    const auto layers = b[0].shape[0], points = b[0].shape[1], k = b[2].shape[1];
    if (b[2].shape[0] != layers)
        throw py::value_error("absorption requires matching nonempty layer and point support");
    if (k && layers * points > PY_SSIZE_T_MAX / sizeof(double) / k)
        throw std::overflow_error("absorption tangent output size overflow");
    py::array_t<double> depth({layers, points}), tangent({layers, points, k});
    auto *v = depth.mutable_data(), *dv = tangent.mutable_data();
    const auto sigma = tangent_input(cross_sections);
    const auto n = tangent_input(columns);
    const auto dn = tangent_input(column_tangent);
    {
        py::gil_scoped_release release;
        absorption_tangent_kernel(sigma.data(), n.data(), dn.data(), layers, points, k, v, dv);
    }
    return py::make_tuple(depth, tangent);
}


PYBIND11_MODULE(absorption_native, module, py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    module.def("cross_section_optical_depth", &cross_section_optical_depth,
               "Compute owned float64 optical depth without directional storage.");
    module.def("cross_section_optical_depth_tangent",
               &cross_section_optical_depth_tangent,
               "Compute owned float64 absorption and last-axis directional tangents.");
}
