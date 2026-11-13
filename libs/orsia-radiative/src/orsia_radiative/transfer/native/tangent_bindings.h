/* Python storage adaptation for native directional kernels. */
#ifndef ORSIA_RADIATIVE_TANGENT_BINDINGS_H
#define ORSIA_RADIATIVE_TANGENT_BINDINGS_H
#include <cstdint>

static py::array_t<double, py::array::c_style> tangent_input(py::handle input) {
    auto array = py::array_t<double, py::array::c_style>::ensure(input);
    if (!array) throw py::value_error("tangent buffer conversion failed");
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double))
        array = array.attr("copy")().cast<py::array_t<double, py::array::c_style>>();
    return array;
}
#endif
