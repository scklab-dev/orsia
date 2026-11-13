/* Thermal Python bindings; numerical kernels have no Python dependencies. */
#include "thermal_kernel.h"

static py::buffer_info thermal_buffer(py::handle object) {
    auto buffer = request_buffer(object, false);
    int precision = 0;
    same_precision(buffer, precision);
    if (reinterpret_cast<std::uintptr_t>(buffer.ptr) % alignof(double))
        throw py::value_error("thermal buffers must be aligned");
    return buffer;
}

static py::tuple thermal_terms(
        py::object grid, py::object temperature, py::object emissivity,
        py::object absorption, py::object aerosol, py::object mu, py::object weights,
        double surface_temperature, double view_mu, bool analytic, ThermalExpn expn,
        double second_radiation, double rayleigh_jeans, py::tuple tangents = py::tuple()) {
    const std::array<py::handle, 7> objects = {grid, temperature, emissivity, absorption, aerosol, mu, weights};
    std::array<py::buffer_info, 7> buffers;
    for (int i = 0; i < 7; ++i) {
        if ((i == 3 || i == 4) && objects[i].is_none()) continue;
        buffers[i] = thermal_buffer(objects[i]);
        if (buffers[i].ndim != ((i == 3 || i == 4) ? 2 : 1))
            throw py::value_error("thermal buffer rank mismatch");
    }
    const auto points = buffers[0].shape[0], layers = buffers[1].shape[0], angles = buffers[5].shape[0];
    if (points < 1 || layers < 1 || buffers[2].shape[0] != points || buffers[6].shape[0] != angles ||
        (!analytic && angles == 1))
        throw py::value_error("thermal vector support mismatch");
    for (int i : {3, 4})
        if (!objects[i].is_none() && (buffers[i].shape[0] != layers || buffers[i].shape[1] != points))
            throw py::value_error("thermal optical grid mismatch");
    if (!std::isfinite(view_mu) || view_mu <= 0.0 || view_mu > 1.0 ||
        !std::isfinite(surface_temperature) || surface_temperature <= 0.0)
        throw py::value_error("thermal temperature and view cosine require physical support");
    auto input = [&](int i) { return static_cast<const double *>(buffers[i].ptr); };
    for (py::ssize_t i = 0; i < angles; ++i)
        if (!std::isfinite(input(5)[i]) || input(5)[i] <= 0.0 || input(5)[i] > 1.0 ||
            !std::isfinite(input(6)[i]) || input(6)[i] < 0.0)
            throw py::value_error("thermal angular grid requires positive cosines and finite weights");
    py::array_t<double> surface(points), atmospheric(points), reflected(points);
    auto *s = surface.mutable_data(), *a = atmospheric.mutable_data(), *r = reflected.mutable_data();
    const bool directional = tangents.size() != 0;
    std::array<py::buffer_info, 4> db;
    py::ssize_t k = 0;
    if (directional) {
        if (tangents.size() != 4) throw py::value_error("thermal requires four tangent arrays");
        for (int i = 0; i < 4; ++i) db[i] = thermal_buffer(tangents[i]);
        if (db[0].ndim != 3 || db[1].ndim != 2 || db[2].ndim != 1 || db[3].ndim != 2)
            throw py::value_error("thermal tangent rank mismatch");
        k = db[0].shape[2];
        if (db[0].shape[0] != layers || db[0].shape[1] != points ||
            db[1].shape[0] != layers || db[1].shape[1] != k || db[2].shape[0] != k ||
            db[3].shape[0] != points || db[3].shape[1] != k)
            throw py::value_error("thermal tangent layer, spectral or direction support mismatch");
        if (k && !analytic && angles < 2)
            throw py::value_error("thermal tangents require angular nodes even at zero reflection");
        checked_product(checked_product(layers, points), k);
    }
    py::array_t<double> ds({points, k}), da({points, k}), dr({points, k});
    auto *dsp = ds.mutable_data(), *dap = da.mutable_data(), *drp = dr.mutable_data();
    auto derivative = [&](int i) { return static_cast<const double *>(db[i].ptr); };
    {
        py::gil_scoped_release release;
        if (directional)
            for (const auto &b : db)
                for (py::ssize_t i = 0; i < b.size; ++i)
                    if (!std::isfinite(static_cast<const double *>(b.ptr)[i]))
                        throw std::invalid_argument("thermal tangents must be finite");
        thermal_kernel(input(0), input(1), input(2), input(3), input(4), input(5), input(6),
                       layers, points, angles, surface_temperature, view_mu, analytic, expn,
                       second_radiation, rayleigh_jeans, s, a, r, k,
                       derivative(0), derivative(1), derivative(2), derivative(3), dsp, dap, drp);
    }
    if (directional) return py::make_tuple(surface, atmospheric, reflected, ds, da, dr);
    return py::make_tuple(surface, atmospheric, reflected);
}

static py::array_t<double> thermal_intervals(py::object below, py::object depth, ThermalExpn expn) {
    auto a = thermal_buffer(below), d = thermal_buffer(depth);
    if (a.shape != d.shape) throw py::value_error("thermal interval support mismatch");
    py::array_t<double> result(d.shape);
    auto *output = result.mutable_data();
    const auto *ap = static_cast<const double *>(a.ptr), *dp = static_cast<const double *>(d.ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < d.size; ++i)
            output[i] = thermal_e3_interval(expn, ap[i], dp[i]);
    }
    return result;
}

static ThermalExpn thermal_expn_pointer(py::object owner, double second_radiation, double rayleigh_jeans) {
    if (!std::isfinite(second_radiation) || second_radiation <= 0.0 ||
        !std::isfinite(rayleigh_jeans) || rayleigh_jeans <= 0.0)
        throw py::value_error("Planck coefficients must be positive finite scalars");
    // Match the exported fused double ABI explicitly, including Cython's dispatch argument.
    // Each callable retains its interpreter's provider; every evaluation owns its scratch.
    py::capsule capsule = owner.attr("__pyx_capi__")["__pyx_fuse_0expn"];
    void *pointer = PyCapsule_GetPointer(capsule.ptr(), "double (double, double, int __pyx_skip_dispatch)");
    if (!pointer) throw py::error_already_set();
    return reinterpret_cast<ThermalExpn>(pointer);
}

static py::tuple bind_thermal_expn(py::object owner, double second_radiation, double rayleigh_jeans) {
    auto expn = thermal_expn_pointer(owner, second_radiation, rayleigh_jeans);
    auto terms = py::cpp_function([owner, expn, second_radiation, rayleigh_jeans](
            py::object grid, py::object temperature, py::object emissivity,
            py::object absorption, py::object aerosol, py::object mu, py::object weights,
            double surface_temperature, double view_mu, bool analytic) {
        return thermal_terms(grid, temperature, emissivity, absorption, aerosol, mu, weights,
                             surface_temperature, view_mu, analytic, expn,
                             second_radiation, rayleigh_jeans);
    });
    auto intervals = py::cpp_function([owner, expn](py::object below, py::object depth) {
        return thermal_intervals(below, depth, expn);
    });
    return py::make_tuple(terms, intervals);
}

static py::cpp_function bind_thermal_tangents(py::object owner, double second_radiation, double rayleigh_jeans) {
    auto expn = thermal_expn_pointer(owner, second_radiation, rayleigh_jeans);
    return py::cpp_function([owner, expn, second_radiation, rayleigh_jeans](
            py::object grid, py::object temperature, py::object emissivity,
            py::object absorption, py::object aerosol, py::object mu, py::object weights,
            double surface_temperature, double view_mu, bool analytic, py::tuple tangents) {
        return thermal_terms(grid, temperature, emissivity, absorption, aerosol, mu, weights,
                             surface_temperature, view_mu, analytic, expn,
                             second_radiation, rayleigh_jeans, tangents);
    });
}
