/* Serial IEEE float/double preparation and recurrence. No fast math, FMA, retained
 * pointers or writable global state. Preparation owns a bounded cosine lookup;
 * recurrence allocates nothing. Buffers stay exported during the call.
 * Callers must not mutate input arrays concurrently; outputs are call-owned.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// The supported SciPy LP64 Cython BLAS supplies this native Fortran ABI.
using Dgemm = void (*)(char *, char *, int *, int *, int *, double *, double *,
                      int *, double *, int *, double *, double *, int *);

/* Instantiate identical operation order for float and double. Typed constants
 * and standard math overloads avoid double promotion in float kernels;
 * build flags prohibit contraction/FMA. */
template <typename Real>
static Real phi(Real x) {
    return x < std::numeric_limits<Real>::epsilon() ? Real{1} : -std::expm1(-x) / x;
}

template <typename Real>
static Real integral(Real a, Real b) {
    return std::exp(std::fmax(a, b)) * phi(std::fabs(b - a));
}

template <typename Real>
static void prepare_kernel(const Real *depth, const Real *mu,
                           Real *transmission, Real *half_transmission,
                           Real *weight, Real *half_weight,
                           Real *solar_weight, Real *solar_half_weight,
                           const std::ptrdiff_t *previous, bool solar, Real solar_mu,
                           std::ptrdiff_t cells, std::ptrdiff_t directions) {
    Real above = Real{0};
    for (std::ptrdiff_t step = 0; step < cells; ++step) {
        std::ptrdiff_t c = cells - 1 - step;
        Real top = solar ? -above / solar_mu : Real{0};
        Real bottom = solar ? -(above + depth[c]) / solar_mu : Real{0};
        Real midpoint = solar ? -(above + depth[c] / Real{2}) / solar_mu : Real{0};
        for (std::ptrdiff_t d = 0; d < directions; ++d) {
            std::ptrdiff_t i = c * directions + d;
            if (previous && previous[d] != d) {
                std::ptrdiff_t j = c * directions + previous[d];
                transmission[i] = transmission[j];
                half_transmission[i] = half_transmission[j];
                weight[i] = weight[j];
                half_weight[i] = half_weight[j];
                if (solar) {
                    solar_weight[i] = solar_weight[j];
                    solar_half_weight[i] = solar_half_weight[j];
                }
                continue;
            }
            Real direction = std::fabs(mu[d]);
            Real attenuation = depth[c] / direction;
            transmission[i] = std::exp(-attenuation);
            half_transmission[i] = std::exp(-attenuation / Real{2});
            weight[i] = phi(attenuation) / direction;
            half_weight[i] = phi(attenuation / Real{2}) / (Real{2} * direction);
            if (solar) {
                Real entry = mu[d] > 0 ? bottom : top;
                Real end = mu[d] > 0 ? top : bottom;
                solar_weight[i] = integral(entry - attenuation, end) / direction;
                solar_half_weight[i] = integral(entry - attenuation / Real{2}, midpoint) /
                                       (Real{2} * direction);
            }
        }
        if (solar) above += depth[c];
    }
}

template <typename Real>
static void original_apply_kernel(const Real *mu, const Real *source,
                         const Real *transmission, const Real *weight,
                         const Real *half_transmission, const Real *half_weight,
                         Real *field, Real *exits, Real bottom,
                         std::ptrdiff_t cells, std::ptrdiff_t directions) {
    for (std::ptrdiff_t d = 0; d < directions; ++d)
        exits[d] = mu[d] > 0 ? bottom : Real{0};
    for (int upward = 0; upward < 2; ++upward) {
        for (std::ptrdiff_t step = 0; step < cells; ++step) {
            std::ptrdiff_t c = upward ? step : cells - 1 - step;
            std::ptrdiff_t offset = c * directions;
            for (std::ptrdiff_t d = 0; d < directions; ++d) {
                if (upward ? mu[d] <= 0 : mu[d] >= 0) continue;
                std::ptrdiff_t i = offset + d;
                Real incoming = exits[d];
                if (field)
                    field[i] = incoming * half_transmission[i] + source[i] * half_weight[i];
                exits[d] = incoming * transmission[i] + source[i] * weight[i];
            }
        }
    }
}

/* The two writable regions are disjoint from every input and each other.
 * Read-only inputs may overlap. Keep the original per-ray expression and cell
 * order while sweeping contiguous signed direction ranges at each cell. */
#if defined(_MSC_VER)
#define ORSIA_RESTRICT __restrict
#else
#define ORSIA_RESTRICT __restrict__
#endif
template <typename Real, bool Centres>
static void range_apply_kernel(const Real *source, const Real *transmission,
                               const Real *weight, const Real *half_transmission,
                               const Real *half_weight, Real *ORSIA_RESTRICT field,
                               Real *ORSIA_RESTRICT exits, Real bottom,
                               std::ptrdiff_t cells, std::ptrdiff_t directions,
                               std::ptrdiff_t nb, std::ptrdiff_t ne,
                               std::ptrdiff_t pb, std::ptrdiff_t pe) {
    for (int upward = 0; upward < 2; ++upward) {
        const auto begin = upward ? pb : nb;
        const auto end = upward ? pe : ne;
        for (auto d = begin; d < end; ++d) exits[d] = upward ? bottom : Real{0};
        for (std::ptrdiff_t step = 0; step < cells; ++step) {
            const auto c = upward ? step : cells - 1 - step;
            const auto offset = c * directions;
            for (auto d = begin; d < end; ++d) {
                const auto i = offset + d;
                Real incoming = exits[d];
                if constexpr (Centres)
                    field[i] = incoming * half_transmission[i] + source[i] * half_weight[i];
                exits[d] = incoming * transmission[i] + source[i] * weight[i];
            }
        }
    }
}
#undef ORSIA_RESTRICT

static bool overlaps(const void *a, std::size_t an, const void *b, std::size_t bn) {
    if (!a || !b || !an || !bn) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

template <typename Real>
static void apply_kernel(const Real *mu, const Real *source,
                         const Real *transmission, const Real *weight,
                         const Real *half_transmission, const Real *half_weight,
                         Real *field, Real *exits, Real bottom,
                         std::ptrdiff_t cells, std::ptrdiff_t directions) {
    std::ptrdiff_t nb = directions, ne = 0, pb = directions, pe = 0, nc = 0, pc = 0;
    bool valid = true;
    for (std::ptrdiff_t d = 0; d < directions; ++d) {
        if (mu[d] < 0) { nb = std::min(nb, d); ne = d + 1; ++nc; }
        else if (mu[d] > 0) { pb = std::min(pb, d); pe = d + 1; ++pc; }
        else valid = false;
    }
    if (!nc) nb = ne = 0;
    if (!pc) pb = pe = 0;
    valid = valid && ne - nb == nc && pe - pb == pc;
    // Buffer acquisition already established contiguous, representable sizes.
    const auto matrix_bytes = static_cast<std::size_t>(cells) * directions * sizeof(Real);
    const auto vector_bytes = static_cast<std::size_t>(directions) * sizeof(Real);
    valid = valid && !overlaps(field, matrix_bytes, exits, vector_bytes);
    const Real *inputs[] = {mu, source, transmission, weight,
                           field ? half_transmission : nullptr, field ? half_weight : nullptr};
    for (int i = 0; i < 6; ++i) {
        const auto bytes = i == 0 ? vector_bytes : matrix_bytes;
        valid = valid && !overlaps(field, matrix_bytes, inputs[i], bytes)
                      && !overlaps(exits, vector_bytes, inputs[i], bytes);
    }
    if (!valid) {
        original_apply_kernel(mu, source, transmission, weight, half_transmission,
                              half_weight, field, exits, bottom, cells, directions);
    } else if (field) {
        range_apply_kernel<Real, true>(source, transmission, weight, half_transmission,
            half_weight, field, exits, bottom, cells, directions, nb, ne, pb, pe);
    } else {
        range_apply_kernel<Real, false>(source, transmission, weight, half_transmission,
            half_weight, field, exits, bottom, cells, directions, nb, ne, pb, pe);
    }
}

/* buffer::request() requests strides, not C contiguity. Keep the original
 * export flags (and exporter-specific errors) while using buffer_info for RAII.
 * No NumPy dependency, forcecast, copies, or retained pointers at this boundary. */
static py::buffer_info request_buffer(py::handle object, bool writable) {
    auto view = std::make_unique<Py_buffer>();
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object.ptr(), view.get(), flags) < 0)
        throw py::error_already_set();
    try {
        py::buffer_info info(view.get());
        view.release(); // buffer_info now owns the export and its descriptor.
        return info;
    } catch (...) {
        PyBuffer_Release(view.get());
        throw;
    }
}

static void same_precision(const py::buffer_info &buffer, int &single) {
    bool is_float = buffer.itemsize == sizeof(float) && buffer.format == "f";
    bool is_double = buffer.itemsize == sizeof(double) && buffer.format == "d";
    if ((!is_float && !is_double) || (single >= 0 && single != is_float))
        throw py::value_error("propagation buffers must be matching native floats or doubles");
    single = is_float;
}

static double real_scalar(py::handle object) {
    // Match PyArg_ParseTuple's "d": unlike the default pybind11 double caster,
    // preserve overflow and exceptions raised by user __float__/__index__ methods.
    double value = PyFloat_AsDouble(object.ptr());
    if (value == -1.0 && PyErr_Occurred()) throw py::error_already_set();
    return value;
}

template <typename Real>
static void prepare_buffers(std::array<py::buffer_info, 8> &buffers,
                            bool solar, double solar_mu) {
    const auto cells = buffers[0].shape[0];
    const auto directions = buffers[1].shape[0];
    const auto *mu = static_cast<const Real *>(buffers[1].ptr);
    /* Exact repeated signed cosines occur across azimuths. Bound lookup work;
     * very wide arbitrary grids use the ordinary fused scalar loop instead. */
    std::vector<std::ptrdiff_t> previous;
    if (directions && directions <= 1024) {
        previous.resize(directions);
        for (std::ptrdiff_t d = 0; d < directions; ++d) {
            previous[d] = d;
            for (std::ptrdiff_t j = 0; j < d; ++j)
                if (mu[j] == mu[d]) { previous[d] = j; break; }
        }
    }
    // Exports outlive this guard; releases and Python refcounts run attached.
    py::gil_scoped_release release;
    prepare_kernel(static_cast<const Real *>(buffers[0].ptr), mu,
                   static_cast<Real *>(buffers[2].ptr), static_cast<Real *>(buffers[3].ptr),
                   static_cast<Real *>(buffers[4].ptr), static_cast<Real *>(buffers[5].ptr),
                   static_cast<Real *>(buffers[6].ptr), static_cast<Real *>(buffers[7].ptr),
                   previous.empty() ? nullptr : previous.data(), solar,
                   static_cast<Real>(solar_mu), cells, directions);
}

static void prepare(py::object depth, py::object mu, py::object solar_cosine,
                    py::object transmission, py::object half_transmission,
                    py::object weight, py::object half_weight,
                    py::object solar_weight, py::object solar_half_weight) {
    double solar_mu = real_scalar(solar_cosine);
    bool solar = !solar_weight.is_none();
    if (solar != !solar_half_weight.is_none() ||
        (solar && !(solar_mu > 0.0 && solar_mu <= 1.0)))
        throw py::value_error("invalid solar propagation buffers or cosine");
    const std::array<py::handle, 8> objects = {depth, mu, transmission, half_transmission,
                                             weight, half_weight, solar_weight, solar_half_weight};
    std::array<py::buffer_info, 8> buffers;
    int single = -1;
    for (int i = 0; i < 8; ++i) {
        if (!solar && i >= 6) continue;
        buffers[i] = request_buffer(objects[i], i >= 2);
        same_precision(buffers[i], single);
    }
    if (solar && single && static_cast<float>(solar_mu) == 0.0f)
        throw py::value_error("solar cosine underflows float precision");
    if (buffers[0].ndim != 1 || buffers[1].ndim != 1)
        throw py::value_error("preparation requires vector depth and directions");
    for (int i = 2; i < 8; ++i) {
        if (!solar && i >= 6) continue;
        if (buffers[i].ndim != 2 || buffers[i].shape[0] != buffers[0].shape[0] ||
            buffers[i].shape[1] != buffers[1].shape[0])
            throw py::value_error("preparation matrix grid mismatch");
    }
    if (single) prepare_buffers<float>(buffers, solar, solar_mu);
    else prepare_buffers<double>(buffers, solar, solar_mu);
}

template <typename Real>
static void apply_buffers(std::array<py::buffer_info, 8> &buffers, double bottom) {
    py::gil_scoped_release release;
    apply_kernel(static_cast<const Real *>(buffers[0].ptr),
                 static_cast<const Real *>(buffers[1].ptr),
                 static_cast<const Real *>(buffers[2].ptr),
                 static_cast<const Real *>(buffers[3].ptr),
                 static_cast<const Real *>(buffers[4].ptr),
                 static_cast<const Real *>(buffers[5].ptr),
                 static_cast<Real *>(buffers[6].ptr), static_cast<Real *>(buffers[7].ptr),
                 static_cast<Real>(bottom), buffers[1].shape[0], buffers[0].shape[0]);
}

static void apply(py::object mu, py::object source, py::object transmission,
                  py::object weight, py::object half_transmission, py::object half_weight,
                  py::object bottom_radiance, py::object field, py::object exits) {
    double bottom = real_scalar(bottom_radiance);
    bool centres = !field.is_none();
    const std::array<py::handle, 8> objects = {mu, source, transmission, weight,
                                             half_transmission, half_weight, field, exits};
    std::array<py::buffer_info, 8> buffers;
    int single = -1;
    for (int i = 0; i < 8; ++i) {
        if (!centres && (i == 4 || i == 5 || i == 6)) continue;
        buffers[i] = request_buffer(objects[i], i >= 6);
        same_precision(buffers[i], single);
    }
    if (buffers[0].ndim != 1 || buffers[1].ndim != 2)
        throw py::value_error("propagation requires vector directions and matrix source");
    const auto cells = buffers[1].shape[0];
    const auto directions = buffers[0].shape[0];
    if (buffers[1].shape[1] != directions || buffers[7].ndim != 1 ||
        buffers[7].shape[0] != directions)
        throw py::value_error("propagation direction grid mismatch");
    for (int i = 2; i <= 6; ++i) {
        if (!centres && i >= 4) continue;
        if (buffers[i].ndim != 2 || buffers[i].shape[0] != cells ||
            buffers[i].shape[1] != directions)
            throw py::value_error("propagation matrix grid mismatch");
    }
    if (single) apply_buffers<float>(buffers, bottom);
    else apply_buffers<double>(buffers, bottom);
}

/* Fixed seven-factor products. Each cell's moments stay on the stack; independent
 * output directions vectorize under strict floating-point flags. Incoming sums
 * traverse directions in storage order. This replaces BLAS arithmetic
 * with separate double multiply/add and therefore permits roundoff differences.
 * Input and output storage are disjoint, and inputs may share read-only storage. */
static void rayleigh_redistribute(const double *field, const double *incoming,
                                 const double *outgoing, const double *depth,
                                 double *source, std::ptrdiff_t cells,
                                 std::ptrdiff_t directions) {
    for (std::ptrdiff_t c = 0; c < cells; ++c) {
        double m[7] = {};
        for (std::ptrdiff_t d = 0; d < directions - 1; ++d)
            for (int k = 0; k < 7; ++k)
                m[k] += field[c * directions + d] * incoming[d * 7 + k];
        for (std::ptrdiff_t d = 0; d < directions; ++d) {
            double value = m[0] * outgoing[d];
            for (int k = 1; k < 7; ++k)
                value += m[k] * outgoing[k * directions + d];
            source[c * directions + d] = value * depth[c];
        }
    }
}

static void rayleigh_blas_redistribute(
        Dgemm gemm, const double *field, const double *incoming, const double *outgoing,
        const double *depth, double *moments, double *source, int cells, int directions) {
    // Row-major C = A B becomes column-major C^T = B^T A^T, including the
    // observer-column gap in field. The two products keep separate rounding.
    char n = 'N';
    int seven = 7, angular = directions - 1;
    double one = 1.0, zero = 0.0;
    gemm(&n, &n, &seven, &cells, &angular, &one, const_cast<double *>(incoming),
         &seven, const_cast<double *>(field), &directions, &zero, moments, &seven);
    gemm(&n, &n, &directions, &cells, &seven, &one, const_cast<double *>(outgoing),
         &directions, moments, &seven, &zero, source, &directions);
    for (int c = 0; c < cells; ++c)
        for (int d = 0; d < directions; ++d)
            source[std::ptrdiff_t{c} * directions + d] *= depth[c];
}

/* Balanced contiguous reduction with small sequential leaves bounds summation
 * depth for large angular grids. Its tree is explicit and backend-independent. */
static double rayleigh_flux_sum(const double *flux, std::size_t size) {
    if (size <= 8) {
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) sum += flux[i];
        return sum;
    }
    const auto half = size / 2;
    return rayleigh_flux_sum(flux, half) + rayleigh_flux_sum(flux + half, size - half);
}

/* One point's recurrence uses only pointers, sizes and call-owned scratch.
 * The binding owns buffer validation, allocation and thread-state transitions. */
static void rayleigh_order_kernel(
        const double *depth, const double *incoming, const double *outgoing,
        const double *beam, const double *mu, const double *transmission,
        const double *half_transmission, const double *weight, const double *half_weight,
        const double *solar_weight, const double *solar_half_weight,
        const double *boundary_centres, const double *boundary_exits,
        const std::ptrdiff_t *down, const double *flux_weight, std::size_t down_count,
        double direct_boundary, double reflectance, std::ptrdiff_t max_order,
        std::ptrdiff_t cells, std::ptrdiff_t directions, double *field,
        double *source, double *exits, double *beam_centres, double *beam_exits,
        double *flux, double *values, double *moments, Dgemm gemm) {
    const auto count = cells * directions;
    apply_kernel(mu, beam, transmission, solar_weight, half_transmission,
                 solar_half_weight, beam_centres, beam_exits, 0.0, cells, directions);
    for (std::ptrdiff_t i = 0; i < count; ++i)
        field[i] = boundary_centres[i] * direct_boundary;
    for (std::ptrdiff_t order = 0; order < max_order; ++order) {
        if (gemm && cells && directions > 1)
            rayleigh_blas_redistribute(gemm, field, incoming, outgoing, depth,
                                      moments, source, static_cast<int>(cells),
                                      static_cast<int>(directions));
        else
            rayleigh_redistribute(field, incoming, outgoing, depth, source, cells, directions);
        const bool centres = order + 1 < max_order;
        apply_kernel(mu, source, transmission, weight, half_transmission, half_weight,
                     centres ? field : nullptr, exits, 0.0, cells, directions);
        if (order == 0) {
            for (std::ptrdiff_t d = 0; d < directions; ++d) exits[d] += beam_exits[d];
            if (centres)
                for (std::ptrdiff_t i = 0; i < count; ++i) field[i] += beam_centres[i];
        }
        for (std::size_t j = 0; j < down_count; ++j)
            flux[j] = flux_weight[j] * exits[down[j]];
        const double boundary = (4.0 * reflectance) * rayleigh_flux_sum(flux, down_count);
        if (boundary != 0.0) {
            for (std::ptrdiff_t d = 0; d < directions; ++d)
                exits[d] += boundary_exits[d] * boundary;
            if (centres)
                for (std::ptrdiff_t i = 0; i < count; ++i)
                    field[i] += boundary_centres[i] * boundary;
        }
        values[order] = exits[directions - 1];
    }
}

static py::array_t<double> rayleigh_orders(
        py::object depth, py::tuple factors, py::object beam_source,
        py::tuple coefficients, py::object boundary_centres, py::object boundary_exits,
        py::object weights, double direct_boundary, double reflectance,
        py::ssize_t max_order, Dgemm gemm) {
    if (max_order < 0 || factors.size() != 2 || coefficients.size() != 7)
        throw py::value_error("invalid Rayleigh order count or coefficient tuple");
    // depth, incoming, outgoing, beam, mu, transmission, half transmission,
    // diffuse weight, diffuse half weight, solar weight, solar half weight,
    // unit boundary centres, unit boundary exits, quadrature weights.
    const std::array<py::handle, 14> objects = {
        depth, factors[0], factors[1], beam_source, coefficients[0], coefficients[1],
        coefficients[2], coefficients[3], coefficients[4], coefficients[5],
        coefficients[6], boundary_centres, boundary_exits, weights};
    std::array<py::buffer_info, 14> buffers;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        buffers[i] = request_buffer(objects[i], false);
        int precision = 0; // Only native double is supported by this private ABI.
        same_precision(buffers[i], precision);
        if (reinterpret_cast<std::uintptr_t>(buffers[i].ptr) % alignof(double))
            throw py::value_error("Rayleigh buffers must be aligned");
    }
    if (buffers[0].ndim != 1 || buffers[4].ndim != 1)
        throw py::value_error("Rayleigh depth and directions must be vectors");
    const auto cells = buffers[0].shape[0], directions = buffers[4].shape[0];
    if (directions < 1)
        throw py::value_error("Rayleigh requires an observer direction");
    for (int i : {3, 5, 6, 7, 8, 9, 10, 11})
        if (buffers[i].ndim != 2 || buffers[i].shape[0] != cells ||
            buffers[i].shape[1] != directions)
            throw py::value_error("Rayleigh matrix grid mismatch");
    if (buffers[1].ndim != 2 || buffers[1].shape[0] != directions - 1 ||
        buffers[1].shape[1] != 7 || buffers[2].ndim != 2 ||
        buffers[2].shape[0] != 7 || buffers[2].shape[1] != directions ||
        buffers[12].ndim != 1 || buffers[12].shape[0] != directions ||
        buffers[13].ndim != 1 || buffers[13].shape[0] != directions - 1)
        throw py::value_error("Rayleigh factor or boundary grid mismatch");
    auto input = [&](int i) { return static_cast<const double *>(buffers[i].ptr); };
    const auto *mu = input(4);
    std::vector<std::ptrdiff_t> down;
    std::vector<double> flux_weight;
    for (py::ssize_t d = 0; d < directions; ++d) {
        if (!std::isfinite(mu[d]) || mu[d] == 0.0)
            throw py::value_error("Rayleigh directions must be finite and nonzero");
        if (d < directions - 1 && mu[d] < 0.0) {
            down.push_back(d);
            flux_weight.push_back(input(13)[d] * -mu[d]);
        }
    }
    py::array_t<double> values(max_order);
    if (max_order == 0) return values;
    const auto count = cells * directions;
    // Every scratch element is written before reading; avoid zero-initializing
    // matrix-sized storage that the first products/sweeps immediately replace.
    std::unique_ptr<double[]> field(new double[count]), redistributed(new double[count]);
    std::unique_ptr<double[]> exits(new double[directions]), beam_exits(new double[directions]);
    std::unique_ptr<double[]> beam_centres(max_order > 1 ? new double[count] : nullptr);
    std::unique_ptr<double[]> flux(new double[down.size()]);
    if (cells > std::numeric_limits<int>::max() || directions > std::numeric_limits<int>::max())
        gemm = nullptr;
    std::unique_ptr<double[]> moments(gemm ? new double[cells * 7] : nullptr);
    auto *v = values.mutable_data();
    {
        // Exports and return storage outlive this single detached region.
        py::gil_scoped_release release;
        rayleigh_order_kernel(input(0), input(1), input(2), input(3), mu,
            input(5), input(6), input(7), input(8), input(9), input(10), input(11), input(12),
            down.data(), flux_weight.data(), down.size(), direct_boundary, reflectance,
            max_order, cells, directions, field.get(), redistributed.get(), exits.get(),
            beam_centres.get(), beam_exits.get(), flux.get(), v, moments.get(), gemm);
    }
    return values;
}

static py::cpp_function make_rayleigh_orders(py::object owner, Dgemm gemm) {
    // The callable retains the exporting module for the function pointer's life.
    // Its immutable binding belongs to this interpreter; scratch belongs to calls.
    return py::cpp_function(
        [owner = std::move(owner), gemm](
                py::object depth, py::tuple factors, py::object beam_source,
                py::tuple coefficients, py::object boundary_centres, py::object boundary_exits,
                py::object weights, double direct_boundary, double reflectance,
                py::ssize_t max_order) {
            return rayleigh_orders(depth, factors, beam_source, coefficients,
                boundary_centres, boundary_exits, weights, direct_boundary, reflectance,
                max_order, gemm);
        });
}

static Dgemm dgemm_pointer(py::handle owner) {
    py::capsule capsule = owner.attr("__pyx_capi__")["dgemm"];
    return reinterpret_cast<Dgemm>(capsule.get_pointer());
}

#include "orders_native.h"
#include "thermal_native.h"

static py::cpp_function bind_rayleigh_blas(py::object owner) {
    auto gemm = dgemm_pointer(owner);
    return make_rayleigh_orders(std::move(owner), gemm);
}

static py::cpp_function bind_spectral_blas(py::object owner) {
    auto gemm = dgemm_pointer(owner);
    return make_spectral_orders(std::move(owner), gemm);
}

PYBIND11_MODULE(propagation_native, module, py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    // Unnamed arguments retain the original positional-only native API.
    module.def("prepare", &prepare,
               "Prepare transport in fresh, disjoint contiguous float/double buffers.");
    module.def("apply", &apply,
               "Apply transport to validated, contiguous float/double buffers.");
    module.attr("rayleigh_orders") = make_rayleigh_orders(py::none(), nullptr);
    module.attr("spectral_orders") = make_spectral_orders(py::none(), nullptr);
    module.def("bind_spectral_blas", &bind_spectral_blas,
               "Bind SciPy BLAS to the full-spectrum detached SOS calculation.");
    module.def("bind_rayleigh_blas", &bind_rayleigh_blas,
               "Bind SciPy BLAS to a detached native Rayleigh callable.");
    module.def("bind_thermal_expn", &bind_thermal_expn,
               "Bind SciPy exponential integrals to serial native LTE thermal transport.");
}
