/* Full-spectrum scalar SOS with call-local geometry and scratch.
 * Buffer exports live through detached computation; spectral points run serially.
 */
#include <cstring>
#include <deque>
#include <stdexcept>

static double strided_value(const py::buffer_info &b, py::ssize_t row,
                            py::ssize_t column = 0) {
    if (!b.ptr) return 0.0;
    const auto *p = static_cast<const char *>(b.ptr) + row * b.strides[0];
    if (b.ndim == 2) p += column * b.strides[1];
    double value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

static py::ssize_t checked_product(py::ssize_t a, py::ssize_t b) {
    if (a < 0 || b < 0 || (b && a > PY_SSIZE_T_MAX / b / sizeof(double)))
        throw std::overflow_error("scattering workspace size overflow");
    return a * b;
}

static double transport_phase(double cosine, double g, int moments, bool rayleigh,
                              const double *coefficients = nullptr) {
    const double x = std::clamp(cosine, -1.0, 1.0);
    if (rayleigh) return 0.75 * (1.0 + x * x);
    if (moments == 0 || g <= 0.0)
        return (1.0 - g * g) / std::pow(1.0 + g * g - 2.0 * g * x, 1.5);
    // Clenshaw's Legendre recurrence, matching the retained NumPy reference.
    const double f = coefficients ? 0.0 : std::pow(g, moments);
    auto coefficient = [&](int k) {
        return coefficients ? coefficients[k] :
            (2.0 * k + 1.0) * (std::pow(g, k) - f) / (1.0 - f);
    };
    double c0 = coefficient(moments - 2), c1 = coefficient(moments - 1);
    int nd = moments;
    for (int i = 3; i <= moments; ++i) {
        const double old = c0;
        --nd;
        c0 = coefficient(moments - i) - c1 * (nd - 1.0) / nd;
        c1 = old + c1 * x * (2.0 * nd - 1.0) / nd;
    }
    return c0 + c1 * x;
}

static void dense_redistribute(Dgemm gemm, const double *field, const double *phase,
                               double *source, int cells, int directions) {
    int angular = directions - 1;
    if (gemm) {
        char n = 'N';
        double one = 1.0, zero = 0.0;
        gemm(&n, &n, &directions, &cells, &angular, &one, const_cast<double *>(phase),
             &directions, const_cast<double *>(field), &directions, &zero, source, &directions);
    } else {
        for (int c = 0; c < cells; ++c)
            for (int d = 0; d < directions; ++d) {
                double sum = 0.0;
                for (int j = 0; j < angular; ++j)
                    sum += field[std::ptrdiff_t{c} * directions + j] * phase[std::ptrdiff_t{j} * directions + d];
                source[std::ptrdiff_t{c} * directions + d] = sum;
            }
    }
}

static py::array_t<double> spectral_orders(
        py::tuple optical, py::object irradiance_object, py::object reflectance_object,
        py::object directions_object, py::object weights_object,
        py::ssize_t layers, py::ssize_t subdivisions, py::ssize_t max_order,
        int moments, double zenith, double view_zenith, double azimuth, Dgemm gemm) {
    if (layers < 1 || subdivisions < 1 || max_order < 0 ||
        max_order == PY_SSIZE_T_MAX || moments < 0 || moments == 1)
        throw py::value_error("invalid spectral orders dimensions or moments");
    const std::array<py::handle, 4> objects = {irradiance_object, reflectance_object, directions_object, weights_object};
    std::array<py::buffer_info, 4> b;
    for (int i = 0; i < 4; ++i) {
        b[i] = py::reinterpret_borrow<py::buffer>(objects[i]).request();
        if (b[i].itemsize != sizeof(double) ||
            (b[i].format != "d" && b[i].format != "=d" && b[i].format != "@d"))
            throw py::value_error("spectral orders buffers must be native doubles");
        if (b[i].ndim != (i == 2 ? 2 : 1))
            throw py::value_error("spectral orders buffer rank mismatch");
    }
    const auto points = b[0].shape[0];
    if (b[1].shape[0] != points || b[2].shape[1] != 3 ||
        b[3].shape[0] != b[2].shape[0])
        throw py::value_error("spectral orders angular or spectral grid mismatch");
    if (!std::isfinite(zenith) || zenith < 0 || zenith >= 90 ||
        !std::isfinite(view_zenith) || view_zenith < 0 || view_zenith >= 90 ||
        !std::isfinite(azimuth))
        throw py::value_error("spectral orders geometry outside supported hemisphere");
    const auto directions = b[2].shape[0] + 1;
    const auto angular = directions - 1;
    const auto cells = max_order && angular ? checked_product(layers, subdivisions) : 0;
    const auto count = checked_product(cells, directions);
    checked_product(max_order + 1, points);
    checked_product(angular, directions);
    checked_product(directions, 7);
    if (cells > std::numeric_limits<int>::max() || directions > std::numeric_limits<int>::max())
        throw std::overflow_error("scattering dimensions exceed native BLAS range");
    double mu0, muv;
    std::vector<double> irradiance(points), reflectance(points), vectors, mu, weights,
                        beam_cosines, incoming, outgoing, rayleigh_beam, rayleigh_matrix;
    std::vector<std::ptrdiff_t> previous, down;
    {
        py::gil_scoped_release release;
        constexpr double pi = 3.14159265358979323846;
        // Match math.radians: round pi/180 once, then multiply each angle.
        // Reassociating as angle*pi/180 changes near-grazing cosines materially.
        constexpr double radians_per_degree = pi / 180.0;
        const double zenith_radians = zenith * radians_per_degree;
        const double view_radians = view_zenith * radians_per_degree;
        const double azimuth_radians = azimuth * radians_per_degree;
        mu0 = std::cos(zenith_radians);
        muv = std::cos(view_radians);
        for (py::ssize_t i = 0; i < points; ++i) {
            irradiance[i] = strided_value(b[0], i);
            reflectance[i] = strided_value(b[1], i);
            if (!std::isfinite(irradiance[i]) || irradiance[i] < 0)
                throw std::invalid_argument("radiance values must be finite and nonnegative (source irradiance)");
            if (!std::isfinite(reflectance[i]) || reflectance[i] < 0 || reflectance[i] > 1)
                throw std::invalid_argument("surface reflectance must be finite and in [0, 1]");
        }
        if (cells) {
            vectors.resize(directions * 3); mu.resize(directions); weights.resize(angular);
            beam_cosines.resize(directions); incoming.resize(angular * 7); outgoing.resize(7 * directions);
            rayleigh_beam.resize(directions);
            previous.resize(directions);
            const double vx = std::sin(view_radians) * std::cos(azimuth_radians);
            const double vy = std::sin(view_radians) * std::sin(azimuth_radians);
            for (py::ssize_t d = 0; d < directions; ++d) {
                double x = d < angular ? strided_value(b[2], d, 0) : vx;
                double y = d < angular ? strided_value(b[2], d, 1) : vy;
                double z = d < angular ? strided_value(b[2], d, 2) : muv;
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z == 0 || std::abs(z) > 1)
                    throw std::invalid_argument("scattering directions must be finite with nonzero cosines");
                vectors[d * 3] = x; vectors[d * 3 + 1] = y; vectors[d * 3 + 2] = z;
                mu[d] = z;
                beam_cosines[d] = x * std::sin(zenith_radians) - z * mu0;
                rayleigh_beam[d] = transport_phase(beam_cosines[d], 0, 0, true);
                double basis[] = {1.0, x*x, y*y, z*z, x*y, x*z, y*z};
                if (d < angular) {
                    weights[d] = strided_value(b[3], d);
                    if (!std::isfinite(weights[d]) || weights[d] < 0)
                        throw std::invalid_argument("scattering quadrature weights must be finite and nonnegative");
                    for (int k = 0; k < 7; ++k) incoming[d * 7 + k] = basis[k] * weights[d];
                    if (z < 0) down.push_back(d);
                }
                for (int k = 0; k < 7; ++k) outgoing[k * directions + d] = basis[k] * (k < 4 ? 0.75 : 1.5);
                previous[d] = d;
                if (directions <= 1024)
                    for (py::ssize_t j = 0; j < d; ++j)
                        if (mu[j] == z) { previous[d] = j; break; }
            }
            if (layers <= 2) {
                rayleigh_matrix.resize(angular * directions);
                for (py::ssize_t j = 0; j < angular; ++j)
                    for (py::ssize_t d = 0; d < directions; ++d) {
                        double cosine = 0.0;
                        for (int k = 0; k < 3; ++k) cosine += vectors[j * 3 + k] * vectors[d * 3 + k];
                        rayleigh_matrix[j * directions + d] = transport_phase(cosine, 0, 0, true) * weights[j];
                    }
            }
        }
    }
    // Release fixed-support exports before acquiring current optical arrays.
    for (auto &buffer : b) buffer = py::buffer_info();
    if (optical.size() != 5) throw py::value_error("spectral orders requires five optical roles");
    std::array<py::buffer_info, 5> optical_buffers;
    auto &ob = optical_buffers;
    for (int i = 0; i < 5; ++i) {
        if (optical[i].is_none()) continue;
        ob[i] = py::reinterpret_borrow<py::buffer>(optical[i]).request();
        if (ob[i].itemsize != sizeof(double) ||
            (ob[i].format != "d" && ob[i].format != "=d" && ob[i].format != "@d"))
            throw py::value_error("spectral orders buffers must be native doubles");
        if (ob[i].ndim != 2 || ob[i].shape[0] != layers || ob[i].shape[1] != points)
            throw py::value_error("spectral orders optical grid mismatch");
    }
    if (bool(ob[2].ptr) != bool(ob[3].ptr) || bool(ob[2].ptr) != bool(ob[4].ptr))
        throw py::value_error("spectral orders requires all aerosol arrays together");
    py::array_t<double> values({max_order + 1, points});
    auto *output = values.mutable_data();
    {
        py::gil_scoped_release release;
        constexpr double pi = 3.14159265358979323846;
        const auto &b = optical_buffers;
        std::fill(output, output + (max_order + 1) * points, 0.0);
        std::vector<double> ray(cells), particle(layers), g(layers), depth(cells),
                            t(count), ht(count), w(count), hw(count), sw(count), shw(count),
                            zero(count), bc(count), be(directions), field(count), source(count),
                            beam(count), beam_centres(max_order > 1 ? count : 0),
                            beam_exits(directions), exits(directions), temp(count),
                            temp_exits(directions), flux(down.size()), m(checked_product(cells, 7)),
                            oversized, coefficients(cells ? moments : 0);
        // Reuse phases across this call's layers, orders and spectral points.
        std::deque<std::pair<double, std::vector<double>>> cache;
        auto prepare_phase = [&](double parameter) {
            if (moments && parameter > 0 && parameter <= 0.8) {
                const double f = std::pow(parameter, moments);
                for (int k = 0; k < moments; ++k)
                    coefficients[k] = (2.0 * k + 1.0) * (std::pow(parameter, k) - f) / (1.0 - f);
            }
        };
        constexpr std::size_t budget = 16 * 1024 * 1024;
        auto phase_operator = [&](double parameter) -> const double * {
            if (parameter == 2.0) return rayleigh_matrix.data();
            for (auto &entry : cache) if (entry.first == parameter) return entry.second.data();
            const auto n = angular * directions;
            const auto bytes = static_cast<std::size_t>(n) * sizeof(double);
            while (!cache.empty() && (cache.size() + 1) * bytes > budget) cache.pop_front();
            std::vector<double> *matrix;
            if (bytes <= budget) {
                cache.emplace_back(parameter, std::vector<double>(n));
                matrix = &cache.back().second;
            } else {
                oversized.resize(n); matrix = &oversized;
            }
            prepare_phase(parameter);
            for (py::ssize_t j = 0; j < angular; ++j)
                for (py::ssize_t d = 0; d < directions; ++d) {
                    double cosine = 0.0;
                    for (int k = 0; k < 3; ++k) cosine += vectors[j * 3 + k] * vectors[d * 3 + k];
                    (*matrix)[j * directions + d] = transport_phase(cosine, parameter, moments, false, coefficients.data()) * weights[j];
                }
            return matrix->data();
        };
        for (py::ssize_t point = 0; point < points; ++point) {
            double total = 0.0, physical_total = 0.0;
            bool active = false;
            for (py::ssize_t layer = 0; layer < layers; ++layer) {
                double ext = 0.0;
                for (int role = 0; role < 3; ++role) {
                    const double value = strided_value(b[role], layer, point);
                    if (!std::isfinite(value)) throw std::invalid_argument("summed optical depth overflow");
                    if (value < 0)
                        throw std::invalid_argument("optical depths must be finite and nonnegative");
                    ext += value;
                }
                physical_total += ext;
                double r = strided_value(b[1], layer, point);
                double a = strided_value(b[3], layer, point);
                if (!std::isfinite(a) || a < 0 || a > strided_value(b[2], layer, point))
                    throw std::invalid_argument("aerosol scattering depth must be within extinction");
                g[layer] = strided_value(b[4], layer, point);
                if (!std::isfinite(g[layer]) || std::abs(g[layer]) > 0.8)
                    throw std::invalid_argument("aerosol HG asymmetry must be finite and in [-0.8, 0.8]");
                const double fraction = moments && g[layer] > 0 ? std::pow(g[layer], moments) : 0.0;
                active = active || r != 0 || a != 0;
                ext -= fraction * a;
                total += ext;
                particle[layer] = a / subdivisions * (1.0 - fraction);
                for (py::ssize_t sub = 0; cells && sub < subdivisions; ++sub) {
                    const auto c = layer * subdivisions + sub;
                    depth[c] = ext / subdivisions; ray[c] = r / subdivisions;
                }
            }
            if (!std::isfinite(physical_total)) throw std::invalid_argument("summed optical depth overflow");
            const double amplitude = irradiance[point] * reflectance[point] * mu0 / pi;
            const double direct = amplitude * std::exp(-total * (1.0 / mu0 + 1.0 / muv));
            if (!std::isfinite(direct) || direct < 0) throw std::invalid_argument("radiance values must be finite and nonnegative");
            output[point] = direct;
            if (!max_order || !active) continue;
            if (!cells) throw std::invalid_argument("scattering orders requires angular directions");
            prepare_kernel(depth.data(), mu.data(), t.data(), ht.data(), w.data(), hw.data(), sw.data(), shw.data(), previous.data(), true, mu0, cells, directions);
            apply_kernel(mu.data(), zero.data(), t.data(), w.data(), ht.data(), hw.data(), bc.data(), be.data(), 1.0, cells, directions);
            bool reuse = true;
            for (py::ssize_t d = 0; d < directions; ++d)
                if (mu[d] > 0 && be[d] < std::numeric_limits<double>::min()) reuse = false;
            const double boundary0 = amplitude * std::exp(-rayleigh_flux_sum(depth.data(), depth.size()) / mu0);
            if (reuse)
                for (py::ssize_t i = 0; i < count; ++i) field[i] = bc[i] * boundary0;
            else apply_kernel(mu.data(), zero.data(), t.data(), w.data(), ht.data(), hw.data(), field.data(), exits.data(), boundary0, cells, directions);
            for (py::ssize_t layer = 0; layer < layers; ++layer) {
                const double r = irradiance[point] / (4.0 * pi) * strided_value(b[1], layer, point) / subdivisions;
                const double a = irradiance[point] / (4.0 * pi) * strided_value(b[3], layer, point) *
                    (1.0 - (moments && g[layer] > 0 ? std::pow(g[layer], moments) : 0.0)) / subdivisions;
                prepare_phase(g[layer]);
                for (py::ssize_t d = 0; d < directions; ++d) {
                    const double value = r * rayleigh_beam[d] +
                        (a == 0 ? 0.0 : a * transport_phase(beam_cosines[d], g[layer], moments, false, coefficients.data()));
                    for (py::ssize_t sub = 0; sub < subdivisions; ++sub) beam[(layer * subdivisions + sub) * directions + d] = value;
                }
            }
            apply_kernel(mu.data(), beam.data(), t.data(), sw.data(), ht.data(), shw.data(), max_order > 1 ? beam_centres.data() : nullptr, beam_exits.data(), 0.0, cells, directions);
            for (py::ssize_t order = 1; order <= max_order; ++order) {
                std::fill(source.begin(), source.end(), 0.0);
                if (b[1].ptr && layers > 2) {
                    if (gemm) rayleigh_blas_redistribute(gemm, field.data(), incoming.data(), outgoing.data(), ray.data(), m.data(), source.data(), static_cast<int>(cells), static_cast<int>(directions));
                    else rayleigh_redistribute(field.data(), incoming.data(), outgoing.data(), ray.data(), source.data(), cells, directions);
                }
                for (int role = 0; role < 2; ++role) {
                    if (role == 0 && (!b[1].ptr || layers > 2)) continue;
                    if (role == 1 && !b[3].ptr) continue;
                    for (py::ssize_t layer = 0; layer < layers; ++layer) {
                        const double scale = role == 0 ? ray[layer * subdivisions] : particle[layer];
                        if (scale == 0.0) continue;
                        const auto offset = layer * subdivisions * directions;
                        const double *phase = phase_operator(role == 0 ? 2.0 : g[layer]);
                        dense_redistribute(gemm, field.data() + offset, phase, temp.data(), static_cast<int>(subdivisions), static_cast<int>(directions));
                        for (py::ssize_t i = 0; i < subdivisions * directions; ++i) source[offset + i] += temp[i] * scale;
                    }
                }
                const bool retain = order < max_order;
                apply_kernel(mu.data(), source.data(), t.data(), w.data(), ht.data(), hw.data(), retain ? field.data() : nullptr, exits.data(), 0.0, cells, directions);
                if (order == 1) {
                    for (py::ssize_t d = 0; d < directions; ++d) exits[d] += beam_exits[d];
                    if (retain) for (py::ssize_t i = 0; i < count; ++i) field[i] += beam_centres[i];
                }
                for (std::size_t j = 0; j < down.size(); ++j) flux[j] = weights[down[j]] * -mu[down[j]] * exits[down[j]];
                const double boundary = 4.0 * reflectance[point] * rayleigh_flux_sum(flux.data(), flux.size());
                if (boundary != 0.0) {
                    if (reuse) {
                        for (py::ssize_t d = 0; d < directions; ++d) exits[d] += be[d] * boundary;
                        if (retain) for (py::ssize_t i = 0; i < count; ++i) field[i] += bc[i] * boundary;
                    } else {
                        apply_kernel(mu.data(), zero.data(), t.data(), w.data(), ht.data(), hw.data(), retain ? temp.data() : nullptr, temp_exits.data(), boundary, cells, directions);
                        for (py::ssize_t d = 0; d < directions; ++d) exits[d] += temp_exits[d];
                        if (retain) for (py::ssize_t i = 0; i < count; ++i) field[i] += temp[i];
                    }
                }
                output[order * points + point] = exits.back();
            }
        }
        // Preserve the public error priority across the complete spectral field.
        for (py::ssize_t i = 0; i < (max_order + 1) * points; ++i)
            if (output[i] < 0) throw std::invalid_argument("Delta-M produced negative order radiance; refine moments/angular sampling or disable Delta-M (no intensity correction is implemented)");
        for (py::ssize_t i = 0; i < (max_order + 1) * points; ++i)
            if (!std::isfinite(output[i])) throw std::invalid_argument("scattering order radiance must be finite");
    }
    return values;
}

static py::cpp_function make_spectral_orders(py::object owner, Dgemm gemm) {
    return py::cpp_function([owner = std::move(owner), gemm](
            py::tuple optical, py::object irradiance, py::object reflectance,
            py::object directions, py::object weights, py::ssize_t layers,
            py::ssize_t subdivisions, py::ssize_t max_order, int moments,
            double zenith, double view_zenith, double azimuth) {
        return spectral_orders(optical, irradiance, reflectance, directions, weights,
            layers, subdivisions, max_order, moments, zenith, view_zenith, azimuth, gemm);
    });
}
