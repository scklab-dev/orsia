/* Absorption-only LTE thermal transport, included by propagation_native.cpp.
 * Bottom-to-top layers; cm-1 coordinates; K temperatures; radiance per cm-1.
 * Serial pointer kernels use call-local scratch and borrowed stable inputs.
 */
using ThermalExpn = double (*)(double, double, int);

// Eight asymptotic terms retain E2/E3's representable opaque tail; the next
// relative term is below 6e-19 at x >= 700, beneath subnormal rounding.
static double thermal_expn(ThermalExpn expn, int order, double x) {
    if (x >= 745.0) return 0.0;
    if (x >= 700.0) {
        double term = 1.0, total = 1.0;
        for (int degree = 1; degree <= 8; ++degree) {
            term *= -(order + degree - 1) / x;
            total += term;
        }
        return std::exp(-x) * (total / x);
    }
    return expn(order, x, 1);
}

// Keep d independent of a+d, including intervals smaller than one ulp of a.
// Thin intervals integrate the midpoint E2 expansion through fourth order;
// its next term is d**7 E_-4(midpoint)/322560. The origin uses expm1 and
// the remaining near-origin intervals use the integrated series through
// degree 16 (next term < 2e-21 relative to d for a+d <= .5).
static double thermal_e3_interval(ThermalExpn expn, double a, double d) {
    if (d == 0.0 || a >= 745.0) return 0.0;
    const double b = a + d;
    if (a == 0.0 && d <= 1.0)
        return (-std::expm1(-d) + d * (std::exp(-d) - d * expn(1, d, 1))) / 2.0;
    if (a > 0.0 && d <= 0.02 * std::min(a, 1.0)) {
        const double midpoint = a + d / 2.0, ratio = d / midpoint;
        const double correction = std::exp(-midpoint) * (d * ratio) *
            (1.0 / 24.0 + (d * d + 2.0 * d * ratio + 2.0 * ratio * ratio) / 1920.0);
        return d * (thermal_expn(expn, 2, midpoint) + correction);
    }
    if (b <= 0.5) {
        constexpr double euler_gamma = 0.57721566490153286061;
        double average = 1.0 + (a + b) * (0.5 * std::log(b) + euler_gamma / 2.0 - 0.75)
            + 0.5 * a * (a / d) * (std::log(b) - std::log(a));
        double divided_power = a + b, a_power = a, coefficient = -0.5;
        for (int degree = 2; degree <= 16; ++degree) {
            a_power *= a;
            divided_power = b * divided_power + a_power;
            average += (coefficient / (degree + 1)) * divided_power;
            coefficient *= -double(degree - 1) / (degree * (degree + 1));
        }
        return d * average;
    }
    return thermal_expn(expn, 3, a) - thermal_expn(expn, 3, b);
}

static double thermal_planck(double nu, double log_nu, double temperature, double log_t,
                             double second_radiation, double log_second_radiation,
                             double log_rayleigh_jeans, double numerator) {
    // Same logarithmic Rayleigh-Jeans/Wien partition as radiometry.planck_wavenumber.
    // Coefficients derived from scipy.constants at binding time include cm-1 units.
    const double x = (nu / temperature) * second_radiation;
    // Algebraically equivalent Planck law on normal, bounded support. The
    // spectral numerator is shared by all layers. Extreme values retain the
    // logarithmic path, including subnormal tails and overflow rejection.
    if (x >= 1e-4 && x <= 500.0 && std::isfinite(numerator) &&
        numerator >= std::numeric_limits<double>::min()) {
        const double value = numerator / std::expm1(x);
        if (std::isfinite(value) && value >= std::numeric_limits<double>::min())
            return value;
    }
    const double log_x = log_second_radiation + log_nu - log_t;
    const double correction = log_x < -10.0 ? -x / 2.0 - x * x / 24.0
        : log_x - x - std::log(-std::expm1(-x));
    const double value = std::exp(log_rayleigh_jeans + log_t + 2.0 * log_nu + correction);
    if (!std::isfinite(value))
        throw std::invalid_argument("Planck radiance exceeds finite float64 support");
    return value;
}

static void thermal_kernel(
        const double *grid, const double *temperature, const double *emissivity,
        const double *absorption, const double *aerosol, const double *mu, const double *weights,
        std::ptrdiff_t layers, std::ptrdiff_t points, std::ptrdiff_t angles,
        double surface_temperature, double view_mu, bool analytic, ThermalExpn expn,
        double second_radiation, double rayleigh_jeans,
        double *surface, double *atmospheric, double *reflected) {
    std::vector<double> depth(layers), planck(layers), log_temperature(layers);
    for (std::ptrdiff_t layer = 0; layer < layers; ++layer)
        log_temperature[layer] = std::log(temperature[layer]);
    const double log_surface_temperature = std::log(surface_temperature);
    const double log_second_radiation = std::log(second_radiation);
    const double log_rayleigh_jeans = std::log(rayleigh_jeans);
    for (std::ptrdiff_t point = 0; point < points; ++point) {
        const double log_nu = std::log(grid[point]);
        double numerator = rayleigh_jeans;
        for (double factor : {grid[point], grid[point], second_radiation, grid[point]}) {
            numerator *= factor;
            // Check each intermediate: supplied coefficients can have extreme
            // magnitudes even when their final product is normal.
            if (!std::isfinite(numerator) || numerator < std::numeric_limits<double>::min()) {
                numerator = 0.0;
                break;
            }
        }
        double upward = 0.0, slant_sum = 0.0, below = 0.0, down = 0.0;
        for (std::ptrdiff_t layer = 0; layer < layers; ++layer) {
            const auto i = layer * points + point;
            const double gas_depth = absorption ? absorption[i] : 0.0;
            const double aerosol_depth = aerosol ? aerosol[i] : 0.0;
            depth[layer] = gas_depth + aerosol_depth;
            if (!std::isfinite(gas_depth) || gas_depth < 0.0 ||
                !std::isfinite(aerosol_depth) || aerosol_depth < 0.0 ||
                !std::isfinite(depth[layer]))
                throw std::invalid_argument("Schwarzschild depths and radiances must be finite and nonnegative");
            planck[layer] = thermal_planck(grid[point], log_nu, temperature[layer], log_temperature[layer],
                                         second_radiation, log_second_radiation, log_rayleigh_jeans,
                                         numerator);
            const double slant = depth[layer] / view_mu;
            upward = upward * std::exp(-slant) + planck[layer] * -std::expm1(-slant);
            slant_sum += slant;
            if (analytic && emissivity[point] < 1.0) {
                down += planck[layer] * thermal_e3_interval(expn, below, depth[layer]);
                below += depth[layer];
            }
        }
        const double transmission = std::exp(-slant_sum);
        // Preserve the surface's established radiometry rounding, including
        // exact vacuum agreement; layers amortize the direct spectral numerator.
        surface[point] = emissivity[point] * thermal_planck(
            grid[point], log_nu, surface_temperature, log_surface_temperature,
            second_radiation, log_second_radiation, log_rayleigh_jeans, 0.0) * transmission;
        atmospheric[point] = upward;
        if (analytic) down *= 2.0;
        else if (emissivity[point] < 1.0) {
            for (std::ptrdiff_t angle = 0; angle < angles; ++angle) {
                double ray = 0.0;
                for (std::ptrdiff_t step = 0; step < layers; ++step) {
                    const auto layer = layers - 1 - step;
                    const double slant = depth[layer] / mu[angle];
                    ray = ray * std::exp(-slant) + planck[layer] * -std::expm1(-slant);
                }
                down += (weights[angle] * mu[angle]) * ray;
            }
        }
        reflected[point] = down * ((1.0 - emissivity[point]) * transmission);
        for (double value : {surface[point], atmospheric[point], reflected[point]})
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument("thermal radiance terms must be finite and nonnegative");
    }
}

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
        double second_radiation, double rayleigh_jeans) {
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
    {
        py::gil_scoped_release release;
        thermal_kernel(input(0), input(1), input(2), input(3), input(4), input(5), input(6),
                       layers, points, angles, surface_temperature, view_mu, analytic, expn,
                       second_radiation, rayleigh_jeans, s, a, r);
    }
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

static py::tuple bind_thermal_expn(py::object owner, double second_radiation, double rayleigh_jeans) {
    if (!std::isfinite(second_radiation) || second_radiation <= 0.0 ||
        !std::isfinite(rayleigh_jeans) || rayleigh_jeans <= 0.0)
        throw py::value_error("Planck coefficients must be positive finite scalars");
    // Match the exported fused double ABI explicitly, including Cython's dispatch argument.
    // Each callable retains its interpreter's provider; every evaluation owns its scratch.
    py::capsule capsule = owner.attr("__pyx_capi__")["__pyx_fuse_0expn"];
    void *pointer = PyCapsule_GetPointer(capsule.ptr(), "double (double, double, int __pyx_skip_dispatch)");
    if (!pointer) throw py::error_already_set();
    auto expn = reinterpret_cast<ThermalExpn>(pointer);
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
