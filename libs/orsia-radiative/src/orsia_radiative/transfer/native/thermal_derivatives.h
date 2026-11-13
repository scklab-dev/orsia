/* LTE physical partial derivatives; included after Planck and E3 intervals. */
#include <orsia_numerics/tangent.h>

static double thermal_planck_temperature(double value, double nu, double temperature,
                                         double second_radiation, double rayleigh_jeans) {
    const double x = (nu / temperature) * second_radiation;
    const double factor = x == 0.0 ? 1.0 : x / -std::expm1(-x);
    const double scaled = value / temperature;
    const double direct = scaled * factor;
    if (std::isfinite(direct) && scaled >= std::numeric_limits<double>::min() &&
        direct >= std::numeric_limits<double>::min()) return direct;
    // dB/dT = RJ*nu^2*[x/(2*sinh(x/2))]^2. Its logarithmic form
    // preserves derivatives when B or B/T underflows before multiplication.
    const double log_x = std::log(second_radiation) + std::log(nu) - std::log(temperature);
    const double correction = x < 1e-4 ? -x * x / 12.0
        : 2.0 * log_x - x - 2.0 * std::log(-std::expm1(-x));
    return std::exp(std::log(rayleigh_jeans) + 2.0 * std::log(nu) + correction);
}

// E2(a)-E2(a+d) = integral_a^(a+d) E1(x) dx. Preserve sub-ulp d.
static double thermal_e2_interval(ThermalExpn expn, double a, double d) {
    if (d == 0.0 || a >= 745.0) return 0.0;
    const double b = a + d;
    if (a == 0.0) return -std::expm1(-d) + d * thermal_expn(expn, 1, d);
    if (d <= 0.02 * std::min(a, 1.0)) {
        const double m = a + d / 2.0, ratio = d / m;
        const double correction = std::exp(-m) * (
            (d * ratio + ratio * ratio) / 24.0 +
            (d * d * d * ratio + 3.0 * d * d * ratio * ratio +
             6.0 * d * ratio * ratio * ratio + 6.0 * ratio * ratio * ratio * ratio) / 1920.0);
        return d * (thermal_expn(expn, 1, m) + correction);
    }
    if (b <= 0.5)
        return std::exp(-a) * -std::expm1(-d) + b * expn(1, b, 1) - a * expn(1, a, 1);
    return thermal_expn(expn, 2, a) - thermal_expn(expn, 2, b);
}

// Directional recurrence for incoming*t + B*(1-t), t=exp(-tau/mu).
static void thermal_ray_tangent(std::ptrdiff_t k, double incoming, double planck,
        double transmission, double emission_weight, double mu,
        const double *dplanck, const double *ddepth, double *dray, double *scratch) {
    orsia_numerics::scale_tangent(k, transmission, dray, 1, dray);
    orsia_numerics::product_tangent(k, (transmission * (planck - incoming)) / mu,
        dplanck, 1, emission_weight, ddepth, 1, scratch);
    orsia_numerics::product_tangent(k, 1.0, dray, 1, 1.0, scratch, 1, dray);
}
