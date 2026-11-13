/* Absorption-only reflected solar physics with explicit directional arithmetic.
 * Included after the established direct-value and compensated-sum kernels.
 */
#include <orsia_numerics/tangent.h>

static void solar_tangent_kernel(
        const double *depth, const double *depth_tangent, const double *irradiance,
        const double *reflectance, const double *reflectance_tangent,
        std::ptrdiff_t layers, std::ptrdiff_t points, std::ptrdiff_t k,
        double zenith, double view_zenith, double *v, double *dv) {
    std::vector<double> layer_depth(layers), layer_tangent(layers), total_tangent(k);
    constexpr double pi = 3.14159265358979323846;
    constexpr double radians_per_degree = pi / 180.0;
    const double mu0 = std::cos(zenith * radians_per_degree);
    const double mu = std::cos(view_zenith * radians_per_degree);
    const double airmass = 1.0 / mu0 + 1.0 / mu;
    for (std::ptrdiff_t p = 0; p < points; ++p) {
        const double incident = irradiance[p], rho = reflectance[p];
        if (!std::isfinite(incident) || incident < 0 ||
            !std::isfinite(rho) || rho < 0 || rho > 1)
            throw std::invalid_argument("solar irradiance or reflectance outside physical bounds");
        for (std::ptrdiff_t l = 0; l < layers; ++l) {
            layer_depth[l] = depth[l * points + p];
            if (!std::isfinite(layer_depth[l]) || layer_depth[l] < 0)
                throw std::invalid_argument("absorption optical depths must be finite and nonnegative");
        }
        const double total = rayleigh_flux_sum(layer_depth.data(), layers);
        if (!std::isfinite(total)) throw std::invalid_argument("summed optical depth overflow");
        const double amplitude = incident * rho * mu0 / pi;
        v[p] = direct_reflected_value(amplitude, total, airmass);
        const double gain = direct_reflected_value(incident * mu0 / pi, total, airmass);
        for (std::ptrdiff_t d = 0; d < k; ++d) {
            if (!std::isfinite(reflectance_tangent[p * k + d]))
                throw std::invalid_argument("reflectance tangents must be finite");
            for (std::ptrdiff_t l = 0; l < layers; ++l) {
                layer_tangent[l] = depth_tangent[(l * points + p) * k + d];
                if (!std::isfinite(layer_tangent[l]))
                    throw std::invalid_argument("optical depth tangents must be finite");
            }
            total_tangent[d] = rayleigh_flux_sum(layer_tangent.data(), layers);
            if (!std::isfinite(total_tangent[d]))
                throw std::invalid_argument("summed optical depth tangent overflow");
        }
        // Product rule at rho=0 and tau=0 uses no division by either value.
        orsia_numerics::product_tangent(k, -(v[p] * airmass),
            reflectance_tangent + p * k, 1, gain, total_tangent.data(), 1, dv + p * k);
        for (std::ptrdiff_t d = 0; d < k; ++d)
            if (!std::isfinite(dv[p * k + d]))
                throw std::invalid_argument("reflected radiance tangent overflow");
    }
}
