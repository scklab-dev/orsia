/* Fixed-cross-section absorption. Contiguous borrowed arrays; owned outputs. */
#include <orsia_numerics/tangent.h>
#include <cmath>
#include <stdexcept>

static double absorption_depth(double sigma, double column) {
    if (!std::isfinite(sigma) || sigma < 0)
        throw std::invalid_argument("cross sections must be finite and nonnegative");
    const double depth = sigma * column;
    if (!std::isfinite(depth))
        throw std::invalid_argument("absorption optical depth overflow");
    return depth;
}

static void validate_absorption_column(double column) {
    if (!std::isfinite(column) || column < 0)
        throw std::invalid_argument("columns must be finite and nonnegative");
}

static void absorption_kernel(
        const double *cross_sections, const double *columns,
        std::ptrdiff_t layers, std::ptrdiff_t points, double *depth) {
    for (std::ptrdiff_t l = 0; l < layers; ++l) {
        const double column = columns[l];
        validate_absorption_column(column);
        for (std::ptrdiff_t p = 0; p < points; ++p) {
            const auto i = l * points + p;
            depth[i] = absorption_depth(cross_sections[i], column);
        }
    }
}

static void absorption_tangent_kernel(
        const double *cross_sections, const double *columns, const double *column_tangent,
        std::ptrdiff_t layers, std::ptrdiff_t points, std::ptrdiff_t directions,
        double *depth, double *tangent) {
    for (std::ptrdiff_t l = 0; l < layers; ++l) {
        const double n = columns[l];
        validate_absorption_column(n);
        for (std::ptrdiff_t d = 0; d < directions; ++d)
            if (!std::isfinite(column_tangent[l * directions + d]))
                throw std::invalid_argument("column tangents must be finite");
        for (std::ptrdiff_t p = 0; p < points; ++p) {
            const auto i = l * points + p;
            const double sigma = cross_sections[i];
            depth[i] = absorption_depth(sigma, n);
            orsia_numerics::scale_tangent(directions, sigma,
                column_tangent + l * directions, 1, tangent + i * directions);
            for (std::ptrdiff_t d = 0; d < directions; ++d)
                if (!std::isfinite(tangent[i * directions + d]))
                    throw std::invalid_argument("absorption tangent overflow");
        }
    }
}
