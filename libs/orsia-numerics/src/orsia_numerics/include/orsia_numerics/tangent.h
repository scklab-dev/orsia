/* Explicit directional arithmetic, binary64, final direction axis.
 * Counts may be zero. Strides are in doubles and may be negative or zero.
 * Inputs must remain stable for the call. Output is contiguous caller-owned
 * storage. Exact in-place aliasing is supported for unit-stride inputs;
 * other overlap is unsupported. No allocation, Python state or physical laws.
 */
#ifndef ORSIA_NUMERICS_TANGENT_H
#define ORSIA_NUMERICS_TANGENT_H
#include <cstddef>

namespace orsia_numerics {
inline void scale_tangent(std::ptrdiff_t count, double factor,
                          const double *input, std::ptrdiff_t stride, double *output) {
    for (std::ptrdiff_t d = 0; d < count; ++d)
        output[d] = factor * input[d * stride];
}

// Chain rule for a product: d(a*b) = b*da + a*db. Also a generic
// two-term linear combination; derivatives may be signed at value boundaries.
inline void product_tangent(std::ptrdiff_t count, double a, const double *da,
                            std::ptrdiff_t a_stride, double b, const double *db,
                            std::ptrdiff_t b_stride, double *output) {
    for (std::ptrdiff_t d = 0; d < count; ++d)
        output[d] = b * da[d * a_stride] + a * db[d * b_stride];
}
}  // namespace orsia_numerics
#endif
