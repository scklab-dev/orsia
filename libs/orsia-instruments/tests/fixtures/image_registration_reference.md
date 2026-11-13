# Measured-image registration regression fixture

`image_registration_reference.npz` contains numeric constant-translation, joint-row and interpolation fields extracted from the existing independent Julia source-qualification bundle. The source bundle contains the original 72-by-80 constant-shift and 96-by-96 row-field cases. The fixture retains measured radiance, support and expected estimator/covariance fields used by the Instruments tests.

The original qualification manifest SHA256 is `520823828ec909dcaca49ba01e899c8bc1a919e318475edb492f7b62cbc2dc35`. The NPZ SHA256 is `ba8fbe7957f867ed04a1c452c2e57237448c1342f6bb74d8337aabe62ae201ec`. Extraction used the existing hash-validating reader. The bundle and expected Julia values were not regenerated or modified. The extraction script is retained with the owning migration experiment under `qairbon-e2e/outputs/e2e-completion/`.

Tests retain the original correction and conditional-covariance tolerances. Separate synthetic tests compare signed value/variance propagation and selected-state covariance with dense matrix oracles. These checks establish bounded numerical reproduction. Conditional estimator covariance, full-scene cost and instrument pointing accuracy require separate qualification.
