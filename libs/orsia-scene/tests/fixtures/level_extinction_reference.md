# Level-grid extinction regression fixture

`level_extinction_reference.npz` contains aligned numeric inputs and pre-transfer Python outputs for the bounded QAIRBON source-rendering case. The inputs combine a provider-derived compact atmosphere, terrain and surface with artificial spectroscopy and solar spectra. Spatial dimensions are 16 rows by 17 columns. The archive retains layout-one response arrays and reference, plume and separate-solar-path output fields.

The oracle was evaluated from the immutable pre-turn `scene/optical_depth.py` and `scene/radiometry.py` sources, before the new Scene evaluator was used for comparison. The source snapshot, hashes, extraction command and generation script are retained in the owning migration experiment at `qairbon-e2e/outputs/e2e-completion/`. Existing Julia reference bundles were read and hash-verified; they were not regenerated or modified.

The input rendering manifest SHA256 is `1ec755cfb54e5740574285ad2a44031d064108e35b1314b5900838a3cccac26b`. The retained NPZ SHA256 is `c935ac852ae4ab782a8901186e0f746bece2d70f9206dce3cc42dd92c5370ab8`.

Tests require exact field equality for the unchanged block size and check independent repeated/parallel outputs. Separate expansion/cube comparisons use their established floating-point allowance. The fixture establishes bounded method-reproduction evidence. Physical spectroscopy, retrieval accuracy, complete provider-scene cost and general native-library concurrency remain outside this fixture's scope.
