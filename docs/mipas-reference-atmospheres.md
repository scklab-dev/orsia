# MIPAS 2001 reference atmospheres

`orsia-data` provides explicit acquisition of five historical reference atmospheres and local decoding of a bounded molecular subset of RFM text. Native point levels and concentrations remain available for caller-selected physical calculations. Radiative layer construction, interpolation, integration, composition closure and concentration scaling are separate scientific operations.

## Acquisition and local reading

```python
from pathlib import Path

from orsia_data.atmosphere.rfm import read_rfm_atmosphere
from orsia_data.services.mipas import download_mipas_profile

# Explicit network operation; choose a fresh destination.
path = download_mipas_profile(
    edition="2001",
    profile="equatorial_day",
    destination=Path("outputs/equatorial-2001.atm"),
)

# Local reading, also usable in a later offline session.
altitude_m, pressure_pa, temperature_k, gas_mole_fractions = read_rfm_atmosphere(path)
water_mol_mol = gas_mole_fractions["H2O"]
co2_mol_mol = gas_mole_fractions["CO2"]
```

Both functions are also exported from `orsia_data`; the reader is exported from `orsia_data.atmosphere`. The return value is `(altitude_m, pressure_pa, temperature_k, gas_mole_fractions)`: three vectors and a dictionary mapping uppercase RFM molecular labels to vectors. This follows the existing native-reader array contracts. Physical `Atmosphere` materialisation remains owned by `orsia-radiative`.

Each vector has shape `(n_levels,)`, owns an independent writable `numpy.float64` buffer and is detached from the file, other vectors and other calls. Decimal text is decoded in double precision before unit conversion. Caller mutations affect only the returned buffers and dictionary. Keep the source file stable during reading. The reader closes its file before returning and performs local decoding only.

`download_mipas_profile(*, edition="2001", profile="equatorial_day", destination, expected_sha256=None, timeout_s=30.0, max_bytes=134217728)` returns the destination `Path`. The edition and profile defaults select the historical equatorial daytime profile; explicit arguments make that choice visible at the call site. Supported selections resolve to the Oxford collection as follows:

| Profile | Native file |
| --- | --- |
| `midlatitude_day` | `day.atm` |
| `midlatitude_night` | `ngt.atm` |
| `polar_winter` | `win.atm` |
| `polar_summer` | `sum.atm` |
| `equatorial_day` | `equ.atm` |

The downloader uses `acquisition.download_file` for bounded streaming, optional lowercase SHA-256 verification and atomic no-clobber publication. Existing files and symlinks raise `FileExistsError` and remain intact, including a destination created during transfer. Transfer failures clean temporary files; created parent directories can remain. Unsupported edition/profile choices raise `ValueError` before filesystem or network access. The timeout applies to network operations. The download transfers original bytes; the separate reader establishes format validity. Transport exceptions propagate. Retained native bytes and caller-recorded digests provide reproducible offline replay; the provider controls the mutable URLs.

## Supported RFM subset

The upstream [RFM atmospheric format](https://eodg.atm.ox.ac.uk/RFM/sum/atmfil.html) defines block-labelled profiles and standard units. This reader implements the following bounded subset:

- UTF-8 text, optional blank records and full-line `!` comments between blocks, followed by a positive integer level count. Trailing text on the count record is descriptive. A single level represents a homogeneous sample.
- Each field starts on its own record with `*LABEL`, followed by exactly the declared number of values across one or more records. Field order is free. Blank value records are accepted. Comment records belong outside value blocks.
- Labels are case-insensitive and separated from descriptive suffixes by whitespace. Supported labels fit within the upstream 20-character bound. Unknown or overlong labels raise `ValueError`.
- `HGT`, `PRE` and `TEM` are each required exactly once. Every molecular label is unique after case normalization. A geometry/thermodynamics-only file returns an empty gas dictionary.
- A mandatory `*END` record closes the profiles. A descriptive suffix on that record and subsequent blank/comment records are accepted. Subsequent data records raise `ValueError`.
- Numeric values support decimal notation, signed `E`/`D` exponents, whitespace and single comma separators. Repetition expressions, empty comma fields, slash termination and inline data comments raise `ValueError`.
- Bracketed units are optional because RFM fixes the native units. Recognized annotations, case-insensitively, are `[km]` for height, `[mb]`/`[mbar]`/`[hPa]` for pressure, `[K]` for temperature and `[ppmv]` for gases. Conflicting, malformed or repeated annotations raise `ValueError`. Parenthesized descriptions after whitespace, such as `*F14 (CF4) [ppmv]`, remain descriptive.

The molecular vocabulary is `N2`, `O2`, `CO2`, `O3`, `H2O`, `CH4`, `N2O`, `HNO3`, `CO`, `NO2`, `N2O5`, `CLO`, `HOCL`, `CLONO2`, `NO`, `HNO4`, `HCN`, `NH3`, `F11`, `F12`, `F14`, `F22`, `CCL4`, `COF2`, `H2O2`, `C2H2`, `C2H6`, `OCS`, `SO2` and `SF6`. These are the 30 species in the selected collection. Labels `AEROSOL`, isotopologue qualifiers such as `H2O(4)`, vibrational qualifiers such as `CO2(1)(2)`, and additional species raise `ValueError`: their scientific interpretation requires separate support. RFM's broader free-format grammar and ignored-field behavior remain outside this subset.

Conversions are km × 1000 → m, hPa/mb/mbar × 100 → Pa and ppmv × 10⁻⁶ → mol/mol. Temperature remains in kelvins. Gas fractions use total air, including water, as the denominator. Native levels and their order are preserved. Every decoded value and SI coordinate must be finite; altitude must increase strictly, pressure must be positive and decrease strictly, temperature must be positive, and each individual gas fraction must lie in `[0, 1]`. Finite negative altitudes are accepted. Duplicate/missing fields, malformed data, truncation, invalid units and physical-support violations raise `ValueError`; filesystem and Unicode errors propagate.

The sum of tabulated species remains unchanged, including values above one. The format documentation describes a warning for such sums, and all five acquired files contain them. The reader validates individual fractions while preserving the supplied composition. Scientific calculations requiring composition closure must select and document their treatment explicitly.

## Source and validation evidence

The [Oxford collection](https://eodg.atm.ox.ac.uk/RFM/atm/) identifies the MIPAS model atmospheres as the 2001 collection supplied by the University of Leicester, with 121 levels from 0 to 120 km at 1 km spacing. Its historical CO2 values belong to that reference epoch. Collection and format pages were inspected on 17 October 2026; their displayed revision dates were 12 February 2026 and 17 January 2026 respectively.

The explicit five-file probe on 17 October 2026 observed the following byte identities and sums of all 30 molecular fractions. Every native HGT/PRE/TEM and gas value agreed exactly with an independent block extraction after the declared SI conversions.

| Profile | Bytes | Minimum sum | Maximum sum | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `midlatitude_day` | 42274 | 0.8032301822 | 1.0130309041 | `d3a8d9a664e45e587c336ef64ee7b5c57e8e4f1191b0ce98ce38fc061860464a` |
| `midlatitude_night` | 42276 | 0.8032308822 | 1.0130309038 | `539da9443f6d9d61610922c8b698ef8a7c147488bbcf827d58cf3ae8e1a76b4c` |
| `polar_winter` | 42270 | 0.8034446839 | 1.0043769063 | `5d230b97d33b7101b34beaaa8867ebd6ae01e893afbc3faefc243a7b54f45e1f` |
| `polar_summer` | 42270 | 0.8031835511 | 1.0091588181 | `27c01e6bd5d5469929707feb7cba62df0bc3efd2b54e09a3f16841eb7c2cf7fe` |
| `equatorial_day` | 42270 | 0.8033013288 | 1.0286207135 | `4f366cfc5de2bfde21a82ab27c5c1a3ad9cd2e25ee3296bb5d6109980196a9f6` |

Each file has 80 levels with a sum above one. These observations establish reader interoperability for the listed snapshots. Atmospheric accuracy, present-day representativeness, retrieval performance, future endpoint availability and concurrent-call qualification require their own evidence. File redistribution rights remain to be established; authentic validation inputs stay in ignored local storage, and committed fixtures are synthetic.

Run the focused offline tests with `devbox run -- just test-mipas`. The standalone live probe and replay command are retired; acquisition and local reading remain available through the public functions shown above. Historical file identities in this section describe their original observations, not a fresh provider qualification.
