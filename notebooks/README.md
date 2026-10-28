# Radiance and instrument notebooks

The two canonical notebooks combine the public Orsia libraries with locally acquired atmospheric inputs. [Five-band transport](era5_hitran_sasktran2_five_bands.ipynb) compares Orsia with an isolated SASKTRAN2 installation for UV, VNIR, SWIR, MWIR and LWIR. [Instrument responses](era5_hitran_instrument_responses.ipynb) evaluates six illustrative SWIR optical layouts on a synthetic ground image using the same atmospheric preparation. The [scenario snapshot](data/gesat-toulouse-2025.json) contains derived orbit geometry for the illustrative Toulouse scene, with source and qualification recorded in that file.

## Local inputs and environment

Run these commands from a clone's repository root after installing Devbox and configuring Nix. Copy `.env.example` to `.env` and supply your own CDS API URL and key. The acquisition script reads the ignored root `.env` using its source-file location. Public HITRAN, LISIRD, MIPAS, Zenodo and Bremen acquisitions require network access and their respective provider availability. Preparation writes provider data, checksums and manifests under ignored `outputs/`.

```sh
cp .env.example .env
# Edit .env with your own credentials before the acquisition step.
devbox run -- just setup
devbox run -- just setup-sasktran2-reference
devbox run -- just prepare-five-band
devbox run -- just prepare-five-band-enriched --download
devbox run -- just prepare-five-band-enriched
```

`prepare-five-band` acquires ERA5 for 2024-01-15 12:00 UTC at 0° latitude and longitude, HITRAN lines and LISIRD TSIS1 HSRS. It writes `outputs/five-band-inputs/manifest.json`. `prepare-five-band-enriched --download` adds MIPAS and UV cross sections and LWIR lines. Repeating the enrichment recipe without `--download` checks the completed receipt offline. Provider editions marked `current` can change; local receipts retain checksums for the downloaded bytes. The [enrichment guide](../docs/five-band-enrichment.md) describes the spectroscopy and conversion choices.

## Execution and offline checks

```sh
devbox run -- just test-notebooks
devbox run -- just execute-five-band
devbox run -- just execute-instrument-responses
devbox run -- just replay-instrument-response
```

`test-notebooks` runs focused local software checks on synthetic inputs. The two `execute-*` recipes require the prepared enriched manifest; five-band execution also requires the isolated SASKTRAN2 environment. These recipes register a project-local kernelspec under `outputs/jupyter/` and execute the tracked notebooks in place. A successful execution replaces notebook displays and execution metadata. Instrument execution writes `outputs/instrument-responses/` with its archive, evidence receipt and extracted figures. The offline replay uses those locally generated archives as its reference, evaluates current instrument cells on their spectral scene, and writes the comparison under `outputs/ground-response-integration/`. Replay requires an instrument execution in this clone and uses its locally saved spectral inputs.

The tracked notebooks include saved figures and counts from earlier runs. Some new instrument cells retain source-only status with empty output; the earlier displays and timings describe their original execution. A fresh run updates the execution evidence. Replay compares instrument outputs only against a reference produced locally. These examples and checks establish bounded software composition; instrument-space agreement, retrieval performance and engineering feasibility need their own evidence.

## Provenance and attribution

The saved notebook graphics are author-created outputs derived in part from ERA5 and HITRAN inputs. The instrument scene also uses the [Space-Track-derived scenario](data/gesat-toulouse-2025.json). The publication owner confirms rights and applicable attributions for these historical saved figures in the complete notebooks. The repository's Apache-2.0 declaration covers Orsia notebook code and author-created figures; provider terms and attribution apply to the underlying inputs.

- **ERA5:** The notebooks use a pressure-level profile for 2024-01-15 12:00 UTC at 0° latitude and longitude. The [ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) lists DOI `10.24381/cds.bd0915c6` and a [CC-BY licence](https://cds.climate.copernicus.eu/licences/cc-by). Attribution for the derived figures: “Contains modified Copernicus Climate Change Service information (ERA5 profile dated 2024-01-15).” The notebook authors alone bear responsibility for use of Copernicus information or data in these notebooks. The [ECMWF citation guide](https://confluence.ecmwf.int/spaces/CKB/pages/197704114/How+to+acknowledge+and+cite+a+Climate+Data+Store+CDS+catalogue+entry+and+the+data+published+as+part+of+it) calls for dataset citation and Copernicus attribution. The [Copernicus product licence](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) excludes datasets flagged with separate terms (clause 1.5); section 5 specifies attribution for products covered by that licence. The local manifest records the ERA5 request and labels the edition `current`.
- **HITRAN:** The spectral inputs come from HITRAN. The [HITRAN citation policy](https://hitran.org/citepolicy/) requests the article for the database edition actually used. The local manifest labels the acquisition `current`; verify its actual edition with the provider before selecting an article to cite.
- **Scenario geometry:** The [scenario snapshot](data/gesat-toulouse-2025.json) records Space-Track GP_HISTORY elements propagated with Skyfield. [Space-Track documentation](https://www.space-track.org/documentation) requires appropriate citation for published analyses based on USSPACECOM data. Cite Space-Track alongside the scenario geometry.

Local acquisitions also include LISIRD, MIPAS, Zenodo and Bremen material. Redistribution of separately acquired provider data follows the applicable provider terms. The [candidate manifest](../CANDIDATE-MANIFEST.md#license-and-scope) records the owner's confirmation for the included historical saved figures.
