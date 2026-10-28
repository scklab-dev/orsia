# Orsia

Orsia is a work in progress. Its six independently installable scientific libraries support explicit numerical execution, uncertainty propagation, radiative transfer, scientific data access, instrument simulation and numerical inversion. The repository provides software components and synthetic examples; scientific validation of instrument-space agreement, retrieval performance and engineering feasibility is a separate activity.

## Libraries

| Distribution | Scope | Guide |
| --- | --- | --- |
| `orsia-numerics` | Explicit CPU execution and numerical storage | [API and examples](libs/orsia-numerics/README.md) |
| `orsia-uncertainty` | Covariance, signed bias and ensembles | [API and examples](libs/orsia-uncertainty/README.md) |
| `orsia-radiative` | Atmospheric and surface optics, solar and thermal transfer | [API and examples](libs/orsia-radiative/README.md) |
| `orsia-data` | Explicit acquisition and scientific file readers | [API and examples](libs/orsia-data/README.md) |
| `orsia-instruments` | Radiance sampling, optics, photoelectrons and detector readout | [API and examples](libs/orsia-instruments/README.md) |
| `orsia-inverse` | Caller-supplied forward prediction and fixed-covariance estimation | [API and examples](libs/orsia-inverse/README.md) |

Each library contains a `pyproject.toml`, `src/`, `tests/`, API reference and editable Draw.io diagrams with SVG/PNG exports. The root `pyproject.toml` defines a development-only UV workspace. The six library distributions each declare their runtime requirements.

## Development Environment

Install Devbox and configure Nix. From this directory, run:

```bash
devbox run -- just setup
devbox run -- just check
```

`setup` provisions the locked CPython 3.14 free-threaded workspace in a local `.venv`. `check` runs formatting, Ruff, Pyrefly, dependency boundaries, free-threaded runtime checks and the software test suite with a 90% combined statement-and-branch coverage threshold for the workspace and each library. `devbox run -- just build` builds the six wheels. `devbox run -- just clean-install` checks installation outside workspace resolution.

The [radiative composition guide](docs/atmosphere-to-radiance.md), [thermal composition guide](docs/atmosphere-to-thermal-radiance.md), [data acquisition guide](docs/atmospheric-data.md), [data products guide](docs/data-products.md), [data services guide](docs/data-services.md), [solar spectrum guide](docs/hsrs.md) and [MIPAS reference guide](docs/mipas-reference-atmospheres.md) explain the library boundaries and examples.

## License

Orsia source, tests, synthetic fixtures, documentation and diagrams, including their SVG/PNG exports, are licensed under Apache-2.0. See the root [LICENSE](LICENSE) and each library's `LICENSE`.
