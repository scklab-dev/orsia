# From atmospheric and spectroscopic files to thermal radiance

This guide explains non-scattering LTE thermal composition. The [radiative library guide](../libs/orsia-radiative/src/orsia_radiative/README.md) gives executable synthetic examples. Public reader and thermal transport regressions remain in the library tests; the [reflected-solar guide](atmosphere-to-radiance.md) describes the companion data path.

## The data path

```text
meteorology.nc                 co2.par + partition.csv
       |                                  |
orsia-data: local readers       orsia-data: local readers
       |                                  |
 native p, T, q                 LineDatabase + Q(T)
       |                                  |
 layer means + CO2 columns                |
       |                                  |
   Atmosphere --------> line_set <---- thermal SpectralGrid
       |                                  |
       +-----------------> line_absorption
                                  |
                         AbsorptionComponent
                                  |
                      compose_absorption_on_grid
                                  |
                              Absorption
                                  |
                  atmospheric_optics
                                  |
                           AtmosphericOptics
                                  |
Atmosphere + thermal surface + view -> RadiativeScene
                                  |
                transfer.thermal.schwarzschild_radiance
                                  |
                               Radiance
```

The file boundary and spectroscopy responsibilities are the same as in the solar example: `orsia-data` decodes files, while `orsia-radiative` constructs molecular columns, evaluates absorption and solves transfer. The transfer solver consumes optical depths and temperatures, never spectroscopic files. The caller chooses the grid and physics; no solar object is created in this branch.

## 1. Native profiles become a layered atmosphere

`read_netcdf_profiles` reads pressure in Pa, temperature in K and specific humidity in kg/kg at the selected time and location. `pressure_level_layer_means` converts five native pressure knots into four layers between 100000 and 20000 Pa, ordered bottom to top. `gas_columns_from_fraction` converts an illustrative 420 ppm dry-air CO2 mole fraction into molecules/m2. `hydrostatic_atmosphere` constructs the `Atmosphere` using those columns and explicit gravity. The humidity is zero; no water absorption is present. Numerical records retain independent writable copies, while intermediate arrays are caller-owned.

## 2. Spectroscopy becomes vertical optical depth

`read_hitran_lines` and `read_partition_table` read an invented transition and Q(T) table. `line_set` prepares their layer-dependent inputs from the atmosphere and molecular mass. `line_absorption` evaluates Voigt absorption, producing dimensionless vertical optical depths `tau[layer, point]`; `compose_absorption_on_grid` sums the selected components, and `atmospheric_optics` binds the result to the atmosphere and grid.

The thermal grid is 1000–1002 cm-1, near 10 micrometres, with 1001 points. The invented line is centred at 1001 cm-1 and uses an explicit 25 cm-1 wing cutoff with declared line-centre coverage of 974–1028 cm-1. This is a format/design demonstration, not a real CO2 band or a spectroscopic reference. The 0.002 cm-1 step is numerical sampling, not instrument resolution. Switching from the solar grid requires fresh line materialisation and absorption evaluation, not reuse of its sampled optical depths.

## 3. Temperatures provide the thermal sources

In local thermodynamic equilibrium (LTE), an isothermal absorbing layer has source radiance `B(nu, T_layer)`, the Planck function. The same absorption that removes incident radiation also determines how much the layer emits. Spectroscopy supplies absorption; the transfer solver combines it with Planck radiance from the supplied temperatures. Layer temperatures are about 285, 272.5, 257.5 and 240 K; the surface is 300 K.

The opaque Lambertian surface has emissivity 0.97 on the same grid, hence reflectivity 0.03. It emits `0.97 * B(nu, 300 K)` and reflects part of the downward atmospheric thermal irradiance. There is no incident radiation at the top of the truncated column and no Sun. The view zenith is 10 degrees.

```python
scene = RadiativeScene(
    atmosphere,
    SurfaceState(grid, emissivity=np.full(1001, 0.97), temperature_k=300.0),
    ObservationGeometry(10.0),
)
radiance = thermal.schwarzschild_radiance(
    scene, optics=optics, hemispheric_method="expn"
)
terms = thermal.schwarzschild_terms(
    scene, optics=optics, hemispheric_method="expn"
)
```

`Radiance` contains only the grid, total values and geometry. `schwarzschild_terms` separately returns fresh writable arrays named `surface_emission`, `atmospheric_emission` and `reflected_atmospheric_emission`. Each has shape `(1001,)` and units W m-2 sr-1 per (cm-1); the total is their sum. The solver has no retained source, provider, cache or implicit parallel policy. See the [thermal API contract](../libs/orsia-radiative/README.md#lte-thermal-transfer).

## 4. What reaches the top?

Let `mu = cos(view_zenith)`, `tau_total` be the sum of the vertical layer depths, and `tau_above[k]` the depth above layer k. For the supplied isothermal layers, direct emission is

```text
surface_emission     = epsilon * B(nu, T_surface) * exp(-tau_total / mu)
atmospheric_emission = sum_k B(nu, T_k)
                            * [1 - exp(-tau_k / mu)]
                            * exp(-tau_above[k] / mu)
```

The third term is `(1 - epsilon) * (F_down / pi) * exp(-tau_total / mu)`, where `F_down` is downward atmospheric irradiance at the surface. This is a hemispheric integral over atmospheric emission, not the intensity along the viewing ray. The `expn` method evaluates it with E3 exponential integrals; the alternative uses angular Gauss–Legendre quadrature, whose order must be refined separately.

Unlike the solar example, the atmosphere is not merely an attenuator: it emits along the path. Increasing absorption suppresses the warm surface contribution while increasing the contribution from cooler layers. This particular temperature profile produces a line-centre dip; that is not universal. A sufficiently warm emitting layer can instead produce an emission feature.

## Checks and limits

The script checks the molecular-column conversion, optical-depth shape, independent frequency-density Planck expression with the `100*c` Jacobian to per-cm-1 units, explicit layer-sum surface/atmospheric formulas and analytic E3 surface reflection. It checks the total against the three separate terms, compares 128-node angular quadrature with E3 for this case, verifies the transparent-atmosphere limit, and checks finite nonnegative output below the hottest blackbody spectrum and a line-centre dip. These are bounded numerical checks; the reflection reference also uses SciPy E3 and is not an independent external solver.

The atmosphere is dry, CO2-only and truncated at 20000 Pa; every file is synthetic. No authentic-provider compatibility, spectral convergence, complete atmospheric physics, instrument accuracy, retrieval performance or concurrent native-library safety is established. The closure is plane-parallel, LTE and non-scattering: it rejects nonzero atmospheric scattering and does not mix solar and thermal radiation. Real thermal work needs selected gas/continuum data, compatible wing conventions, supported temperatures and pressures, a complete column and a justified surface model. Clouds and ice remain outside Orsia's scope.
