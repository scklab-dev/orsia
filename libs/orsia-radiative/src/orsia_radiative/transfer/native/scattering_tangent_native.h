/* Fixed-support SOS directional derivatives. Reuse the primal phase, formal
 * coefficients and dense redistribution; differentiate the physical recurrence
 * explicitly. Directions run serially with bounded call-local scratch, rather
 * than retaining every order or allocating a phase Jacobian for all seeds.
 */
#include <orsia_numerics/tangent.h>

static double transport_phi_tangent(double x, double dx) {
    // The series preserves the transparent limit without cancellation. For
    // opaque cells, form the directional product before phi' can underflow.
    if (x < 0.01)
        return (-0.5 + x * (1.0/3.0 + x * (-1.0/8.0 + x * (1.0/30.0 +
            x * (-1.0/144.0 + x * (1.0/840.0 - x/5760.0)))))) * dx;
    return (std::exp(-x) - phi(x)) * (dx / x);
}

static double transport_integral_tangent(double a, double b, double da, double db) {
    if (a == b) return std::exp(a) * (0.5 * da + 0.5 * db);
    const bool upper_b = b > a;
    const double x = std::abs(b - a), dx = upper_b ? db - da : da - db;
    return std::exp(std::max(a, b)) *
        ((upper_b ? db : da) * phi(x) + transport_phi_tangent(x, dx));
}

static double beam_weight_tangent(double a, double b, double da, double db,
        double depth, double dd, double direction, double mu0, bool up, double fraction) {
    if (std::isfinite(a) && std::isfinite(b))
        return transport_integral_tangent(a, b, da, db) / (direction / fraction);
    const double weight = beam_source_weight(a, b, depth, direction, mu0, up, fraction);
    if (weight == 0.0) return 0.0;
    return weight * ((b > a ? db : da) - dd / depth);
}

static double transport_phase_derivative(double cosine, double g, int moments) {
    const double x = std::clamp(cosine, -1.0, 1.0);
    if (!moments || g <= 0.0) {
        const double q = 1.0 + g*g - 2.0*g*x;
        return -2.0*g/std::pow(q, 1.5) - 3.0*(1.0-g*g)*(g-x)/std::pow(q, 2.5);
    }
    const double f = std::pow(g, moments), df = moments * std::pow(g, moments-1);
    auto coefficient = [&](int l) {
        const double power = std::pow(g, l);
        const double derivative = l ? l * std::pow(g, l-1) : 0.0;
        return (2.0*l+1.0) * ((derivative-df) + (power-f)*df/(1.0-f)) / (1.0-f);
    };
    double c0 = coefficient(moments-2), c1 = coefficient(moments-1);
    int nd = moments;
    for (int i = 3; i <= moments; ++i) {
        const double old = c0;
        --nd;
        c0 = coefficient(moments-i) - c1*(nd-1.0)/nd;
        c1 = old + c1*x*(2.0*nd-1.0)/nd;
    }
    return c0 + c1*x;
}

static void prepare_transport_tangent(
        const double *depth, const double *dd, const double *mu,
        const double *t, const double *ht, double *dt, double *dht,
        double *dw, double *dhw, double *dsw, double *dshw,
        double mu0, std::ptrdiff_t cells, std::ptrdiff_t directions) {
    double above = 0.0, dabove = 0.0;
    for (std::ptrdiff_t step = 0; step < cells; ++step) {
        const auto c = cells-1-step;
        const double top = -above/mu0, bottom = -(above+depth[c])/mu0;
        const double mid = -(above+depth[c]/2.0)/mu0;
        const double dtop = -dabove/mu0, dbottom = -(dabove+dd[c])/mu0;
        const double dmid = -(dabove+dd[c]/2.0)/mu0;
        for (std::ptrdiff_t d = 0; d < directions; ++d) {
            const auto i = c*directions+d;
            const double m = std::abs(mu[d]), x = depth[c]/m, dx = dd[c]/m;
            dt[i] = t[i] == 0.0 ? 0.0 : -t[i]*dx;
            dht[i] = ht[i] == 0.0 ? 0.0 : -ht[i]*(dx/2.0);
            if (!std::isfinite(x) || !std::isfinite(dx)) {
                // Differentiate the opaque vertical-depth quotient before
                // either slant depth or its directional derivative overflows.
                const double ratio = dd[c] / depth[c];
                dw[i] = (std::isfinite(x) ? x * t[i] + std::expm1(-x) : -1.0) * ratio / depth[c];
                dhw[i] = (std::isfinite(x) ? (x/2.0) * ht[i] + std::expm1(-x/2.0) : -1.0) * ratio / depth[c];
            } else {
                dw[i] = transport_phi_tangent(x,dx)/m;
                dhw[i] = transport_phi_tangent(x/2.0,dx/2.0)/(2.0*m);
            }
            const bool up = mu[d] > 0;
            const double entry = up ? bottom : top, end = up ? top : bottom;
            const double de = up ? dbottom : dtop, dend = up ? dtop : dbottom;
            dsw[i] = beam_weight_tangent(entry-x, end, de-dx, dend,
                depth[c], dd[c], m, mu0, up, 1.0);
            dshw[i] = beam_weight_tangent(entry-x/2.0, mid, de-dx/2.0, dmid,
                depth[c], dd[c], m, mu0, up, 0.5);
        }
        above += depth[c]; dabove += dd[c];
    }
}

static void apply_transport_tangent(
        const double *mu, const double *source, const double *ds,
        const double *t, const double *dt, const double *w, const double *dw,
        const double *ht, const double *dht, const double *hw, const double *dhw,
        double *field, double *df, double *exits, double *de,
        double bottom, double dbottom, std::ptrdiff_t cells, std::ptrdiff_t directions) {
    // This is the derivative companion of apply_kernel. Compute the incoming
    // value and tangent together so zero primal terms still carry derivatives.
    for (std::ptrdiff_t d = 0; d < directions; ++d) {
        double incoming = mu[d] > 0 ? bottom : 0.0;
        double di = mu[d] > 0 ? dbottom : 0.0;
        for (std::ptrdiff_t step = 0; step < cells; ++step) {
            const auto c = mu[d] > 0 ? step : cells-1-step, i = c*directions+d;
            if (field) {
                field[i] = incoming*ht[i] + source[i]*hw[i];
                df[i] = di*ht[i] + incoming*dht[i] + ds[i]*hw[i] + source[i]*dhw[i];
            }
            double propagated, emitted;
            orsia_numerics::product_tangent(1, incoming, &di, 1, t[i], dt+i, 1, &propagated);
            orsia_numerics::product_tangent(1, source[i], ds+i, 1, w[i], dw+i, 1, &emitted);
            di = propagated + emitted;
            incoming = incoming*t[i] + source[i]*w[i];
        }
        exits[d] = incoming; de[d] = di;
    }
}

static py::tuple scattering_orders_tangent(
        py::tuple optical, py::tuple seeds, py::object irradiance,
        py::object reflectance, py::object reflectance_seed,
        py::object directions_object, py::object weights_object,
        py::ssize_t layers, py::ssize_t subdivisions, py::ssize_t max_order,
        int moments, double zenith, double view_zenith, double azimuth, Dgemm gemm) {
    if (seeds.size() != 5) throw py::value_error("scattering tangent requires five seed roles");
    std::array<py::buffer_info, 5> sb;
    auto rb = py::reinterpret_borrow<py::buffer>(reflectance_seed).request();
    auto validate = [](const py::buffer_info &b, int rank) {
        if (b.itemsize != sizeof(double) ||
            (b.format != "d" && b.format != "=d" && b.format != "@d"))
            throw py::value_error("scattering tangent inputs must be native float64");
        if (b.ndim != rank) throw py::value_error("scattering tangent input rank mismatch");
    };
    validate(rb, 2);
    const auto points = rb.shape[0], k = rb.shape[1];
    if (points < 1 || layers < 1) throw py::value_error("scattering tangent requires nonempty support");
    for (int role = 0; role < 5; ++role) {
        if (seeds[role].is_none()) continue;
        sb[role] = py::reinterpret_borrow<py::buffer>(seeds[role]).request();
        validate(sb[role], 3);
        if (sb[role].shape[0] != layers || sb[role].shape[1] != points || sb[role].shape[2] != k)
            throw py::value_error("scattering tangent layer, spectral or direction support mismatch");
    }
    // The existing implementation validates every primal role and control and
    // defines the returned value, including its negative Delta-M order policy.
    auto values = spectral_orders(optical, irradiance, reflectance, directions_object,
        weights_object, layers, subdivisions, max_order, moments, zenith, view_zenith, azimuth, gemm);
    if (values.shape(1) != points) throw py::value_error("scattering tangent spectral support mismatch");
    const auto size = checked_product(max_order+1, points);
    checked_product(size, k);
    py::array_t<double> tangents({max_order+1, points, k});
    if (k == 0) return py::make_tuple(values,tangents);
    auto *output = tangents.mutable_data();
    std::array<py::buffer_info, 5> ob;
    for (int role = 0; role < 5; ++role) {
        if (optical[role].is_none()) continue;
        ob[role] = py::reinterpret_borrow<py::buffer>(optical[role]).request();
    }
    auto ib = py::reinterpret_borrow<py::buffer>(irradiance).request();
    auto rho = py::reinterpret_borrow<py::buffer>(reflectance).request();
    auto vb = py::reinterpret_borrow<py::buffer>(directions_object).request();
    auto wb = py::reinterpret_borrow<py::buffer>(weights_object).request();
    const auto angular = vb.shape[0], directions = angular+1;
    const auto cells = max_order ? checked_product(layers, subdivisions) : 0;
    const auto count = checked_product(cells, directions);
    if (max_order && !angular) throw py::value_error("scattering tangents require angular support");
    {
        py::gil_scoped_release release;
        constexpr double pi = 3.14159265358979323846, rad = pi/180.0;
        const double mu0 = std::cos(zenith*rad), muv = std::cos(view_zenith*rad);
        std::vector<double> mu(directions), vectors(directions*3), weights(angular),
            beam_cos(directions), depth(cells), dd(cells), ray(layers), dray(layers),
            particle(layers), dparticle(layers), g(layers), dg(layers),
            t(count), ht(count), w(count), hw(count), sw(count), shw(count),
            dt(count), dht(count), dw(count), dhw(count), dsw(count), dshw(count),
            zero(count), field(count), df(count), source(count), ds(count),
            beam(count), dbeam(count), bf(count), dbf(count), be(directions), dbe(directions),
            bc(count), dbc(count), bx(directions), dbx(directions),
            exits(directions), de(directions), flux(angular), dflux(angular),
            matrix(checked_product(angular, directions)), dmatrix(matrix.size()), rayleigh_matrix(matrix.size()),
            temp(count), dtemp(count), tx(directions), dtx(directions), term(count);
        for (py::ssize_t d = 0; d < directions; ++d) {
            vectors[d*3] = d < angular ? strided_value(vb,d,0) : std::sin(view_zenith*rad)*std::cos(azimuth*rad);
            vectors[d*3+1] = d < angular ? strided_value(vb,d,1) : std::sin(view_zenith*rad)*std::sin(azimuth*rad);
            mu[d] = vectors[d*3+2] = d < angular ? strided_value(vb,d,2) : muv;
            beam_cos[d] = vectors[d*3]*std::sin(zenith*rad)-mu[d]*mu0;
            if (d < angular) weights[d] = strided_value(wb,d);
        }
        for (py::ssize_t j = 0; j < angular; ++j)
            for (py::ssize_t d = 0; d < directions; ++d) {
                double cosine = 0.0;
                for (int axis = 0; axis < 3; ++axis) cosine += vectors[j*3+axis]*vectors[d*3+axis];
                rayleigh_matrix[j*directions+d] = weights[j]*transport_phase(cosine,0,0,true);
            }
        // Cache only seed-independent P(g) and dP/dg, never seeded dP. Equal
        // primal g values with different direction seeds share these safely.
        std::deque<std::pair<double, std::array<std::vector<double>, 2>>> cache;
        std::array<std::vector<double>, 2> oversized;
        constexpr std::size_t budget = 16*1024*1024;
        auto phase_partials = [&](double parameter) -> const std::array<std::vector<double>, 2> & {
            for (const auto &entry : cache) if (entry.first == parameter) return entry.second;
            const auto bytes = matrix.size()*2*sizeof(double);
            while (!cache.empty() && (cache.size()+1)*bytes > budget) cache.pop_front();
            std::array<std::vector<double>, 2> *phase;
            if (bytes <= budget) {
                cache.emplace_back(parameter, std::array<std::vector<double>, 2>{});
                phase = &cache.back().second;
            } else phase = &oversized;
            for (auto &part : *phase) part.resize(matrix.size());
            for (py::ssize_t j = 0; j < angular; ++j)
                for (py::ssize_t d = 0; d < directions; ++d) {
                    double cosine = 0.0;
                    for (int axis = 0; axis < 3; ++axis) cosine += vectors[j*3+axis]*vectors[d*3+axis];
                    (*phase)[0][j*directions+d] = weights[j]*transport_phase(cosine,parameter,moments,false);
                    (*phase)[1][j*directions+d] = weights[j]*transport_phase_derivative(cosine,parameter,moments);
                }
            return *phase;
        };
        for (py::ssize_t p = 0; p < points; ++p) {
            const double incident = strided_value(ib,p), reflect = strided_value(rho,p);
            for (py::ssize_t seed = 0; seed < k; ++seed) {
                const double drho = strided_value(rb,p,seed);
                if (!std::isfinite(drho)) throw std::invalid_argument("reflectance tangents must be finite");
                double total = 0.0, dtotal = 0.0;
                for (py::ssize_t l = 0; l < layers; ++l) {
                    double s[5];
                    for (int role = 0; role < 5; ++role) {
                        s[role] = strided_value(sb[role],l,p,seed);
                        if (!std::isfinite(s[role])) throw std::invalid_argument("optical tangents must be finite");
                    }
                    const double a = strided_value(ob[3],l,p);
                    g[l] = strided_value(ob[4],l,p); dg[l] = s[4];
                    const double f = moments && g[l] > 0 ? std::pow(g[l],moments) : 0.0;
                    const double dfrac = moments && g[l] > 0 ? moments*std::pow(g[l],moments-1)*dg[l] : 0.0;
                    const double ext = strided_value(ob[0],l,p)+strided_value(ob[1],l,p)+strided_value(ob[2],l,p)-f*a;
                    const double dext = s[0]+s[1]+s[2]-f*s[3]-dfrac*a;
                    total += ext; dtotal += dext;
                    ray[l] = strided_value(ob[1],l,p)/subdivisions;
                    dray[l] = s[1]/subdivisions;
                    particle[l] = a*(1.0-f)/subdivisions;
                    dparticle[l] = (s[3]*(1.0-f)-a*dfrac)/subdivisions;
                    for (py::ssize_t sub = 0; max_order && sub < subdivisions; ++sub) {
                        const auto c = l*subdivisions+sub;
                        depth[c] = ext/subdivisions; dd[c] = dext/subdivisions;
                    }
                }
                if (!std::isfinite(dtotal)) throw std::invalid_argument("summed optical tangent overflow");
                const double gain = incident*mu0/pi, amplitude = gain*reflect;
                const double air = 1.0/mu0+1.0/muv;
                output[p*k+seed] = direct_reflected_value(gain,total,air)*drho -
                    direct_reflected_value(amplitude,total,air)*air*dtotal;
                if (!max_order) continue;
                prepare_kernel(depth.data(),mu.data(),t.data(),ht.data(),w.data(),hw.data(),sw.data(),shw.data(),
                    nullptr,true,mu0,cells,directions);
                prepare_transport_tangent(depth.data(),dd.data(),mu.data(),t.data(),ht.data(),dt.data(),dht.data(),
                    dw.data(),dhw.data(),dsw.data(),dshw.data(),mu0,cells,directions);
                const double atten = std::exp(-total/mu0), boundary0 = amplitude*atten;
                const double dboundary0 = gain*atten*drho-boundary0*dtotal/mu0;
                apply_transport_tangent(mu.data(),zero.data(),zero.data(),t.data(),dt.data(),w.data(),dw.data(),
                    ht.data(),dht.data(),hw.data(),dhw.data(),field.data(),df.data(),exits.data(),de.data(),
                    boundary0,dboundary0,cells,directions);
                // Unit surface response is reused at every order; no division by
                // transmission, reflectance or scattering depth is required.
                apply_transport_tangent(mu.data(),zero.data(),zero.data(),t.data(),dt.data(),w.data(),dw.data(),
                    ht.data(),dht.data(),hw.data(),dhw.data(),bc.data(),dbc.data(),bx.data(),dbx.data(),
                    1.0,0.0,cells,directions);
                bool reuse = true;
                for (py::ssize_t d = 0; d < directions; ++d)
                    if (mu[d] > 0 && bx[d] < std::numeric_limits<double>::min()) reuse = false;
                for (py::ssize_t l = 0; l < layers; ++l)
                    for (py::ssize_t d = 0; d < directions; ++d) {
                        const double pr = transport_phase(beam_cos[d],0,0,true);
                        const double pa = transport_phase(beam_cos[d],g[l],moments,false);
                        const double dpa = transport_phase_derivative(beam_cos[d],g[l],moments)*dg[l];
                        for (py::ssize_t sub = 0; sub < subdivisions; ++sub) {
                            const auto i = (l*subdivisions+sub)*directions+d;
                            beam[i] = incident/(4*pi)*(ray[l]*pr+particle[l]*pa);
                            dbeam[i] = incident/(4*pi)*(dray[l]*pr+dparticle[l]*pa+particle[l]*dpa);
                        }
                    }
                apply_transport_tangent(mu.data(),beam.data(),dbeam.data(),t.data(),dt.data(),sw.data(),dsw.data(),
                    ht.data(),dht.data(),shw.data(),dshw.data(),bf.data(),dbf.data(),be.data(),dbe.data(),
                    0.0,0.0,cells,directions);
                for (py::ssize_t order = 1; order <= max_order; ++order) {
                    for (py::ssize_t l = 0; l < layers; ++l) {
                        const auto &phase = phase_partials(g[l]);
                        for (std::size_t i = 0; i < matrix.size(); ++i) {
                            matrix[i] = ray[l]*rayleigh_matrix[i]+particle[l]*phase[0][i];
                            dmatrix[i] = dray[l]*rayleigh_matrix[i]+dparticle[l]*phase[0][i]+particle[l]*dg[l]*phase[1][i];
                        }
                        const auto offset = l*subdivisions*directions;
                        dense_redistribute(gemm,field.data()+offset,matrix.data(),source.data()+offset,subdivisions,directions);
                        dense_redistribute(gemm,df.data()+offset,matrix.data(),ds.data()+offset,subdivisions,directions);
                        dense_redistribute(gemm,field.data()+offset,dmatrix.data(),term.data(),subdivisions,directions);
                        for (py::ssize_t i = 0; i < subdivisions*directions; ++i) ds[offset+i] += term[i];
                    }
                    apply_transport_tangent(mu.data(),source.data(),ds.data(),t.data(),dt.data(),w.data(),dw.data(),
                        ht.data(),dht.data(),hw.data(),dhw.data(),field.data(),df.data(),exits.data(),de.data(),
                        0.0,0.0,cells,directions);
                    if (order == 1) {
                        for (py::ssize_t i = 0; i < count; ++i) { field[i] += bf[i]; df[i] += dbf[i]; }
                        for (py::ssize_t d = 0; d < directions; ++d) { exits[d] += be[d]; de[d] += dbe[d]; }
                    }
                    for (py::ssize_t j = 0; j < angular; ++j) {
                        flux[j] = mu[j] < 0 ? weights[j]*-mu[j]*exits[j] : 0.0;
                        dflux[j] = mu[j] < 0 ? weights[j]*-mu[j]*de[j] : 0.0;
                    }
                    const double sum = rayleigh_flux_sum(flux.data(),angular);
                    const double boundary = 4.0*reflect*sum;
                    const double dboundary = 4.0*(drho*sum+reflect*rayleigh_flux_sum(dflux.data(),angular));
                    if (reuse) {
                        for (py::ssize_t d = 0; d < directions; ++d) {
                            exits[d] += bx[d]*boundary;
                            de[d] += dbx[d]*boundary+bx[d]*dboundary;
                        }
                        for (py::ssize_t i = 0; i < count; ++i) {
                            field[i] += bc[i]*boundary;
                            df[i] += dbc[i]*boundary+bc[i]*dboundary;
                        }
                    } else {
                        // Preserve the primal recurrence's exponent range when
                        // the unit response underflows before amplitude scaling.
                        apply_transport_tangent(mu.data(),zero.data(),zero.data(),t.data(),dt.data(),w.data(),dw.data(),
                            ht.data(),dht.data(),hw.data(),dhw.data(),temp.data(),dtemp.data(),tx.data(),dtx.data(),
                            boundary,dboundary,cells,directions);
                        for (py::ssize_t d = 0; d < directions; ++d) { exits[d] += tx[d]; de[d] += dtx[d]; }
                        for (py::ssize_t i = 0; i < count; ++i) { field[i] += temp[i]; df[i] += dtemp[i]; }
                    }
                    output[(order*points+p)*k+seed] = de.back();
                }
            }
        }
        for (py::ssize_t i = 0; i < size*k; ++i)
            if (!std::isfinite(output[i])) throw std::invalid_argument("scattering tangent overflow");
    }
    return py::make_tuple(values,tangents);
}

static py::cpp_function make_scattering_orders_tangent(py::object owner, Dgemm gemm) {
    return py::cpp_function([owner = std::move(owner), gemm](
            py::tuple optical, py::tuple seeds, py::object irradiance,
            py::object reflectance, py::object reflectance_seed,
            py::object directions, py::object weights, py::ssize_t layers,
            py::ssize_t subdivisions, py::ssize_t max_order, int moments,
            double zenith, double view_zenith, double azimuth) {
        return scattering_orders_tangent(optical,seeds,irradiance,reflectance,reflectance_seed,
            directions,weights,layers,subdivisions,max_order,moments,zenith,view_zenith,azimuth,gemm);
    });
}
