#!/usr/bin/env python
"""
Dumusque bisector of the spot CCF scaled by a function of mu so that the RV of the MURaM spot relative to the MURaM
quiet photosphere, ring by ring, is the dv_conv of the spots of Palumbo, Saar & Haywood (2024, ApJ 973, 11;
arXiv:2404.16747, Appendix B, Table 2: SDO/HMI 2014-2018, umbra I < 0.45 <I_quiet>, penumbra 0.45 - 0.89, each
relative to the quiet Sun at the same mu).

Same setup as spot_dumusque_mu.ipynb: MURaM spectra (quiet SSD 200 G; spot: vx-drift corrected SSD x the black body
ratio for T_quiet - dT), HARPS G2 mask, N_RINGS rings, the RV of a ring is the centre of the gaussian fitted to its CCF.
Adding the Dumusque bisector times f to the velocity axis of the spot CCF shifts its RV by f D(mu), so for every ring

    dv_target(mu) = [RV_spot(mu) - RV_quiet(mu)]  +  f(mu) D(mu)
    f(mu) = (dv_target(mu) - [RV_spot - RV_quiet](mu)) / D(mu)

with D(mu) = RV(spot + Dumusque x 1) - RV(spot). f is computed at the rings with 0.15 <= mu <= 0.95 (the bins of
Palumbo et al., interpolated) and fitted with polynomials in mu, g(mu) = c0 + c1 mu + ...: the coefficients
(increasing powers) are dumusque_bisector(mu_scaling=(c0, c1, ...)), or dumusque_mu_scaling = c0, c1, ... in a
configuration. Below mu = 0.15 g is extrapolated.
The spot of Palumbo et al. is umbra and penumbra together, weighted by w_umbra (the fraction of the light of a spot
in its umbra) and 1 - w_umbra. Then the spot grid is built again with g and its RV relative to the quiet one is
compared with the target.

    python spot_dumusque_palumbo.py [--dT 300 600 1100] [--w-umbra 0.1] [--out results/spot_dumusque_palumbo]
"""
import argparse
import copy
import json
import os

os.environ.setdefault('KMP_WARNINGS', '0')
import numpy as np

T_QUIET = 5770                          # [K]
N_RINGS = 30
RV_GRID = np.arange(-18000, 18025, 25)  # [m/s]
BAND_CCF = (3800, 6800)                 # [A] mask lines
# Palumbo et al. (2024), Appendix B, Table 2: dv_conv = v - v_quiet(mu) [m/s], mu bins of 0.1
PALUMBO_MU = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
PALUMBO = {'umbra':    np.array([-161.60, -65.30, 44.70, 110.51, 159.09, 189.38, 215.56, 228.84, 225.24]),
           'penumbra': np.array([-216.29, -128.93, -31.76, 47.32, 115.11, 174.19, 224.75, 255.76, 256.74]),
           'plage':    np.array([-82.47, -56.14, -16.70, 19.24, 49.58, 80.78, 111.50, 145.31, 160.66]),
           'network':  np.array([-53.76, -22.74, 9.67, 37.01, 59.81, 82.34, 105.17, 124.36, 129.50])}


def load_spectra():
    from sunsim import local_path
    spectra = {}
    for key, folder, name in (('quiet', 'muram_spectra', 'av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid.npy'),
                              ('quiet_low', 'muram_lowres_spectra',
                               'av_solar_ssd_200G_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy'),
                              ('spot', 'muram_spot_spectra', 'av_solar_ssd_10_mu_angles_angles_PHOENIX_high_res_grid.npy'),
                              ('spot_low', 'muram_spot_spectra', 'av_solar_ssd_10_mu_angles_PHOENIX_low_res_specint_grid.npy')):
        spectra[key] = np.load(str(local_path(folder) / name), allow_pickle=True).item()
    for mu in spectra['spot_low']:                       # as in spot_dumusque_mu.ipynb
        conv = (1e-8 * spectra['spot_low'][mu]['wav'] ** 2 * 2.99792458e10) / 1e12
        spectra['spot_low'][mu]['intensity'] /= conv
        spectra['quiet_low'][mu]['intensity'] /= conv
    return spectra


def cooler(flux, dT):
    from sunsim import black_body
    out = copy.deepcopy(flux)
    for d in out.values():
        d['intensity'] *= black_body(d['wav'], T_QUIET - dT) / black_body(d['wav'], T_QUIET)
    return out


def ring_brightness(flux_low, typ):
    from sunsim import flux_grid
    g = flux_grid(N_RINGS, flux_low, typ, None, *BAND_CCF, rotation_period=25.0, inclination=0, stellar_radius=1,
                  stellar_rotation=False, mode='photometry')
    g.built_grid()
    return g.amu, g.grid / (g.parea / (4 * np.pi))


def ccf_grid(flux, brightness, b_ref, wvm, fm, convective_blueshift):
    from sunsim import CCF_grid
    g = CCF_grid(N_RINGS, flux, RV_GRID, wvm, fm, convective_blueshift=convective_blueshift, phoenix_resolution=True,
                 mu_ratio=brightness / b_ref, normalization=False)
    g.compute_ccf()
    return g


def ring_rv(g):
    """Centre of the gaussian fitted to the CCF of every ring [m/s] (ring_rv_fit of spot_dumusque_mu.ipynb)."""
    from sunsim import nbspectra
    out = []
    for rv_i, c in zip(g.rvs_ring, g.ccf_shape):
        c = c - c.min() + 1e-6
        cl, cr, _, _ = nbspectra.speed_bisector_nb(rv_i, c / c.max(), True)
        x, y = rv_i[int(cl):int(cr)], c[int(cl):int(cr)]
        k = y.argmax()
        p, ok = nbspectra.fit_gaussian_nb(x, y, np.array([y[k], x[k], 3000., 1e-6]), 500)
        out.append(p[1] if ok else np.nan)
    return np.array(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dT', type=float, nargs='+', default=[300, 600, 1100], help='T_quiet - T_spot [K]')
    p.add_argument('--w-umbra', type=float, default=0.1, help='fraction of the light of the spot in its umbra (0.1)')
    p.add_argument('--degrees', type=int, nargs='+', default=[1, 2, 3], help='polynomials g(mu) fitted')
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                 'spot_dumusque_palumbo'))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sunsim import local_path, mask_weights, dumusque_bisector

    target_bins = a.w_umbra * PALUMBO['umbra'] + (1 - a.w_umbra) * PALUMBO['penumbra']
    spectra = load_spectra()
    wvm, fm = mask_weights(str(local_path('ccf_masks') / 'HARPS_G20_mps_friendly.mas'))
    amu, b_quiet = ring_brightness(spectra['quiet_low'], 'quiet')
    b_ref = b_quiet[0]
    quiet = ccf_grid(spectra['quiet'], b_quiet, b_ref, wvm, fm, 0)
    quiet.rv_treatment(bisector=None)
    rv_quiet = ring_rv(quiet)
    fit_rings = (amu >= PALUMBO_MU[0]) & (amu <= PALUMBO_MU[-1])
    target = np.interp(amu, PALUMBO_MU, target_bins)
    mu_fine = np.linspace(0.05, 1, 200)

    results = {'w_umbra': a.w_umbra, 'mu_rings': amu.tolist(), 'target': target.tolist(), 'dT': {}}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for j, dT in enumerate(a.dT):
        spot = ccf_grid(cooler(spectra['spot'], dT), ring_brightness(cooler(spectra['spot_low'], dT), 'sp')[1], b_ref,
                        wvm, fm, 1)
        spot.rv_treatment(bisector=None)
        base = ring_rv(spot) - rv_quiet                                  # spot - quiet without any bisector
        spot.rv_treatment(bisector=dumusque_bisector(normalization=False))
        D = ring_rv(spot) - rv_quiet - base                              # shift of the Dumusque bisector x 1
        f = (target - base) / D
        polys = {}
        for deg in a.degrees:
            c = np.polyfit(amu[fit_rings], f[fit_rings], deg)
            polys[deg] = np.poly1d(c)
        # check: the spot grid with the fitted g, ring by ring
        check = {}
        for deg, g in polys.items():
            spot.rv_treatment(bisector=dumusque_bisector(normalization=False, mu_scaling=g))
            check[deg] = ring_rv(spot) - rv_quiet
        r = {'base': base.tolist(), 'D': D.tolist(), 'f': f.tolist(),
             'dumusque_mu_scaling': {deg: g.coeffs[::-1].tolist() for deg, g in polys.items()},
             'rms': {deg: float(np.std((check[deg] - target)[fit_rings])) for deg in polys}}
        results['dT'][str(int(dT))] = r
        print('dT = %d K: base (spot - quiet, no bisector) %.0f to %.0f m/s, D %.0f to %.0f m/s'
              % (dT, base[fit_rings].min(), base[fit_rings].max(), D[fit_rings].min(), D[fit_rings].max()))
        for deg, g in polys.items():
            print('   degree %d: dumusque_mu_scaling = %s (c0, c1, ...: c0 + c1 mu + ...), rms vs target %.1f m/s'
                  % (deg, ', '.join('%.4f' % c for c in g.coeffs[::-1]), r['rms'][deg]))
        colour = 'C%d' % j
        ax[0].plot(amu, base, '--', color=colour, label='spot - quiet, no bisector, dT = %d K' % dT)
        ax[1].plot(amu, f, 'o', ms=4, color=colour, label='f(mu) needed, dT = %d K' % dT)
        best = max(polys)
        ax[1].plot(mu_fine, polys[best](mu_fine), '-', color=colour, label='degree %d fit' % best)
        ax[2].plot(amu, check[best], 'o-', ms=3, color=colour, label='SunSim, g of degree %d, dT = %d K' % (best, dT))

    for q, ls in (('umbra', ':'), ('penumbra', '-.')):
        ax[2].plot(PALUMBO_MU, PALUMBO[q], ls, color='0.4', label='Palumbo+24 %s' % q)
    ax[2].plot(PALUMBO_MU, target_bins, 'ks', ms=6, mfc='none', label='target: %.2f umbra + %.2f penumbra'
               % (a.w_umbra, 1 - a.w_umbra))
    ax[2].plot(PALUMBO_MU, PALUMBO['plage'], '-', color='C2', alpha=0.5, label='Palumbo+24 plage')
    ax[0].plot(PALUMBO_MU, target_bins, 'ks', ms=6, mfc='none', label='target (Palumbo+24 spot)')
    for k, label in enumerate(('RV - RV_quiet [m/s]', 'Dumusque factor f(mu)', 'RV_spot - RV_quiet [m/s]')):
        ax[k].set_xlabel('mu')
        ax[k].set_ylabel(label)
        ax[k].axhline(0, color='0.85', lw=0.8)
        ax[k].axvspan(0, PALUMBO_MU[0], color='0.95', zorder=0)
        ax[k].legend(fontsize=7)
    ax[0].set_title('MURaM spot without bisector')
    ax[1].set_title('factor of the Dumusque bisector')
    ax[2].set_title('spot with the fitted g(mu)')
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, 'spot_dumusque_palumbo.png'), dpi=130)
    with open(os.path.join(a.out, 'spot_dumusque_palumbo.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    print('written: %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
