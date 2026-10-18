"""
plot_utils: figures of StarSim simulations (time series of the filling factors, photometry, RV, FWHM, BIS, contrast)
with an optional drawing of the disc: the path of the planets and the active regions at one epoch.

Colormaps
---------
The colours come from a colormap, of matplotlib (default) or of cmasher (https://cmasher.readthedocs.io, optional:
pip install cmasher). Choose it once for all the figures:
    set_colormap('viridis')                     # matplotlib
    set_colormap('rainforest', cmasher=True)    # cmasher ('cmr.rainforest' works too)
or for one figure with cmap=... / cmasher=... in every function.

Functions (sims = one StarSim after generate_timeseries(), a list of them, or a dict {label: StarSim})
---------
    plot_photometry(sims, ...)          filling factors + light curve
    plot_rv(sims, ...)                  filling factors + RV
    plot_photometry_rv(sims, ...)       light curve + RV
    plot_activity(sims, ...)            light curve + RV + FWHM + BIS
    plot_rv_indicators(sims, ...)       RV + FWHM + BIS
    plot_timeseries(sims, quantities)   any panels among 'ff', 'flux', 'rv', 'fwhm', 'bis', 'contrast'
    plot_disc(ss, ...)                  the disc alone: path of the planets, regions at one epoch
With show_disc=True every time-series figure gets the disc on its left.

Common options: quiet_flux (flux of the quiet star, e.g. flux_grid_qp.total_brightness: the light curve is then
Delta F / F of the quiet star, otherwise relative to the median of each light curve), reference (what is subtracted
from rv / fwhm / bis: 'median', 'first' epoch, None or a number; the contrast is divided by it), remove_kepler (RV without
the Keplerian orbit of the planets), time_unit ('d' or 'h'), t0 (origin of the time axis), flux_unit ('%' or 'ppm').
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Circle

_STYLE = {'cmap': 'viridis', 'cmasher': False}

#panels: attribute of StarSim, label, mode needed
QUANTITIES = {
    'ff': (None, 'filling factor [%]', None),
    'flux': ('flux_var', None, 'photometry'),
    'rv': ('rv_var', 'RV [m/s]', 'ccf'),
    'fwhm': ('fwhm_var', 'FWHM [m/s]', 'ccf'),
    'bis': ('bis_var', 'BIS [m/s]', 'ccf'),
    'contrast': ('contrast_var', 'contrast', 'ccf'),
}


# ------------------------------------------------------------------------------------------- colours
def set_colormap(name='viridis', cmasher=False):
    """Colormap of all the figures: a matplotlib name, or a cmasher one with cmasher=True ('rainforest' or
    'cmr.rainforest')."""
    get_cmap(name, cmasher)                     # checks that it exists
    _STYLE['cmap'], _STYLE['cmasher'] = name, bool(cmasher)


def get_cmap(name=None, cmasher=None):
    """The colormap object: name / cmasher default to those of set_colormap."""
    name = _STYLE['cmap'] if name is None else name
    cmasher = _STYLE['cmasher'] if cmasher is None else cmasher
    if not isinstance(name, str):
        return name                             # already a colormap
    if cmasher:
        try:
            import cmasher as cmr               # noqa: F401  (registers the 'cmr.*' colormaps in matplotlib)
        except ImportError:
            raise ImportError('cmasher is not installed (pip install cmasher); use cmasher=False for the matplotlib '
                              'colormaps') from None
        if not name.startswith('cmr.'):
            name = 'cmr.' + name
    try:
        return colormaps[name]
    except KeyError:
        raise ValueError('unknown colormap %r%s' % (name, ' (cmasher)' if cmasher else '')) from None


def colours(n, cmap=None, cmasher=None, lo=0.1, hi=0.85):
    """n colours evenly spaced in the colormap (avoiding its extreme ends)."""
    cm = get_cmap(cmap, cmasher)
    return [cm(x) for x in (np.linspace(lo, hi, n) if n > 1 else [0.5 * (lo + hi)])]


# ------------------------------------------------------------------------------------------- inputs
def _as_dict(sims):
    """{label: StarSim} from one StarSim, a list or a dict."""
    if isinstance(sims, dict):
        out = dict(sims)
    elif isinstance(sims, (list, tuple)):
        out = {'simulation %d' % (i + 1): s for i, s in enumerate(sims)}
    else:
        out = {'': sims}
    if not out:
        raise ValueError('no simulation to plot')
    for label, s in out.items():
        if not hasattr(s, 'obs_times') or not hasattr(s, 'ff_sp'):
            raise TypeError('%s is not a StarSim after generate_timeseries() (got %s)'
                            % (label or 'the simulation', type(s).__name__))
    return out


def _series(ss, q, label):
    """Values of the quantity q of a simulation, with a clear error if its mode was not computed."""
    attr, _, mode = QUANTITIES[q]
    val = getattr(ss, attr, None)
    if val is None:
        raise ValueError("%s has no %s: run StarSim with mode=[..., '%s'] (and generate_timeseries())"
                         % (label or 'the simulation', attr, mode))
    return np.asarray(val, dtype=np.float64)


def _relative(y, reference, ratio=False):
    if reference is None:
        return y
    if reference == 'median':
        ref = np.median(y)
    elif reference == 'first':
        ref = y[0]
    elif np.ndim(reference) == 0 and not isinstance(reference, str):
        ref = float(reference)
    else:
        raise ValueError("reference must be 'median', 'first', None or a number, got %r" % (reference,))
    return y / ref if ratio else y - ref


def _ylabel(q, reference, quiet_flux, flux_unit):
    if q == 'flux':
        return '$\\Delta F / F$ [%s]%s' % (flux_unit, '' if quiet_flux is not None else ' (median)')
    label = QUANTITIES[q][1]
    if q == 'contrast':
        return label if reference is None else 'contrast / %s' % ('first epoch' if reference == 'first' else reference)
    if reference is None:
        return label
    return label.replace(' [', ' - %s [' % ('first epoch' if reference == 'first' else reference))


def _values(ss, q, label, quiet_flux, flux_unit, reference, remove_kepler):
    scale = 1e6 if flux_unit == 'ppm' else 100.0
    if q == 'flux':
        f = _series(ss, 'flux', label)
        return scale * (f / (quiet_flux if quiet_flux is not None else np.median(f)) - 1)
    y = _series(ss, q, label)
    if q == 'rv' and remove_kepler and getattr(ss, 'rv_kepler', None) is not None:
        y = y - np.asarray(ss.rv_kepler, dtype=np.float64)
    return _relative(y, reference, ratio=(q == 'contrast'))


# ------------------------------------------------------------------------------------------- disc
def _region_outline(vec, radius, n=180):
    """(y, z) of the visible part of the edge of a circular region (centre vec = (x, y, z) on the unit sphere, x towards
    the observer; angular radius [rad]), NaN where it is behind the star."""
    v = np.asarray(vec, dtype=float) / np.linalg.norm(vec)
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    phi = np.linspace(0, 2 * np.pi, n)
    pts = np.cos(radius) * v[None, :] + np.sin(radius) * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    y, z = pts[:, 1].copy(), pts[:, 2].copy()
    behind = pts[:, 0] < 0
    y[behind], z[behind] = np.nan, np.nan
    return y, z


def _default_epoch(ss):
    """Mid-transit of the first planet (closest to the disc centre while in front of the star) if it transits during
    the run, else the middle of the run."""
    pos = getattr(ss, 'planet_pos', None)
    if pos is not None and np.size(pos) and getattr(ss, 'simulate_planet', True):
        rho = np.array(pos[:, 0, 0], dtype=float)
        vis = getattr(ss, 'visible', None)
        if vis is not None:
            n_reg = len(getattr(ss, 'active_regions_map', None) or [])
            rho[~vis[:, n_reg].astype(bool)] = np.inf          # behind the star
        if np.any(rho < 1 + pos[:, 0, 2]):
            return int(np.argmin(rho))
    return len(ss.obs_times) // 2


def plot_disc(ss, epoch=None, ax=None, show_planets=True, show_regions=True, show_axis=True, cmap=None, cmasher=None,
              star_colour=None, legend=True):
    """The visible disc in the plane of the sky (horizontal: projected equator, vertical: projected rotation axis), with
    the path of every planet on the disc (arrow: direction of motion) and, at one epoch (index of obs_times, or None: the
    middle of the run, or the mid-transit of the first planet if there is one), the planets and the circular regions
    (spots: full line, faculae: dashed; their part on the far side is not drawn)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = get_cmap(cmap, cmasher)
    ax.add_patch(Circle((0, 0), 1, color=star_colour or cm(0.95), alpha=0.25, lw=0))
    ax.add_patch(Circle((0, 0), 1, fill=False, color='k', lw=0.8))
    if show_axis:
        ax.axvline(0, color='grey', lw=0.6, ls=':')
        ax.axhline(0, color='grey', lw=0.6, ls=':')
    times = np.asarray(ss.obs_times, dtype=float)
    pos = getattr(ss, 'planet_pos', None)
    has_planets = show_planets and pos is not None and np.size(pos) > 0 and getattr(ss, 'simulate_planet', True)
    n_reg = len(getattr(ss, 'active_regions_map', None) or [])
    epoch = _default_epoch(ss) if epoch is None else int(epoch)
    if not -len(times) <= epoch < len(times):
        raise IndexError('epoch %d: there are %d epochs' % (epoch, len(times)))

    if show_regions and n_reg and getattr(ss, 'regions_vec', None) is not None and ss.regions_vec.shape[1] == n_reg:
        c_sp, c_fc = colours(2, cmap, cmasher, 0.05, 0.6)
        for i, region in enumerate(ss.active_regions_map):
            r = ss.regions_pos[epoch, i, 2]
            if not ss.visible[epoch, i] or r <= 0:
                continue
            y, z = _region_outline(ss.regions_vec[epoch, i], r)
            sp = region.type == 'sp'
            ax.plot(y, z, '-' if sp else '--', color=c_sp if sp else c_fc, lw=1.2)
        ax.plot([], [], '-', color=c_sp, label='spots')
        ax.plot([], [], '--', color=c_fc, label='faculae')

    if has_planets:
        n_pl = pos.shape[1]
        vis = getattr(ss, 'visible', None)
        for p, c in zip(range(n_pl), colours(n_pl, cmap, cmasher, 0.3, 0.9)):
            rho, theta, rp = pos[:, p, 0], pos[:, p, 1], pos[:, p, 2]
            y, z = rho * np.cos(theta), rho * np.sin(theta)
            front = np.ones(len(times), bool) if vis is None else vis[:, n_reg + p].astype(bool)
            on = front & (rho < 1 + rp)
            if not np.any(on):
                continue
            yy, zz = np.where(on, y, np.nan), np.where(on, z, np.nan)
            ax.plot(yy, zz, '-', color=c, lw=1.0, label='planet %d' % (p + 1) if n_pl > 1 else 'planet')
            idx = np.flatnonzero(on)
            if len(idx) > 2:                                        # direction of motion
                k1, k0 = idx[-1], idx[max(0, len(idx) - 1 - max(1, len(idx) // 10))]
                ax.annotate('', xy=(y[k1], z[k1]), xytext=(y[k0], z[k0]),
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.2))
            if on[epoch]:
                ax.add_patch(Circle((y[epoch], z[epoch]), rp[epoch], color=c, alpha=0.8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('disc at t = %.3g d' % times[epoch], fontsize=9)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc='lower left')
    return ax


# ------------------------------------------------------------------------------------------- time series
def plot_timeseries(sims, quantities, quiet_flux=None, reference='median', remove_kepler=False, show_disc=False,
                    disc_epoch=None, time_unit='d', t0=0.0, flux_unit='%', cmap=None, cmasher=None, marker='-',
                    figsize=None, title=None, mark_epoch=True):
    """Stacked panels (shared time axis) of the quantities, among 'ff' (spot, facula and planet filling factors),
    'flux', 'rv', 'fwhm', 'bis', 'contrast', for one or several simulations. show_disc=True draws the disc on the left
    (plot_disc of the first simulation at disc_epoch, marked by a vertical line on the time series if mark_epoch).
    quiet_flux: a number, or a dict {label: number} for several simulations. Returns fig, axes (dict {quantity: ax},
    plus 'disc')."""
    runs = _as_dict(sims)
    quantities = [quantities] if isinstance(quantities, str) else list(quantities)
    for q in quantities:
        if q not in QUANTITIES:
            raise ValueError('unknown quantity %r: choose among %s' % (q, list(QUANTITIES)))
    if time_unit not in ('d', 'h'):
        raise ValueError("time_unit must be 'd' or 'h', got %r" % (time_unit,))
    if flux_unit not in ('%', 'ppm'):
        raise ValueError("flux_unit must be '%%' or 'ppm', got %r" % (flux_unit,))
    n = len(quantities)
    figsize = figsize or ((13 if show_disc else 9), 1.0 + 2.0 * n)
    fig = plt.figure(figsize=figsize)
    if show_disc:
        gs = fig.add_gridspec(n, 2, width_ratios=[1, 2.6])
        ax_disc = fig.add_subplot(gs[:, 0])
        col = 1
    else:
        gs = fig.add_gridspec(n, 1)
        col = 0
    axes = {}
    first = None
    for i, q in enumerate(quantities):
        axes[q] = fig.add_subplot(gs[i, col], sharex=first)
        first = first or axes[q]
    scale_t = 24.0 if time_unit == 'h' else 1.0
    run_colours = colours(len(runs), cmap, cmasher)
    kinds = (('ff_sp', 'spots', '-'), ('ff_fc', 'faculae', '--'), ('ff_pl', 'planets', ':'))
    kind_colours = colours(3, cmap, cmasher, 0.05, 0.65)
    for (label, ss), c in zip(runs.items(), run_colours):
        t = (np.asarray(ss.obs_times, dtype=float) - t0) * scale_t
        qf = quiet_flux.get(label) if isinstance(quiet_flux, dict) else quiet_flux
        for q in quantities:
            ax = axes[q]
            if q == 'ff':
                for (attr, name, ls), kc in zip(kinds, kind_colours):
                    y = np.asarray(getattr(ss, attr), dtype=float)
                    if not np.any(y):
                        continue
                    lab = name if len(runs) == 1 else '%s, %s' % (label, name)
                    ax.plot(t, y, ls, color=kc if len(runs) == 1 else c, lw=1.3, label=lab)
                continue
            y = _values(ss, q, label, qf, flux_unit, reference, remove_kepler)
            ax.plot(t, y, marker, color=c, lw=1.3, ms=3, label=label or None)
    for q in quantities:
        ax = axes[q]
        ax.set_ylabel(_ylabel(q, reference, quiet_flux, flux_unit) if q != 'ff' else QUANTITIES['ff'][1], fontsize=9)
        ax.tick_params(labelsize=8)
        if q != quantities[-1]:
            ax.tick_params(labelbottom=False)
        if q == 'ff' and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=3)
    others = [q for q in quantities if q != 'ff']
    if len(runs) > 1 and others:
        axes[others[0]].legend(fontsize=7)
    axes[quantities[-1]].set_xlabel('time%s [%s]' % (' - %g d' % t0 if t0 else '', time_unit))
    if show_disc:
        ss0 = next(iter(runs.values()))
        plot_disc(ss0, epoch=disc_epoch, ax=ax_disc, cmap=cmap, cmasher=cmasher)
        axes['disc'] = ax_disc
        if mark_epoch:
            k = _default_epoch(ss0) if disc_epoch is None else int(disc_epoch)
            te = (ss0.obs_times[k] - t0) * scale_t
            for q in quantities:
                axes[q].axvline(te, color='grey', lw=0.7, ls=':')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_photometry(sims, show_disc=False, **kw):
    """Filling factors and light curve (show_disc=True: the disc with the path of the planets)."""
    return plot_timeseries(sims, ['ff', 'flux'], show_disc=show_disc, **kw)


def plot_rv(sims, show_disc=False, **kw):
    """Filling factors and RV (show_disc=True: the disc with the path of the planets)."""
    return plot_timeseries(sims, ['ff', 'rv'], show_disc=show_disc, **kw)


def plot_photometry_rv(sims, **kw):
    """Light curve and RV (needs mode=['photometry', 'ccf'])."""
    return plot_timeseries(sims, ['flux', 'rv'], **kw)


def plot_activity(sims, **kw):
    """Light curve, RV, FWHM and BIS (needs mode=['photometry', 'ccf'])."""
    return plot_timeseries(sims, ['flux', 'rv', 'fwhm', 'bis'], **kw)


def plot_rv_indicators(sims, **kw):
    """RV, FWHM and BIS (mode 'ccf')."""
    return plot_timeseries(sims, ['rv', 'fwhm', 'bis'], **kw)
