"""
Periodic signals in a time series and detection limits of planets: for the residuals of the comparison with the
observations (observations.compare_rv: obs - sim) or for the observations themselves.
  periodogram   : generalised Lomb-Scargle periodogram (GLS, Zechmeister & Kuerster 2009, normalisation ZK), the best
                  sinusoid and its false-alarm probability (FAP), as gls.py of M. Zechmeister (same frequency grid,
                  same FAP)
  prewhitening  : the strongest sinusoid subtracted, one after the other, while its FAP is below a threshold
  wavelet       : Morlet wavelet power spectrum and its 95 % significance (needs pycwt: pip install pycwt)
  injection     : Keplerian signals (planet.keplerian_orbit) injected on a grid of periods, semi-amplitudes and times of
                  transit, and the fraction recovered by the GLS (FAP, period)

Typical use (see analysis/10_years_injection.ipynb):

    from sunsim import observations as obs
    from sunsim.residuals import ResidualAnalysis
    rv = obs.compare_rv(harps, t_sim, rv_sim, interventions=cuts)
    ra = ResidualAnalysis.from_comparison(rv)                 # the residuals; series='obs': the HARPS-N RVs
    ra.plot_periodogram()
    ra.inject(periods=np.linspace(5, 500, 10), amplitudes=np.linspace(0.1, 0.5, 21), n_phases=100, fap=1e-3)
    ra.plot_detection_map(ra.detection_rate(period_tol=0.1))  # % of the injections recovered at their period

The times are in days; the weights of the GLS are 1/err^2 (all equal without errors).
"""
import warnings

import numpy as np

from .planet import planet

__all__ = ['ResidualAnalysis', 'sinusoid']


def sinusoid(t, P, K, phi, offset):
    """offset + K sin(2 pi t / P + phi)."""
    return offset + K * np.sin(2 * np.pi * t / P + phi)


class ResidualAnalysis:
    """GLS periodogram, prewhitening, wavelet and injection-recovery of planets for one time series (t [days], y).

    err             : errors of y (weights 1/err^2 of the GLS); None: equal weights
    fbeg, fend      : frequency range of the GLS [1/day]; default 1/(ofac T) and hifac times the mean Nyquist frequency
                      N/(2T) (T: time span)
    ofac, hifac     : oversampling of the frequency grid (step 1/(ofac T)) and maximum frequency in units of N/(2T)
    cache           : maximum number of values of the cos and sin of the grid kept in memory (2 x frequencies x times,
                      8 bytes each); above it they are recomputed at every periodogram
    After inject(): self.injection (see inject)."""

    def __init__(self, t, y, err=None, name='', fbeg=None, fend=None, ofac=10, hifac=1, cache=5e7):
        t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ok = np.isfinite(t) & np.isfinite(y)
        if err is not None:
            err = np.asarray(err, dtype=np.float64)
            ok &= np.isfinite(err) & (err > 0)
        order = np.argsort(t[ok])
        self.t, self.y = t[ok][order], y[ok][order]
        self.err = None if err is None else err[ok][order]
        self.name = name
        self.N = len(self.t)
        if self.N < 4:
            raise ValueError('at least 4 points are needed, got %d' % self.N)
        self.th = self.t - self.t[0]
        self.tbase = self.th[-1]

        fstep = 1 / self.tbase / ofac
        self.fbeg = fstep if fbeg is None else fbeg
        self.fend = 0.5 * self.N / self.tbase * hifac if fend is None else fend
        if self.fend <= self.fbeg:
            raise ValueError('fend (%g) must be larger than fbeg (%g)' % (self.fend, self.fbeg))
        self.freq = np.arange(self.fbeg, self.fend, fstep)
        self.M = (self.fend - self.fbeg) * self.tbase          # independent frequencies (eq. 24 of ZK09)

        w = np.ones(self.N) if self.err is None else 1 / self.err ** 2
        self.w = w / w.sum()
        self._cache = None
        self._block = max(1, int(2e6 // self.N))                # frequencies per block of cos and sin
        if 2 * len(self.freq) * self.N <= cache:
            self._cache = list(self._trig())
        C, S, CC, CS = (np.empty(len(self.freq)) for _ in range(4))
        for sl, cos, sin in self._trig():
            C[sl], S[sl] = cos @ self.w, sin @ self.w
            CC[sl], CS[sl] = (cos * cos) @ self.w, (cos * sin) @ self.w
        SS = 1 - CC - S * S
        CC, CS = CC - C * C, CS - C * S
        self._C, self._S, self._CC, self._SS, self._CS = C, S, CC, SS, CS
        self._D = CC * SS - CS * CS
        self.injection = None

    @classmethod
    def from_comparison(cls, rv, series='res', errors=False, period_medians=False, **kw):
        """From the dict of observations.compare_rv: series 'res' (obs - sim), 'obs' or 'sim'; errors: weights from
        rv['obs_err']; period_medians: the median of every instrumental period (between rv['interventions']) is
        removed from the series (for 'obs': the offsets of the instrument, without the simulation)."""
        names = {'res': 'residuals', 'obs': 'observations', 'sim': 'simulation'}
        kw.setdefault('name', names.get(series, series))
        y = np.array(rv[series], dtype=np.float64)
        if period_medians:
            p = np.searchsorted(np.sort(rv['interventions']), rv['t'], side='right')
            for k in np.unique(p):
                y[p == k] -= np.nanmedian(y[p == k])
        return cls(rv['t'], y, err=rv['obs_err'] if errors else None, **kw)

    def save(self, path):
        """Saves the series, the frequency range and the injection (if any) to a .npz file (load)."""
        out = {'t': self.t, 'y': self.y, 'name': self.name, 'fbeg': self.fbeg, 'fend': self.fend,
               'fstep': self.freq[1] - self.freq[0]}
        if self.err is not None:
            out['err'] = self.err
        if self.injection is not None:
            out.update({'inj_' + k: v for k, v in self.injection.items()})
        np.savez(path, **out)

    @classmethod
    def load(cls, path, **kw):
        """ResidualAnalysis of a file of save, with its injection."""
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        ofac = 1 / (d['fstep'] * (d['t'][-1] - d['t'][0]))
        ra = cls(d['t'], d['y'], err=d.get('err'), name=str(d['name']), fbeg=float(d['fbeg']), fend=float(d['fend']),
                 ofac=ofac, **kw)
        inj = {k[4:]: d[k] for k in d if k.startswith('inj_')}
        if inj:
            inj['fap_fit'] = float(inj['fap_fit'])
            ra.injection = inj
        return ra

    # ------------------------------------------------------------------------------------------------------------------
    # periodogram
    # ------------------------------------------------------------------------------------------------------------------

    def _trig(self):
        """(slice of the frequencies, cos, sin) of 2 pi f (t - t0), block by block."""
        if self._cache is not None:
            yield from self._cache
            return
        for i in range(0, len(self.freq), self._block):
            sl = slice(i, i + self._block)
            x = 2 * np.pi * self.freq[sl, None] * self.th[None, :]
            yield sl, np.cos(x), np.sin(x)

    def _gls(self, y):
        """GLS of y (N,) or of every column of y (N, m): power (ZK), a, b (y = off + a cos + b sin) and the offset."""
        y = np.asarray(y, dtype=np.float64)
        one = y.ndim == 1
        y = y[:, None] if one else y
        Y = self.w @ y
        wy = y - Y
        YY = self.w @ wy ** 2
        wy = wy * self.w[:, None]
        YC, YS = np.empty((len(self.freq), y.shape[1])), np.empty((len(self.freq), y.shape[1]))
        for sl, cos, sin in self._trig():
            YC[sl], YS[sl] = cos @ wy, sin @ wy
        CC, SS, CS, D = (v[:, None] for v in (self._CC, self._SS, self._CS, self._D))
        a = (YC * SS - YS * CS) / D
        b = (YS * CC - YC * CS) / D
        power = (SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS) / (YY * D)
        off = Y - a * self._C[:, None] - b * self._S[:, None]
        out = (power, a, b, off)
        return tuple(v[:, 0] for v in out) if one else out

    def fap(self, power):
        """False-alarm probability of a power (ZK): 1 - (1 - prob)^M, prob = (1 - p)^((N-3)/2)."""
        prob = (1 - np.asarray(power, dtype=np.float64)) ** ((self.N - 3) / 2)
        Mprob = self.M * prob
        out = np.where(Mprob > 0.01, 1 - (1 - prob) ** self.M, Mprob)
        return out if out.ndim else float(out)

    def power_level(self, fap):
        """Power of the false-alarm probability fap (inverse of self.fap)."""
        prob = 1 - (1 - np.asarray(fap, dtype=np.float64)) ** (1 / self.M)
        return 1 - prob ** (2 / (self.N - 3))

    def periodogram(self, y=None):
        """GLS of y (default self.y; (N,) or (N, m), one periodogram per column). Returns a dict: 'freq', 'power'
        and the highest peak: 'f', 'P', 'amp', 'phi', 'offset' (model offset + amp sin(2 pi f (t - t[0]) + phi)),
        'T0' (time of a maximum minus P/4: amp sin(2 pi f (t - T0))), 'pmax', 'fap'."""
        y = self.y if y is None else y
        power, a, b, off = self._gls(y)
        k = np.argmax(power, axis=0)
        pick = (lambda v: v[k]) if power.ndim == 1 else (lambda v: v[k, np.arange(v.shape[1])])
        f, A, B = self.freq[k], pick(a), pick(b)
        phi = np.arctan2(A, B)
        pmax = pick(power)
        return {'freq': self.freq, 'power': power, 'f': f, 'P': 1 / f, 'amp': np.hypot(A, B), 'phi': phi,
                'offset': pick(off), 'T0': self.t[0] - phi / (2 * np.pi * f), 'pmax': pmax, 'fap': self.fap(pmax)}

    def model(self, best, t=None):
        """The sinusoid of periodogram (best: its dict, one series) at the times t (default self.t)."""
        t = self.t if t is None else np.asarray(t, dtype=np.float64)
        return sinusoid(t - self.t[0], best['P'], best['amp'], best['phi'], best['offset'])

    def plot_periodogram(self, ax=None, y=None, fap_levels=(1e-3, 1e-4, 5e-5), period=True, mark=(), label=None,
                         color='C0'):
        """GLS power against the period [days] (or the frequency, period=False) with the power of the false-alarm
        probabilities fap_levels (dashed) and vertical lines at the periods mark. Returns ax."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        pg = self.periodogram(y)
        x = 1 / pg['freq'] if period else pg['freq']
        ax.plot(x, pg['power'], color=color, lw=1,
                label=(label or self.name or None) and '%s  (P = %.2f d, FAP = %.1e)' % (label or self.name, pg['P'],
                                                                                       pg['fap']))
        for i, level in enumerate(fap_levels):
            ax.axhline(self.power_level(level), color='k', alpha=0.5, lw=0.8, ls=(0, (2 + 2 * i, 2)),
                       label='FAP = %g %%' % (100 * level))
        for p in mark:
            ax.axvline(p if period else 1 / p, color='#76a4f4', alpha=0.7)
        if period:
            ax.set_xscale('log')
        ax.set_xlabel('Period [days]' if period else 'Frequency [1/day]')
        ax.set_ylabel('Power (ZK)')
        ax.legend()
        return ax

    # ------------------------------------------------------------------------------------------------------------------
    # prewhitening
    # ------------------------------------------------------------------------------------------------------------------

    def prewhiten(self, n_max=10, fap=0.1, y=None):
        """Subtracts the best sinusoid of the GLS, n_max times at most, while its FAP is <= fap. Returns a dict:
        'signals' (the dicts of periodogram of the sinusoids subtracted, without 'freq', 'power'), 'series' (the
        series before every subtraction and the final one) and 'residual' (the final one)."""
        y = np.array(self.y if y is None else y, dtype=np.float64)
        signals, series = [], [y.copy()]
        for _ in range(n_max):
            best = self.periodogram(y)
            if best['fap'] > fap:
                break
            y = y - self.model(best)
            signals.append({k: v for k, v in best.items() if k not in ('freq', 'power')})
            series.append(y.copy())
        return {'signals': signals, 'series': series, 'residual': y, 'fap': fap}

    def plot_prewhitening(self, pw, mark=(), figsize=None):
        """One row per step of prewhiten (pw: its dict): the series with the sinusoid subtracted and its periodogram
        with the FAP threshold. Returns fig, axes."""
        import matplotlib.pyplot as plt
        n = len(pw['series'])
        fig, axes = plt.subplots(n, 2, figsize=figsize or (14, 3 * n), squeeze=False)
        for i, y in enumerate(pw['series']):
            axes[i, 0].plot(self.t, y, '.', ms=2, color='C0')
            if i < len(pw['signals']):
                tt = np.linspace(self.t[0], self.t[-1], 2000)
                axes[i, 0].plot(tt, self.model(pw['signals'][i], tt), color='C3', lw=1)
            self.plot_periodogram(axes[i, 1], y=y, fap_levels=(pw['fap'],), mark=mark, label='step %d' % i)
        axes[-1, 0].set_xlabel('Time [days]')
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------------------------------------------------------
    # wavelet
    # ------------------------------------------------------------------------------------------------------------------

    def wavelet(self, y=None, omega0=6, dj=1 / 12, s0=None, octaves=7, significance=0.95):
        """Morlet wavelet power spectrum (pycwt) of y (default self.y) after removing a straight line, normalised by
        its variance and divided by the scale. The time series is taken as evenly sampled with the median step (the
        gaps are ignored). s0: smallest scale (default 2 steps); octaves: powers of 2 above it, with 1/dj scales
        each. Returns a dict: 't', 'period', 'power', 'sig' (power / power of the significance level: > 1 is
        significant), 'coi' (cone of influence [days]), 'global_power', 'global_signif', 'dt'."""
        try:
            import pycwt
        except ImportError:
            raise ImportError('the wavelet needs pycwt (pip install pycwt)') from None
        y = self.y if y is None else np.asarray(y, dtype=np.float64)
        dt = float(np.median(np.diff(self.t)))
        y = y - np.polyval(np.polyfit(self.th, y, 1), self.th)
        std = y.std()
        mother = pycwt.Morlet(omega0)
        s0 = 2 * dt if s0 is None else s0
        alpha, _, _ = pycwt.ar1(y)                     # lag-1 autocorrelation (red noise)
        wave, scales, freqs, coi, _, _ = pycwt.cwt(y / std, dt, dj, s0, octaves / dj, mother)
        power = np.abs(wave) ** 2 / scales[:, None]
        signif, _ = pycwt.significance(1.0, dt, scales, 0, alpha, significance_level=significance, wavelet=mother)
        glbl_signif, _ = pycwt.significance(std ** 2, dt, scales, 1, alpha, significance_level=significance,
                                            dof=len(y) - scales, wavelet=mother)
        return {'t': self.t, 'period': 1 / freqs, 'power': power, 'sig': power / signif[:, None], 'coi': coi,
                'global_power': power.mean(axis=1), 'global_signif': glbl_signif, 'dt': dt}

    def plot_wavelet(self, ax=None, wv=None, period_range=(2, 256), levels=(0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16),
                     cmap='viridis'):
        """Wavelet power (wv: the dict of wavelet, computed if None) on a log2 period axis [days], the contour of the
        significance and the cone of influence (hatched). Returns ax."""
        import matplotlib.pyplot as plt
        wv = self.wavelet() if wv is None else wv
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 5))
        t, lp, dt = wv['t'], np.log2(wv['period']), wv['dt']
        cs = ax.contourf(t, lp, np.log2(wv['power']), np.log2(levels), extend='both', cmap=cmap)
        ax.contour(t, lp, wv['sig'], [-99, 1], colors='k', linewidths=1.5)
        ax.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
                np.concatenate([np.log2(wv['coi']), [1e-9], lp[-1:], lp[-1:], [1e-9]]),
                'k', alpha=0.3, hatch='x')
        ticks = 2.0 ** np.arange(np.ceil(np.log2(period_range[0])), np.floor(np.log2(period_range[1])) + 1)
        ax.set_yticks(np.log2(ticks))
        ax.set_yticklabels(['%g' % v for v in ticks])
        ax.set_ylim(np.log2(period_range[0]), np.log2(period_range[1]))
        ax.set_ylabel('Period [days]')
        ax.set_xlabel('Time [days]')
        if self.name:
            ax.set_title(self.name)
        plt.colorbar(cs, ax=ax, label='log$_2$ power')
        return ax

    # ------------------------------------------------------------------------------------------------------------------
    # injection and recovery of planets
    # ------------------------------------------------------------------------------------------------------------------

    def inject(self, periods, amplitudes, n_phases=100, fap=1e-3, fit=True, esinw=0.0, ecosw=0.0, verbose=True):
        """Injects y + planet.keplerian_orbit(t, [P, K, esinw, ecosw, t_transit]) for every period P [days],
        semi-amplitude K (unit of y) and n_phases times of transit t_transit = t[0] + P i / n_phases, and computes the
        GLS of every injected series. fit: for the series with FAP <= fap, the sinusoid of sinusoid() is fitted with
        a free period (scipy curve_fit, from the GLS peak).
        Returns (and keeps in self.injection) a dict: 'periods', 'amplitudes', 't_transit' (n_periods, n_phases),
        'fap_fit' (= fap) and arrays (n_periods, n_amplitudes, n_phases): 'P', 'amp', 'fap' (GLS peak), 'P_fit',
        'amp_fit', 'phi_fit' (NaN if not fitted)."""
        from scipy.optimize import curve_fit
        periods = np.atleast_1d(np.asarray(periods, dtype=np.float64))
        amplitudes = np.atleast_1d(np.asarray(amplitudes, dtype=np.float64))
        shape = (len(periods), len(amplitudes), n_phases)
        out = {'periods': periods, 'amplitudes': amplitudes, 'fap_fit': fap,
               't_transit': self.t[0] + periods[:, None] * np.arange(n_phases)[None, :] / n_phases}
        out.update({k: np.full(shape, np.nan) for k in ('P', 'amp', 'fap', 'P_fit', 'amp_fit', 'phi_fit')})
        for i, P in enumerate(periods):
            unit = np.stack([planet.keplerian_orbit(self.t, [P, 1.0, esinw, ecosw, tt]) for tt in out['t_transit'][i]],
                            axis=1)                                     # K = 1, (N, n_phases)
            for j, K in enumerate(amplitudes):
                Y = self.y[:, None] + K * unit
                pg = self.periodogram(Y)
                out['P'][i, j], out['amp'][i, j], out['fap'][i, j] = pg['P'], pg['amp'], pg['fap']
                if not fit:
                    continue
                for k in np.flatnonzero(pg['fap'] <= fap):
                    try:
                        p, _ = curve_fit(sinusoid, self.th, Y[:, k], maxfev=20000,
                                         p0=[pg['P'][k], pg['amp'][k], pg['phi'][k], pg['offset'][k]])
                    except RuntimeError:
                        continue
                    out['P_fit'][i, j, k], out['amp_fit'][i, j, k] = p[0], abs(p[1])
                    out['phi_fit'][i, j, k] = np.mod(p[2] + (np.pi if p[1] < 0 else 0), 2 * np.pi)
            if verbose:
                print('P = %7.2f d: detected (FAP <= %g) %5.1f %% of the injections'
                      % (P, fap, 100 * np.mean(out['fap'][i] <= fap)))
        self.injection = out
        return out

    def detection_rate(self, fap=None, period_tol=None, at_period=None, injection=None):
        """Percentage of the injections (inject) detected, shape (n_amplitudes, n_periods):
          FAP <= fap (default the fap of inject);
          period_tol: and the GLS period within period_tol (relative) of the injected period;
          at_period : instead, the GLS period within period_tol (default 0.1) of at_period (e.g. the rotation of the
                      star, 26 days) and closer to it than the injected period (the signal found is not the planet)."""
        inj = self.injection if injection is None else injection
        if inj is None:
            raise ValueError('run inject first')
        fap = inj['fap_fit'] if fap is None else fap
        P_inj = inj['periods'][:, None, None]
        found = inj['fap'] <= fap
        if at_period is not None:
            tol = 0.1 if period_tol is None else period_tol
            d = np.abs(inj['P'] - at_period) / at_period
            found &= (d <= tol) & (d < np.abs(P_inj - at_period) / at_period)
        elif period_tol is not None:
            found &= np.abs(inj['P'] - P_inj) / P_inj <= period_tol
        return 100 * found.mean(axis=2).T

    def detection_limit(self, level=50, injection=None, **kw):
        """Semi-amplitude at which detection_rate(**kw) reaches level [%], for every injected period (linear
        interpolation between the semi-amplitudes of the grid; NaN if it never reaches it)."""
        inj = self.injection if injection is None else injection
        rate = self.detection_rate(injection=inj, **kw)
        K = inj['amplitudes']
        out = np.full(rate.shape[1], np.nan)
        for j in range(rate.shape[1]):
            above = np.flatnonzero(rate[:, j] >= level)
            if len(above) == 0:
                continue
            i = above[0]
            out[j] = K[0] if i == 0 else np.interp(level, rate[i - 1:i + 1, j], K[i - 1:i + 1])
        return out

    def plot_detection_map(self, rate, ax=None, injection=None, title=None, label='Percentage of detections',
                           cmap=None, vmin=0, vmax=100, mark=(365,)):
        """Map of rate (detection_rate, or any array (n_amplitudes, n_periods)) against the injected period and
        semi-amplitude, with vertical lines at the periods mark. cmap: default cmasher sunburst if cmasher is
        installed, else inferno. Returns ax."""
        import matplotlib.pyplot as plt
        inj = self.injection if injection is None else injection
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if cmap is None:
            try:
                import cmasher as cmr
                cmap = cmr.get_sub_cmap('cmr.sunburst', 0, 0.95)
            except ImportError:
                cmap = 'inferno'
        im = ax.pcolormesh(inj['periods'], inj['amplitudes'], rate, shading='nearest', cmap=cmap, vmin=vmin,
                           vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        for p in mark:
            ax.axvline(p, color='#76a4f4')
        ax.set_xlabel('Injected period [days]')
        ax.set_ylabel('Semi-amplitude [m/s]')
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_recovered(self, ax=None, injection=None, fap=None, fitted=True, cmap='viridis'):
        """Median recovered semi-amplitude against the injected one (left) and median recovered period against the
        injected one (right), over the detected injections (FAP <= fap), one colour per period (left) or per
        semi-amplitude (right). fitted: the values of the fit, else of the GLS peak. Returns the two axes."""
        import matplotlib.pyplot as plt
        inj = self.injection if injection is None else injection
        fap = inj['fap_fit'] if fap is None else fap
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(14, 5))
        amp = np.where(inj['fap'] <= fap, inj['amp_fit' if fitted else 'amp'], np.nan)
        per = np.where(inj['fap'] <= fap, inj['P_fit' if fitted else 'P'], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)             # all-NaN cells: nothing detected
            amp_med, per_med = np.nanmedian(amp, axis=2), np.nanmedian(per, axis=2)
        cm = plt.get_cmap(cmap)
        for i, P in enumerate(inj['periods']):
            ax[0].plot(inj['amplitudes'], amp_med[i], 'o', ms=4, color=cm(i / max(1, len(inj['periods']) - 1)),
                       label='%.0f d' % P)
        for j, K in enumerate(inj['amplitudes']):
            ax[1].plot(inj['periods'], per_med[:, j], 'v', ms=4, color=cm(j / max(1, len(inj['amplitudes']) - 1)))
        lim = [0, 1.1 * inj['amplitudes'].max()]
        ax[0].plot(lim, lim, 'k', alpha=0.4)
        ax[0].set_xlim(lim)
        ax[0].set_xlabel('Injected semi-amplitude [m/s]')
        ax[0].set_ylabel('Recovered semi-amplitude [m/s]')
        ax[0].legend(ncols=2, fontsize='small')
        lim = [0, 1.1 * inj['periods'].max()]
        ax[1].plot(lim, lim, 'k', alpha=0.4)
        ax[1].set_xlabel('Injected period [days]')
        ax[1].set_ylabel('Recovered period [days]')
        return ax
