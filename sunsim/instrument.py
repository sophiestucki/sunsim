"""
Instrument: what a spectrograph does to the spectrum of the star.

    resolution   the spectrum is convolved with the instrumental profile (a gaussian of FWHM c/R, spectrum_utils.add_resol)
    orders       the light is recorded in orders: every order has its own pixels (wavelengths) and blaze
    efficiency   the total efficiency of the instrument, multiplied to the blaze of every order

Used by
    CCF_grid(..., instrument=Instrument(...))   the CCFs order by order (the CCF of every order, orders summed)
    StarSim(..., instrument=Instrument(...))    mode 'spectroscopy': the simulated spectra as the instrument records
                                                them, one spectrum per order (StarSim.spec_var = {order: {...}})

    harps = Instrument.harps()                  # HARPS: R = 115000, the 72 orders of the files of the package
    harps.observe(wavelengths, spectra)         # {order: {'wav': (n_pix,), 'flux': (n_spectra, n_pix), 'blaze': ...}}

The instrument is linear in the flux (convolution, resampling, blaze), and the same for every cell of the disc: applied
to the disc-integrated spectrum it gives the same as applied to every cell before the sum, and it is much cheaper.
"""
import os
import warnings
import numpy as np

from . import _checks
from . import nbspectra

C_LIGHT = 2.99792458e8          # [m/s]


def _instrument_files():
    """Folder of the instrument files installed with the package (sunsim/data/instruments)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'instruments')


class Instrument(object):
    """
    A spectrograph: resolution, orders and efficiency.

    resolution    : 'HARPS', 'HARPS-N', 'EXPRESS', 'NEID' (spectrum_utils.INSTRUMENT_RESOLUTION) or the resolving power
                    R as a number
    orders        : None -> no orders: the spectrum is only degraded to the resolution (one spectrum, the wavelengths
                    of the input). (wavelengths, blaze) -> the pixels of every order: two arrays (n_orders, n_pix) or
                    lists of arrays (orders of different lengths); blaze None -> 1 for every pixel.
    efficiency    : None | callable eff(wavelength) | (wavelengths, efficiency) table (interpolated, constant outside).
                    Evaluated on the wavelengths of every order and multiplied by its blaze. Needs orders.
    orders_medium : 'air' (default: HARPS / HARPS-N pipelines) or 'vacuum', the medium of the wavelengths of the orders
    name          : label (default: the resolution if it is a name)
    order_numbers : the number given to every order in the outputs (default 0, 1, ..., n_orders - 1: the index of the
                    order in the files)
    """

    def __init__(self, resolution, orders=None, efficiency=None, orders_medium='air', name=None, order_numbers=None):
        from .spectrum_utils import INSTRUMENT_RESOLUTION
        if isinstance(resolution, str):
            if resolution not in INSTRUMENT_RESOLUTION:
                raise ValueError('unknown instrument %r: give one of %s, or the resolving power R as a number'
                                 % (resolution, sorted(INSTRUMENT_RESOLUTION)))
            R = INSTRUMENT_RESOLUTION[resolution]
        else:
            resolution = R = _checks.number('resolution (resolving power)', resolution, positive=True)
        if orders_medium not in ('air', 'vacuum'):
            raise ValueError("orders_medium must be 'air' or 'vacuum', got %r" % (orders_medium,))
        if efficiency is not None and orders is None:
            raise ValueError('efficiency needs the orders (the efficiency multiplies the blaze of every order)')
        if efficiency is not None and not callable(efficiency):
            if len(efficiency) != 2:
                raise ValueError('efficiency must be None, a function eff(wavelength) or a table (wavelengths, efficiency)')
        self.resolution = resolution            # as given (a name or R), what add_resol takes
        self.R = float(R)
        self.name = name if name is not None else (resolution if isinstance(resolution, str) else 'R = %g' % R)
        self.orders_medium = orders_medium
        self.efficiency = efficiency
        self.wave_orders, self.blaze_orders = None, None
        if orders is not None:
            wave, blaze = orders
            if blaze is None:
                blaze = [np.ones(len(w)) for w in wave]
            if len(wave) != len(blaze):
                raise ValueError('orders: %d orders of wavelengths but %d of blaze' % (len(wave), len(blaze)))
            for j, (w, b) in enumerate(zip(wave, blaze)):
                if len(w) != len(b):
                    raise ValueError("order %d: wavelength and blaze have different lengths" % j)
            self.wave_orders, self.blaze_orders = wave, blaze
        n = 0 if orders is None else len(self.wave_orders)
        self.order_numbers = list(range(n)) if order_numbers is None else list(order_numbers)
        if len(self.order_numbers) != n:
            raise ValueError('order_numbers: %d numbers for %d orders' % (len(self.order_numbers), n))
        self._tables = {}

    def __repr__(self):
        n = 0 if self.wave_orders is None else len(self.wave_orders)
        return 'Instrument(%s, R = %g, %d orders%s)' % (self.name, self.R, n,
                                                       ', with efficiency' if self.efficiency is not None else '')

    # ---------------------------------------------------------------- instruments with files in the package
    @classmethod
    def harps(cls, efficiency=None):
        """HARPS: R = 115000 and the wavelengths and blaze of its 72 orders (files of sunsim/data/instruments, in air).
        efficiency: None (blaze only), or the efficiency of your HARPS data (see Instrument)."""
        from astropy.io import fits
        folder = _instrument_files()
        wave = fits.getdata(os.path.join(folder, 'HARPS.2024-10-17T21-57-53.302_wave_A.fits')).astype(np.float64)
        blaze = fits.getdata(os.path.join(folder, 'HARPS.2025-12-16T23-21-06.037_blaze_A.fits')).astype(np.float64)
        return cls('HARPS', orders=(wave, blaze), efficiency=efficiency, orders_medium='air')

    @classmethod
    def harpsn_example(cls):
        """The HARPS-N example of the tutorial (and of the old StarSim): R = 118000 and the total efficiency of HARPS-N,
        on the orders of HARPS (the only order files in the package)."""
        folder = _instrument_files()
        h = cls.harps()
        t = np.genfromtxt(os.path.join(folder, 'HARPS-N_total_efficiency.csv'), delimiter=',', names=True)
        efficiency = (10 * t[t.dtype.names[0]], t[t.dtype.names[1]] / 100)          # [nm] -> [A], % -> fraction
        return cls('HARPS-N', orders=(h.wave_orders, h.blaze_orders), efficiency=efficiency, orders_medium='air')

    # ---------------------------------------------------------------- properties
    @property
    def has_orders(self):
        return self.wave_orders is not None

    @property
    def n_orders(self):
        return 0 if self.wave_orders is None else len(self.wave_orders)

    # ---------------------------------------------------------------- the steps
    def _to_medium(self, w, given, medium):
        """Wavelengths w in `given` medium converted to `medium` (float64)."""
        from .spectrum_utils import vacuum2air, air2vacuum
        w = np.asarray(w, dtype=np.float64)
        if given == medium:
            return w
        return np.asarray((vacuum2air if medium == 'air' else air2vacuum)(w))

    def _efficiency_on(self, j, wj):
        """Efficiency on the wavelengths wj of order j (1.0 without efficiency)."""
        eff = self.efficiency
        if eff is None:
            return 1.0
        if callable(eff):
            return np.asarray(eff(wj), dtype=np.float64)
        wl, ef = (np.asarray(x, dtype=np.float64) for x in eff)
        srt = np.argsort(wl)
        wl, ef = wl[srt], ef[srt]
        if wj[0] < wl[0] or wj[-1] > wl[-1]:
            warnings.warn("order %d (%.1f-%.1f A) is partly outside the efficiency table (%.1f-%.1f A): "
                          "the efficiency is kept constant outside" % (j, wj[0], wj[-1], wl[0], wl[-1]))
        return np.interp(wj, wl, ef)

    def order_table(self, medium='air', wv_max=np.inf):
        """The orders that start below wv_max, in `medium`: list of dicts with 'order' (number), 'index' (position in
        the files), 'wav' (wavelengths of the pixels), 'edges' (edges of the pixels, float64) and 'b' (blaze x
        efficiency). Built once per (medium, wv_max)."""
        if not self.has_orders:
            raise RuntimeError('%r has no orders' % (self,))
        key = (medium, float(wv_max))
        if key in self._tables:
            return self._tables[key]
        out = []
        for j in range(self.n_orders):
            wj = self._to_medium(self.wave_orders[j], self.orders_medium, medium)
            bj = np.asarray(self.blaze_orders[j], dtype=np.float64)
            if not wj[0] < wv_max:                       # order starts beyond the end of the spectrum
                continue
            out.append({'order': self.order_numbers[j], 'index': j, 'wav': wj,
                        'edges': nbspectra.bin_edges_nb(wj), 'b': bj * self._efficiency_on(j, wj)})
        self._tables[key] = out
        return out

    def degrade(self, wavelength, flux):
        """flux (1D, or 2D: one spectrum per row) degraded to the resolution (spectrum_utils.add_resol). Identical rows
        are degraded once."""
        from .spectrum_utils import add_resol
        wavelength = np.asarray(wavelength, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if flux.ndim == 1:
            return np.asarray(add_resol(wavelength, flux, self.resolution), dtype=np.float64)
        out = np.empty_like(flux)
        done = {}
        for i, row in enumerate(flux):
            k = row.tobytes()
            if k not in done:
                done[k] = np.asarray(add_resol(wavelength, row, self.resolution), dtype=np.float64)
            out[i] = done[k]
        return out

    def observe(self, wavelength, flux, medium='vacuum'):
        """The spectra flux (1D or 2D, one spectrum per row, on `wavelength` [A] in `medium`) as the instrument records
        them:
          1. degraded to the resolution (degrade). Every spectrum is divided by its maximum before, and multiplied by it
             after, so that the continuum matches the value 1 that add_resol puts beyond the ends. The pixels within
             0.1 A + 3 c/R of the ends of the spectrum (where add_resol extrapolates) are set to NaN.
          2. with orders: resampled on the pixels of every order, conserving the flux
             (nbspectra.flux_conserving_resample_nb), and multiplied by blaze x efficiency. The pixels of an order
             outside the simulated wavelengths are NaN; the orders that do not overlap them are left out.
        Returns, with orders, {order number: {'wav': (n_pix,) wavelengths in the medium of the orders, 'flux':
        (n_spectra, n_pix) or (n_pix,), 'blaze': (n_pix,) blaze x efficiency}}, in the order of the files; without
        orders, the degraded spectra on `wavelength`."""
        if medium not in ('air', 'vacuum'):
            raise ValueError("medium must be 'air' or 'vacuum', got %r" % (medium,))
        wv = np.asarray(wavelength, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        one = flux.ndim == 1
        flux2 = np.atleast_2d(flux)
        if flux2.shape[1] != len(wv):
            raise ValueError('the spectra have %d wavelengths, `wavelength` has %d' % (flux2.shape[1], len(wv)))
        scale = np.nanmax(np.abs(flux2), axis=1)[:, None]
        scale[~(scale > 0)] = 1.0
        degraded = self.degrade(wv, flux2 / scale) * scale
        edge = 0.1 + 3 * wv[[0, -1]] / self.R
        degraded[:, (wv < wv[0] + edge[0]) | (wv > wv[-1] - edge[1])] = np.nan
        if not self.has_orders:
            return degraded[0] if one else degraded

        wv = self._to_medium(wv, medium, self.orders_medium)    # in the medium of the orders
        edges = nbspectra.bin_edges_nb(wv)
        out = {}
        for o in self.order_table(self.orders_medium, wv[-1]):
            if not o['wav'][-1] > wv[0]:                         # order ends before the spectrum starts
                continue
            f = np.array([o['b'] * nbspectra.flux_conserving_resample_nb(edges, np.ascontiguousarray(row), o['edges'])
                          for row in degraded])
            out[o['order']] = {'wav': o['wav'], 'flux': f[0] if one else f, 'blaze': o['b']}
        return out
