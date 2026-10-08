"""
active_region_mask: active regions given as 2D maps of the disc (e.g. from SDO images), one map of the spots and one of the
faculae per epoch, for StarSim(..., active_regions_mask=active_region_mask(...)).

Files
-----
In one folder, for every epoch t (in days, the same unit as the rest of SunSim):
    spot_map_{t}.npy     or  spot_map_{t}.txt
    faculae_map_{t}.npy  or  faculae_map_{t}.txt
(the names can be changed with spot_pattern / faculae_pattern). {t} is any number (12, 12.5, 2459000.25, 1e3).
Every map is a square image of the visible disc, the disc filling the image (its diameter is the size of the image),
with the fraction of every pixel covered by spots (resp. faculae): 0 = quiet, 1 = fully covered (binary masks work).
Pixels outside the disc are ignored (they can be NaN). If only one of the two maps exists for an epoch, the other
one is taken as 0 everywhere.
.npy files are much faster to read than .txt (a 4096 x 4096 text map is ~35-200 MB): with cache_npy=True every text
map is converted to .npy (next to it) the first time it is read, and the .npy is used afterwards. Binary masks saved
as uint8 or bool .npy are 4 times smaller (and faster to read) than float32.

Orientation (see nbspectra.pixel_to_cell_nb): the horizontal axis of the image is the projected stellar equator, with
the rotation moving the regions from left to right (as east -> west on the Sun with north up), the vertical axis is
the projected rotation axis. origin='upper' (default): row 0 of the array is the top of the image (north), as
np.loadtxt reads a text image and as imshow shows it; origin='lower': row 0 is the bottom. flip_horizontal=True if the
regions move from right to left in your images.

How the maps are used
---------------------
For a grid of the disc (N_rings), every pixel is assigned once to a cell of the grid (projection, cached for every
image size). At every epoch the spot and facula filling factors of a cell are the means of the two maps over its
pixels; StarSim then computes the light curve / spectra / CCFs exactly as with circular regions (planets included).
The epochs are read and reduced in parallel threads (n_threads, a few maps in memory at a time). With
cache_filling_factors=True the filling factors are saved in the folder (one hidden .npz file per number of rings),
and a new run with the same maps and number of rings does not read the maps again (useful when only the spectra,
the grids or the mode change). The cache is checked against the size and date of every map file.
"""
import os
import re
import sys
import time as _time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import nbspectra


class active_region_mask(object):
    """
    Spot and facula maps of the disc at several epochs (see the module docstring).

    folder          : folder with the maps
    spot_pattern, faculae_pattern : file names without extension, with {time} where the epoch is
    origin          : 'upper' (row 0 = top of the image) or 'lower' (row 0 = bottom)
    flip_horizontal : True if the regions move from right to left in the images
    cache_npy       : convert the .txt maps to .npy the first time they are read (written in the same folder)
    dtype           : type of the text maps in memory (float32 halves the memory of large maps); .npy maps keep
                      their own type (uint8 / bool masks are the smallest)
    n_threads       : number of epochs read and reduced at the same time (default: min(4, number of CPUs)); every
                      thread holds one pair of maps in memory
    cache_filling_factors : save / reuse the filling factors of every epoch in the folder (see the module docstring)

    Attributes: times (sorted epochs found in the folder), files {time: {'sp': path or None, 'fc': path or None}}.
    """
    def __init__(self, folder, spot_pattern='spot_map_{time}', faculae_pattern='faculae_map_{time}', origin='upper',
                 flip_horizontal=False, cache_npy=False, dtype=np.float32, n_threads=None, cache_filling_factors=False):
        if origin not in ('upper', 'lower'):
            raise ValueError("origin must be 'upper' or 'lower', got %r" % (origin,))
        if not os.path.isdir(folder):
            raise FileNotFoundError("the folder of the maps does not exist: %s" % folder)
        self.folder = folder
        self.origin = origin
        self.flip_horizontal = flip_horizontal
        self.cache_npy = cache_npy
        self.dtype = dtype
        self.n_threads = max(1, min(4, os.cpu_count() or 1)) if n_threads is None else max(1, int(n_threads))
        self.cache_filling_factors = cache_filling_factors
        self._projections = {}          # (n_rows, n_cols, N_rings) -> (cell of every pixel, number of pixels of every cell)
        self._lock = threading.Lock()

        #find the files: {time} is any number; .npy is preferred to .txt when both exist
        number = r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
        files = {}
        for kind, pattern in (('sp', spot_pattern), ('fc', faculae_pattern)):
            if '{time}' not in pattern:
                raise ValueError("the pattern must contain {time}: %r" % (pattern,))
            regex = re.compile('^' + re.escape(pattern).replace(re.escape('{time}'), number) + r'\.(npy|txt)$')
            for name in os.listdir(folder):
                match = regex.match(name)
                if not match:
                    continue
                t = float(match.group(1))
                entry = files.setdefault(t, {'sp': None, 'fc': None})
                path = os.path.join(folder, name)
                if entry[kind] is None or name.endswith('.npy'):
                    entry[kind] = path
        if not files:
            raise FileNotFoundError("no map %s.npy/.txt or %s.npy/.txt in %s"
                                    % (spot_pattern.replace('{time}', '*'), faculae_pattern.replace('{time}', '*'), folder))
        self.files = files
        self.times = np.array(sorted(files))

    # ---------------------------------------------------------------- reading
    def _read(self, path):
        """One map as a 2D array (npy, or text with spaces / tabs / commas between the values)."""
        if path.endswith('.npy'):
            return self._native(np.load(path))
        npy = path[:-4] + '.npy'
        if os.path.exists(npy):
            return self._native(np.load(npy))
        with open(path) as f:
            first = f.readline()
        sep = ',' if ',' in first else r'\s+'
        try:                                    # pandas' C parser: much faster than np.loadtxt for large maps
            import pandas as pd
            arr = pd.read_csv(path, sep=sep, header=None, dtype=self.dtype, engine='c').to_numpy()
        except ImportError:
            arr = np.loadtxt(path, delimiter=',' if sep == ',' else None, dtype=self.dtype)
        if self.cache_npy:
            np.save(npy, arr)
        return arr

    @staticmethod
    def _native(arr):
        """.npy maps keep their type (no copy); bool masks are seen as uint8 (same memory, numeric for the kernels)."""
        return arr.view(np.uint8) if arr.dtype == np.bool_ else arr

    def load(self, time):
        """Spot and facula maps of an epoch (2D arrays; a missing map is all zeros)."""
        entry = self._entry(time)
        sp = self._read(entry['sp']) if entry['sp'] is not None else None
        fc = self._read(entry['fc']) if entry['fc'] is not None else None
        if sp is None:
            sp = np.zeros_like(fc)
        if fc is None:
            fc = np.zeros_like(sp)
        if sp.shape != fc.shape or sp.ndim != 2:
            raise ValueError("the spot and facula maps at t=%s must be 2D with the same shape, got %s and %s"
                             % (time, sp.shape, fc.shape))
        return sp, fc

    def _entry(self, time):
        """Files of the epoch closest to `time` (it must match to 1e-6 days)."""
        k = int(np.argmin(np.abs(self.times - time)))
        if abs(self.times[k] - time) > 1e-6 * max(1.0, abs(time)):
            raise ValueError("no map for the epoch t=%s (the maps are at %s ... %s)" % (time, self.times[0], self.times[-1]))
        return self.files[self.times[k]]

    # ---------------------------------------------------------------- projection on the grid
    def projection(self, shape, N_rings, Ngrid_in_ring):
        """Cell of every pixel (flattened image, int32, -1 outside the disc) and number of pixels of every cell, for an
        image of this shape and a grid of N_rings rings (computed once, cached; thread-safe)."""
        key = (shape[0], shape[1], N_rings)
        with self._lock:
            if key not in self._projections:
                if shape[0] != shape[1]:
                    warnings.warn("the maps are not square (%d x %d): the disc is taken as an ellipse filling the image" % shape)
                nin = np.asarray(Ngrid_in_ring, dtype=np.int64)
                cells = nbspectra.pixel_to_cell_nb(shape[0], shape[1], N_rings, nin, self.origin == 'upper',
                                                   bool(self.flip_horizontal))
                npix = np.bincount(cells[cells >= 0], minlength=int(nin.sum())).astype(np.float64)
                self._projections[key] = (cells, npix)
            return self._projections[key]

    # ---------------------------------------------------------------- cache of the filling factors
    def _signature(self, time):
        """Name, size and date of the map files of an epoch: the cache is valid only if they did not change."""
        entry = self._entry(time)
        parts = []
        for kind in ('sp', 'fc'):
            path = entry[kind]
            if path is None:
                parts.append('-')
            else:
                st = os.stat(path)
                parts.append('%s:%d:%d' % (os.path.basename(path), st.st_size, int(st.st_mtime)))
        return '|'.join(parts)

    def _cache_path(self, N_rings):
        return os.path.join(self.folder, '.sunsim_filling_factors_N%d_%s%s.npz'
                            % (N_rings, self.origin, '_flip' if self.flip_horizontal else ''))

    def _read_cache(self, N_rings):
        """{signature of the files of an epoch: (ff_sp, ff_fc)} from the cache file (empty if there is none)."""
        path = self._cache_path(N_rings)
        if not os.path.exists(path):
            return {}
        try:
            d = np.load(path)
            return {str(sig): (d['ff_sp'][k], d['ff_fc'][k]) for k, sig in enumerate(d['signatures'])}
        except Exception:
            return {}

    def _write_cache(self, N_rings, cache):
        sigs = list(cache)
        try:
            np.savez(self._cache_path(N_rings), signatures=np.array(sigs),
                     ff_sp=np.array([cache[k][0] for k in sigs]), ff_fc=np.array([cache[k][1] for k in sigs]))
        except OSError as e:
            warnings.warn("could not write the cache of the filling factors in %s: %s" % (self.folder, e))

    # ---------------------------------------------------------------- filling factors
    def filling_factors(self, times, N_rings, Ngrid_in_ring, cell_area=None, verbose=False):
        """Spot and facula filling factors of every cell at every epoch: two arrays (n_times, n_cells).
        The epochs are read and reduced in n_threads parallel threads; the projection is computed once per image
        size; with cache_filling_factors the epochs already computed for this number of rings are not read again.
        cell_area (projected area of every cell, optional): to warn only if the cells without any pixel (the narrow
        rings at the limb when the maps are small) cover more than 0.1% of the disc, and to print the filling factors
        as fractions of the disc.
        verbose: print the progress, one line updated at every epoch:
            Date t. ff_ph=..%. ff_sp=..%. ff_fc=..%. [k/n] elapsed / remaining time
        (fractions of the projected disc, from the maps, before the planets)."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        ncell = int(np.sum(Ngrid_in_ring))
        ff_sp = np.zeros((len(times), ncell))
        ff_fc = np.zeros((len(times), ncell))
        cache = self._read_cache(N_rings) if self.cache_filling_factors else {}
        sigs = [self._signature(t) for t in times] if self.cache_filling_factors else [None] * len(times)
        todo = []
        for k, sig in enumerate(sigs):
            if sig is not None and sig in cache and len(cache[sig][0]) == ncell:
                ff_sp[k], ff_fc[k] = cache[sig]
            else:
                todo.append(k)

        def one_epoch(k):
            sp, fc = self.load(times[k])
            cells, npix = self.projection(sp.shape, N_rings, Ngrid_in_ring)
            a, b = nbspectra.maps_to_cells_nb(cells, np.ascontiguousarray(sp).ravel(), np.ascontiguousarray(fc).ravel(), npix)
            return k, a, b, npix, sp.shape

        area = np.ones(ncell) if cell_area is None else np.asarray(cell_area, dtype=np.float64)
        area = area / area.sum()

        def status(k, done, start):
            """Progress line of epoch k (done epochs computed so far)."""
            sp_pc, fc_pc = 100*np.dot(ff_sp[k], area), 100*np.dot(ff_fc[k], area)
            elapsed = _time.perf_counter() - start
            left = elapsed / done * (len(todo) - done) if done else 0.0
            sys.stdout.write("\rDate {0:.4f}. ff_ph={1:.3f}%. ff_sp={2:.3f}%. ff_fc={3:.3f}%. [{4}/{5}] {6:.1f} s, ~{7:.1f} s left   "
                             .format(times[k], 100-sp_pc-fc_pc, sp_pc, fc_pc, done, len(todo), elapsed, left))
            sys.stdout.flush()

        if verbose:
            print("2D maps: %d epochs, %d rings (%d cells)%s" % (len(times), N_rings, ncell,
                  ", %d from the cache" % (len(times) - len(todo)) if len(todo) < len(times) else ""))
        start = _time.perf_counter()
        empty_warned = False
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            for done, (k, a, b, npix, shape) in enumerate(pool.map(one_epoch, todo), 1):
                ff_sp[k], ff_fc[k] = a, b
                if verbose:
                    status(k, done, start)
                if sigs[k] is not None:
                    cache[sigs[k]] = (a, b)
                if not empty_warned and np.any(npix == 0):
                    empty = npix == 0
                    frac = 1.0 if cell_area is None else np.sum(np.asarray(cell_area)[empty]) / np.sum(cell_area)
                    if frac > 1e-3:
                        warnings.warn("%d cells of the grid (%.1f%% of the disc) have no pixel in the %d x %d maps (they "
                                      "are taken as quiet): use larger maps or fewer rings"
                                      % (np.sum(empty), 100*frac, shape[0], shape[1]))
                    empty_warned = True
        if verbose and todo:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if self.cache_filling_factors and todo:
            self._write_cache(N_rings, cache)
        return ff_sp, ff_fc
