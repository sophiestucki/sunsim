#!/usr/bin/env python
"""
The two CCF masks of the PoET simulation (poet.dag), in the format of the old configuration (ccf_mask: three columns
wv1, wv2, weight, air wavelengths [A] at rest; wv1 = wv2 = the line, read as their mean and converted to vacuum):

  masks/ESPRESSO_SUN.mas   ESPRESSO mask (SUN_LINELIST_MASK.csv: wave, cont), weight = cont^2 (as masks.ipynb and
                           mask_pipeline.load_espresso_mask)
  masks/PoET_noflag.mas    lines of the PoET mask with no flag at all (every flag_* column of lines_flags.csv of
                           making_mask_flags.ipynb False): Gaussian minimum of the PoET stack (obs_wave_min), weight
                           Bouchy et al. 2001 eq. 8 of the PoET stack (obs_weight)
The weights are summed to 1 (the old configuration normalises them again). Run on the computer with the CAUP outputs:

  python poet_masks.py --poet-output ~/Documents/poet/output
"""
import argparse
import os

import numpy as np
import pandas as pd


def write_mas(path, wave, weight):
    order = np.argsort(wave)
    wave, weight = np.asarray(wave, dtype=np.float64)[order], np.asarray(weight, dtype=np.float64)[order]
    weight = weight / np.sum(weight)
    np.savetxt(path, np.column_stack([wave, wave, weight]), fmt=['%.6f', '%.6f', '%.6e'], delimiter='\t')
    print('%s: %d lines, %.2f-%.2f A' % (path, len(wave), wave[0], wave[-1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--poet-output', required=True, help='output folder of the CAUP notebooks')
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masks'))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    esp = pd.read_csv(os.path.join(a.poet_output, 'SUN_LINELIST_MASK.csv'))
    write_mas(os.path.join(a.out, 'ESPRESSO_SUN.mas'), esp['wave'], esp['cont'] ** 2)

    lines = pd.read_csv(os.path.join(a.poet_output, 'mask_flags', 'lines_flags.csv'))
    flags = [c for c in lines.columns if c.startswith('flag_')]
    keep = ~lines[flags].any(axis=1)
    print('PoET: %d of %d lines without any of the flags %s' % (keep.sum(), len(lines), ', '.join(flags)))
    write_mas(os.path.join(a.out, 'PoET_noflag.mas'), lines.loc[keep, 'obs_wave_min'], lines.loc[keep, 'obs_weight'])


if __name__ == '__main__':
    main()
