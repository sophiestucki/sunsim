"""
nbspectra: the numerical kernels of SunSim, compiled with Numba (plus two small NumPy helpers).

Conventions used everywhere in SunSim
-------------------------------------
* The visible disc is cut in N concentric rings around the disc centre (generate_grid_coordinates_nb); every ring is
  cut in cells of similar area. All the cells of a ring have the same mu = cos(angle between the normal and the line
  of sight). Cells are numbered ring by ring, from the disc centre outwards.
* Cartesian coordinates of the cells and of the regions: x towards the observer, y and z in the plane of the sky.
  z is along the projection of the rotation axis of the star, y along the projected equator.
* Angles in radians inside the kernels, velocities in m/s, wavelengths in Angstrom, times in days.

Contents
--------
* differential rotation helpers (NumPy)                     differential_rotation_coeffs, rotation_rate_relative
* linear interpolation and Doppler shift of spectra         interp_linear_nb, interp_doppler_nb, doppler_shift_grid_nb
* grid of the stellar disc                                  generate_grid_coordinates_nb
* filling factors of the cells (regions, planets)          overlap_fraction, generate_ff, generate_ff_core
* CCF of the cells and cross-correlation with a mask        interpolation_nb, loop_compute_immaculate_nb,
                                                            cross_correlation_mask
* CCF bisector and gaussian                                 speed_bisector_nb, gaussian2
* continuum of the spectra (polynomial in one pass)         polyval_mapped_nb
* orbits                                                    true_anomaly_scalar_nb, true_anomaly_nb, ttrans_2_tperi_nb
* the time series (all epochs in one parallel call)         timeseries_nb
* 2D maps of the disc (e.g. SDO)                            pixel_to_cell_nb, maps_to_cells_nb, timeseries_maps_nb
* parameters of the CCFs (all CCFs in one parallel call)    fit_gaussian_nb, ccf_params_nb
"""
import os
os.environ.setdefault('KMP_WARNINGS', '0')   #silence the OpenMP 'omp_set_nested deprecated' info (only works if set before numpy is imported)
import numba as nb
import numpy as np
import math as m

from numba import njit, prange


#######################################################################################################
# Differential rotation (plain numpy, shared by StarSim and flux_grid)
#######################################################################################################

def differential_rotation_coeffs(differential_rotation):
    """Coefficients (c1, c2, ...) [deg/day] of Omega(lat) = Omega_eq + c1 sin^2(lat) + c2 sin^4(lat) + ...
    from 0 / None (rigid rotation), one number (c1), a sequence, or a law of rotation.polynomial_differential_rotation
    (its coefficients). A general law (any function of the latitude) has no coefficients: error."""
    if callable(differential_rotation):
        if hasattr(differential_rotation, 'coefficients'):
            return tuple(differential_rotation.coefficients)
        raise TypeError('this differential rotation is a general law, not polynomial coefficients')
    dr = 0.0 if differential_rotation is None else differential_rotation
    return tuple(np.atleast_1d(np.asarray(dr, dtype=np.float64)).tolist())


def rotation_rate_relative(sin_lat, rotation_period, differential_rotation):
    """Omega(lat) / Omega_eq at the latitudes given by sin(lat), with Omega_eq = 360/rotation_period deg/day.
    differential_rotation: any form accepted by rotation.differential_rotation_law (coefficients or a law)."""
    from .rotation import differential_rotation_law
    sin_lat = np.clip(np.asarray(sin_lat, dtype=np.float64), -1.0, 1.0)
    domega = np.asarray(differential_rotation_law(differential_rotation)(np.rad2deg(np.arcsin(sin_lat))), dtype=np.float64)
    return 1.0 + domega * rotation_period / 360.0



@njit(cache=True)
def interp_linear_nb(x, xp, fp):
    """Linear interpolation of fp(xp) at the points x (xp ascending, x ascending: one pass over both).
    Points outside [xp[0], xp[-1]] are NaN (unlike np.interp, which extends the end values)."""

    n = len(x)
    out = np.empty(n, dtype=np.float64)

    j = 0
    nxp = len(xp)

    for i in range(n):

        xi = x[i]

        while j < nxp-2 and xp[j+1] < xi:
            j += 1

        if xi < xp[0] or xi > xp[-1]:
            out[i] = np.nan

        else:
            x0 = xp[j]
            x1 = xp[j+1]

            y0 = fp[j]
            y1 = fp[j+1]

            out[i] = y0+(y1-y0)*(xi-x0)/(x1-x0)

    return out

@njit(cache=True)
def _interp_doppler_into_nb(wv, flux, new_wv, factor, out, i0, i1):
    """out[i - i0] for i0 <= i < i1: the spectrum (wv, flux) Doppler shifted by factor = 1 + v/c and interpolated
    linearly at new_wv[i], with exactly the operations of interp_doppler_nb (the same values), written in out (no
    allocation). The index of the first interval is found by bisection, then followed as in interp_doppler_nb."""
    nwv = len(wv)
    j = 0
    if i1 > i0:
        x = new_wv[i0]/factor
        j = np.searchsorted(wv[1:], x)                   # first j with wv[j+1] >= x
        if j > nwv-2:
            j = nwv-2
    for i in range(i0, i1):
        x = new_wv[i]/factor
        while j < nwv-2 and wv[j+1] < x:
            j += 1
        if x < wv[0] or x > wv[-1]:
            out[i-i0] = np.nan
        else:
            x0 = wv[j]
            x1 = wv[j+1]
            y0 = flux[j]
            y1 = flux[j+1]
            out[i-i0] = y0+(y1-y0)*(x-x0)/(x1-x0)


@njit(cache=True, parallel=True)
def doppler_disc_total_nb(wv, new_wv, flp, vrel, Ngrid_in_ring, ring_start, n_blocks=64):
    """Sum over all the cells of their Doppler-shifted spectra (doppler_shift_grid_nb), without the grid of the cells:
    the wavelengths are cut in n_blocks blocks done in parallel, and in every block the cells are added in their order,
    as np.sum(doppler_shift_grid_nb(...), axis=0) does (the same values). Memory: the result and one block per thread."""
    c = 2.99792458e8
    n = len(new_wv)
    total = np.zeros(n)
    edges = np.linspace(0, n, n_blocks+1).astype(np.int64)
    for b in prange(n_blocks):
        i0 = edges[b]
        i1 = edges[b+1]
        buf = np.empty(i1-i0)
        for r in range(len(Ngrid_in_ring)):
            for k in range(ring_start[r], ring_start[r]+Ngrid_in_ring[r]):
                _interp_doppler_into_nb(wv, flp[r, :], new_wv, 1.0+vrel[k]/c, buf, i0, i1)
                for i in range(i1-i0):
                    total[i0+i] += buf[i]
    return total


@njit(cache=True)
def interp_doppler_nb(wv, flux, new_wv, velocity):
    """Spectrum (wv, flux) Doppler shifted by `velocity` [m/s] and interpolated linearly on new_wv.
    The flux observed at new_wv was emitted at new_wv / (1 + v/c) (non-relativistic). Outside the original range: NaN."""

    c = 2.99792458e8
    factor = 1.0+velocity/c

    n = len(new_wv)
    out = np.empty(n, dtype=np.float64)

    j = 0
    nwv = len(wv)

    for i in range(n):

        x = new_wv[i]/factor

        while j < nwv-2 and wv[j+1] < x:
            j += 1

        if x < wv[0] or x > wv[-1]:
            out[i] = np.nan

        else:
            x0 = wv[j]
            x1 = wv[j+1]

            y0 = flux[j]
            y1 = flux[j+1]

            out[i] = y0+(y1-y0)*(x-x0)/(x1-x0)

    return out

@njit(cache=True, parallel=True)
def doppler_shift_grid_nb(wv, new_wv, flp, vrel, Ngrid_in_ring, ring_start):
    """Spectrum of every cell, Doppler shifted by the rotation velocity of the cell (parallel over the cells).
    wv, flp     : wavelength grid and spectrum of every RING (N_rings, len(wv)); all the cells of a ring share it
    new_wv      : output wavelength grid (wv shrunk by the largest shifts, so that there are no NaN)
    vrel        : line-of-sight velocity of every cell [m/s]
    Ngrid_in_ring, ring_start : number of cells of every ring and index of its first cell
    Returns an array (n_cells, len(new_wv))."""

    Ngrids = len(vrel)
    Nwave = len(new_wv)

    final_flp = np.empty((Ngrids, Nwave), dtype=np.float64)

    ring_of_pixel = np.empty(Ngrids, dtype=np.int64)

    for i in range(len(Ngrid_in_ring)):
        start = ring_start[i]
        end = start+Ngrid_in_ring[i]

        for k in range(start, end):
            ring_of_pixel[k] = i

    for k in prange(Ngrids):

        ring = ring_of_pixel[k]

        final_flp[k, :] = interp_doppler_nb(wv, flp[ring, :], new_wv, vrel[k])

    return final_flp

#with this the x and y width of each grid is the same, thus the area of the grids is similar in all the sphere, avoiding an over/under sampling of the poles/center
@nb.njit(cache=True,error_model='numpy')
def generate_grid_coordinates_nb(N):
    """Grid of the visible disc with N concentric rings, in a frame where the pole of the grid faces the observer.
    Ring 0 is a single circular cell at the disc centre; the other rings are cut in cells of (about) the same angular
    width as the rings, width = 180/(2N-1) degrees, so that all the cells have a similar area (no over / under sampling of
    the centre or the limb). There are about 2(2N-1)^2/pi cells in total.
    Returns
      Ngrids        : number of cells
      Ngrid_in_ring : number of cells of every ring (list, N)
      centres       : angular distance of every ring from the disc centre [deg] (0 = centre)
      amu           : mu = cos(centres) of every ring
      rs            : projected distance of every cell from the disc centre (sin of its angular distance)
      alphas        : position angle of every cell around the disc centre [deg]
      xs, ys, zs    : cartesian coordinates of every cell (x towards the observer)
      area, parea   : area of one cell of every ring on the unit sphere, and its projected area on the disc
                      (parea = mu * area; the sum of parea over all the cells is ~pi)"""

    Nt=2*N-1 #N is number of concentric rings. Nt is counting them two times minus the center one.
    width=180.0/(2*N-1) #width of one grid element.

    centres=np.append(0,np.linspace(width,90-width/2,N-1)) #angular distance of every ring from the disc centre (the pole of the GRID faces the observer, not the rotation axis)
    anglesout=np.linspace(0,360-width,2*Nt) #longitudes of the grid edges of the most external grid. This grids fix the area of the grids in other rings.
    
    radi=np.sin(np.pi*centres/180) #projected polar radius of the ring.
    amu=np.cos(np.pi*centres/180) #amus

    ts=[0.0] #central grid radius
    alphas=[0.0] #central grid angle

    area=[2.0*np.pi*(1.0-np.cos(width*np.pi/360.0))] #area of spherical cap (only for the central element)
    parea=[np.pi*np.sin(width*np.pi/360.0)**2]

    Ngrid_in_ring=[1]
    
    for i in range(1,len(amu)): #for each ring except firs
        Nang=int(round(len(anglesout)*(radi[i]))) #Number of longitudes to have grids of same width
        w=360/Nang #width i angles
        Ngrid_in_ring.append(Nang)

        angles=np.linspace(0,360-w,Nang)
        area.append(radi[i]*width*w*np.pi*np.pi/(180*180)) #area of each grid
        parea.append(amu[i]*area[-1]) #PROJ. AREA OF THE GRID

        for j in range(Nang):
            ts.append(centres[i]) #colatitude in the frame of the grid (angular distance from the disc centre)
            alphas.append(angles[j]) #longitude in the frame of the grid (position angle around the disc centre)


    alphas=np.array(alphas) #position angle of every cell around the disc centre
    ts=np.array(ts) #angular distance of every cell from the disc centre
    Ngrids=len(ts)  #number of grids

    rs = np.sin(np.pi*ts/180) #projected polar radius of grid

    xs = np.cos(np.pi*ts/180) #grid elements in cartesian coordinates. Note that pole faces the observer.
    ys = rs*np.sin(np.pi*alphas/180)
    zs = -rs*np.cos(np.pi*alphas/180)

    return Ngrids,Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea



#######################################################################################################
# Filling factors with the spots computed BEFORE the faculae
#######################################################################################################

@nb.njit(cache=True,error_model='numpy')
def overlap_fraction(dist,rad,width,central):
    """Fraction of a grid cell covered by a circular region (spot or facula) on the sphere.
    dist    : angular distance between the centre of the cell and the centre of the region [rad]
    rad     : angular radius of the region [rad]
    width   : angular width of the cells [rad]
    central : True for the central cell (a disc), False for the other ones (squares of side `width`)
    The fraction is an approximation that only depends on dist: 0 beyond half+rad, 1 when the cell is inside the region,
    the area ratio when the region is inside the cell, and a linear ramp in between."""
    half=width/2.0
    if dist>half+rad:
        return 0.0
    if central:
        if half<rad: #the region can cover completely the cell
            if dist<=rad-half:
                return 1.0
            return -(dist-rad-half)/width
        if dist<=half-rad: #all the region is inside the cell
            return (2.0*rad/width)**2
        return -2.0*rad*(dist-half-rad)/width**2
    if width/m.sqrt(2.0)<rad:
        s=m.sqrt(rad**2-half**2)
        if dist<=s-half:
            return 1.0
        return -(dist-rad-half)/(width+rad-s)
    if half>rad:
        if dist<=half-rad:
            return (m.pi/4.0)*(2.0*rad/width)**2
        return (m.pi/4.0)*((2.0*rad/width)**2-(2.0*rad/width**2)*(dist-half+rad))
    s=m.sqrt(rad**2-half**2)
    A1=half*s
    A2=(rad**2/2.0)*(m.pi/2.0-2.0*m.asin(s/rad))
    Ar=4.0*(A1+A2)/width**2
    return -Ar*(dist-half-rad)/(half+rad)


@nb.njit(cache=True,error_model='numpy')
def generate_ff(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,vis, active_region_types, planet_exact=True):
    """Filling factors of every cell at one epoch (wrapper of generate_ff_core with the region types as strings).
    active_region_types : list of 'sp' / 'fc', one per region. One planet: planet_pos = [rho, theta, radius] and vis has
    n_regions + 1 values (the last one is the planet). See generate_ff_core for the other arguments and the outputs."""
    nreg=len(vis)-1
    region_type=np.zeros(nreg,dtype=np.int64)
    for l in range(nreg):
        if active_region_types[l]=='sp':
            region_type[l]=1
        elif active_region_types[l]=='fc':
            region_type[l]=2
    pp=np.empty((1,3))
    for q in range(3):
        pp[0,q]=planet_pos[q]
    return generate_ff_core(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,pp,vis,region_type,planet_exact)


#Gauss-Legendre nodes of the planet coverage (_planet_cell_area_nb)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


@nb.njit(cache=True)
def _arc_in_cell_nb(delta, beta, h):
    """Angle of the arc [delta - beta, delta + beta] (modulo 2 pi) inside the interval [-h, h]."""
    tot=0.0
    for k in (-1,0,1):
        lo=max(-h,delta+2.0*np.pi*k-beta)
        hi=min(h,delta+2.0*np.pi*k+beta)
        if hi>lo:
            tot+=hi-lo
    return tot


@nb.njit(cache=True)
def _half_arc_nb(r, rho, rp):
    """Half-angle of the arc of the circle of radius r (centred on the disc centre) inside the planet disc (centre at
    distance rho, radius rp)."""
    if rho<1e-12 or r<=rp-rho or r<=0.0:
        return np.pi if r<rp else 0.0
    cb=(r*r+rho*rho-rp*rp)/(2.0*r*rho)
    if cb>=1.0:
        return 0.0
    if cb<=-1.0:
        return np.pi
    return m.acos(cb)


@nb.njit(cache=True)
def _planet_cell_area_nb(r_in, r_out, alpha_c, h, rho, alpha_p, rp):
    """Exact projected area of a planet disc (centre at distance rho and position angle alpha_p from the disc centre,
    radius rp) inside a cell of the grid. On the sky a cell is an annular sector: r_in <= r <= r_out, |alpha - alpha_c|
    <= h (alpha = atan2(y, -z), as the position angles of the grid). Area = integral over r of r * (angle of the arc
    of radius r inside both the planet and the cell). The integral is split where the integrand has a kink (an end
    of the arc crosses an edge of the cell; the arc becomes a full circle) and every piece is integrated with
    Gauss-Legendre in t, r = a + (b - a)(1 - cos t)/2, which removes the square-root behaviour at the ends: exact to
    ~1e-6 of the cell. Validated against the pixels of a 4096 x 4096 image (pixel_to_cell_nb) and against the area of
    the planet on the disc (sum over all the cells)."""
    lo=max(r_in,rho-rp,0.0)
    hi=min(r_out,rho+rp)
    if hi<=lo:
        return 0.0
    delta=(alpha_p-alpha_c+np.pi)%(2.0*np.pi)-np.pi
    if rho>rp and abs(delta)>h+m.asin(min(1.0,rp/rho)): #outside the angles of the cell
        return 0.0
    #whole cell under the planet: every point (r, a) of the cell is within R of (r_mid, alpha_c), with
    #R^2 = ((r_out - r_in)/2)^2 + 2 r_out r_mid (1 - cos h) (law of cosines)
    r_mid=0.5*(r_in+r_out)
    R2=(0.5*(r_out-r_in))**2+2.0*r_out*r_mid*(1.0-m.cos(h))
    d2=r_mid*r_mid+rho*rho-2.0*r_mid*rho*m.cos(delta)
    if rp>m.sqrt(R2) and d2<=(rp-m.sqrt(R2))**2:
        return (r_out*r_out-r_in*r_in)*h
    br=np.empty(16) #breakpoints of the integral
    nbk=0
    br[0]=lo
    br[1]=hi
    nbk=2
    q=abs(rp-rho)
    if lo<q<hi:
        br[nbk]=q
        nbk+=1
    if rho>1e-12:
        for e in (-h,h):
            for k in (-1,0,1):
                beta=abs(e-(delta+2.0*np.pi*k)) #an end of the arc on the edge e: half-angle beta
                if beta<=0.0 or beta>=np.pi:
                    continue
                disc=rp*rp-rho*rho*m.sin(beta)**2 #r^2 - 2 r rho cos(beta) + rho^2 - rp^2 = 0
                if disc<0.0:
                    continue
                sq=m.sqrt(disc)
                for r in (rho*m.cos(beta)-sq,rho*m.cos(beta)+sq):
                    if lo<r<hi and nbk<16:
                        br[nbk]=r
                        nbk+=1
    pts=np.sort(br[:nbk])
    area=0.0
    for s in range(nbk-1):
        a=pts[s]
        b=pts[s+1]
        if b<=a:
            continue
        for g in range(_GL_X.shape[0]):
            t=(_GL_X[g]+1.0)*0.5*np.pi #t in (0, pi)
            r=a+(b-a)*0.5*(1.0-m.cos(t))
            dr=(b-a)*0.5*m.sin(t)*0.5*np.pi*_GL_W[g]
            area+=r*_arc_in_cell_nb(delta,_half_arc_nb(r,rho,rp),h)*dr
    return area


@nb.njit(cache=True)
def _planets_overlap_excess_nb(r_in, r_out, alpha_c, h, planet_pos, planet_on, n_side=1000):
    """Projected area of a cell (annular sector r_in <= r <= r_out, |alpha - alpha_c| <= h, alpha = atan2(y, -z)) that is
    covered by more than one planet, counted (k - 1) times where k planets cover it: the sum of the areas of the planets
    in the cell minus this excess is the area of their union. 0 when no two planets in front of the disc overlap each
    other. Computed on a grid of n_side x n_side points over the bounding box of the overlaps of the planet discs, so it
    costs nothing in the usual case (one planet, or planets apart); accuracy ~1e-3 of the area of the overlap (~2e-4
    with 20 rings), 0.2 s for two planets overlapping during a whole transit."""
    n_pl=planet_pos.shape[0]
    y0=1e9
    y1=-1e9
    z0=1e9
    z1=-1e9
    found=False
    for p in range(n_pl):
        if not planet_on[p]:
            continue
        yp=planet_pos[p,0]*m.cos(planet_pos[p,1])
        zp=planet_pos[p,0]*m.sin(planet_pos[p,1])
        rp=planet_pos[p,2]
        for q in range(p+1,n_pl):
            if not planet_on[q]:
                continue
            yq=planet_pos[q,0]*m.cos(planet_pos[q,1])
            zq=planet_pos[q,0]*m.sin(planet_pos[q,1])
            rq=planet_pos[q,2]
            if (yp-yq)**2+(zp-zq)**2>=(rp+rq)**2:
                continue
            #bounding box of the overlap of the two discs (the boxes of the two discs, intersected)
            a0=max(yp-rp,yq-rq)
            a1=min(yp+rp,yq+rq)
            b0=max(zp-rp,zq-rq)
            b1=min(zp+rp,zq+rq)
            if a1<=a0 or b1<=b0:
                continue
            y0=min(y0,a0)
            y1=max(y1,a1)
            z0=min(z0,b0)
            z1=max(z1,b1)
            found=True
    if not found:
        return 0.0
    dy=(y1-y0)/n_side
    dz=(z1-z0)/n_side
    #only the part of the box inside the bounding box of the cell (y = r sin(alpha), z = -r cos(alpha)), same steps
    if h<np.pi:
        cy0=1e9
        cy1=-1e9
        cz0=1e9
        cz1=-1e9
        for j in range(33):
            al=alpha_c-h+2.0*h*j/32.0
            for r in (r_in,r_out):
                cy0=min(cy0,r*m.sin(al))
                cy1=max(cy1,r*m.sin(al))
                cz0=min(cz0,-r*m.cos(al))
                cz1=max(cz1,-r*m.cos(al))
        pad=r_out*(1.0-m.cos(h/32.0))+max(dy,dz)          #the arcs between the sampled angles
        cy0-=pad
        cy1+=pad
        cz0-=pad
        cz1+=pad
    else:
        cy0=-r_out
        cy1=r_out
        cz0=-r_out
        cz1=r_out
    a0=max(0,int((cy0-y0)/dy))
    a1=min(n_side,int((cy1-y0)/dy)+1)
    b0=max(0,int((cz0-z0)/dz))
    b1=min(n_side,int((cz1-z0)/dz)+1)
    excess=0.0
    for a in range(a0,a1):
        y=y0+(a+0.5)*dy
        for b in range(b0,b1):
            z=z0+(b+0.5)*dz
            r2=y*y+z*z
            if r2<r_in*r_in or r2>r_out*r_out:
                continue
            if h<np.pi:
                d=(m.atan2(y,-z)-alpha_c+np.pi)%(2.0*np.pi)-np.pi
                if abs(d)>h:
                    continue
            k=0
            for p in range(n_pl):
                if planet_on[p]:
                    yp=planet_pos[p,0]*m.cos(planet_pos[p,1])
                    zp=planet_pos[p,0]*m.sin(planet_pos[p,1])
                    if (y-yp)**2+(z-zp)**2<=planet_pos[p,2]**2:
                        k+=1
            if k>1:
                excess+=k-1
    return excess*dy*dz


@nb.njit(cache=True,error_model='numpy')
def _planet_cover_nb(c,i,nang,mu,central,width,half,vec_grid,planet_pos,planet_on,exact):
    """Fraction of cell c (ring i, nang cells in the ring, projected angle mu) covered by the planets in front of the
    disc (planet_on).
    exact=True : projected area of the union of the planet discs inside the cell / projected area of the cell (annular
                 sector between the projected radii of the ring): the area of every planet (_planet_cell_area_nb), minus
                 the part covered by several planets where they overlap each other on the disc
                 (_planets_overlap_excess_nb), so that it is counted once. Right for any size of planet: a planet
                 smaller than a cell covers the fraction (planet area / cell area) of it.
    exact=False: the formula of the old StarSim, a linear ramp in the distance between the centre of the cell and the
                 centre of the planet (0 beyond radius + w/2, 1 within radius - w/2, w = projected radial width of the
                 cell). A cell whose centre is under the planet is about half covered whatever the radius of the
                 planet: planets smaller than a cell are much too deep (x8 for 0.01 Rstar with 15 rings)."""
    apl=0.0
    if exact:
        r_in=0.0 if central else m.sin((i-0.5)*width)
        r_out=min(1.0,m.sin((i+0.5)*width))
        h=np.pi/nang
        alpha_c=-1.0 #computed only for the cells near a planet
        n_touch=0
        cell_area=(r_out*r_out-r_in*r_in)*h
        for p in range(planet_pos.shape[0]):
            if not planet_on[p]:
                continue
            rho=planet_pos[p,0]
            if r_out<=rho-planet_pos[p,2] or r_in>=rho+planet_pos[p,2]: #the ring does not reach the planet
                continue
            if alpha_c<0.0:
                alpha_c=0.0 if central else m.atan2(vec_grid[c,1],-vec_grid[c,2])%(2.0*np.pi)
            yp=rho*m.cos(planet_pos[p,1]) #position of the planet on the sky (y, z)
            zp=rho*m.sin(planet_pos[p,1])
            apl+=_planet_cell_area_nb(r_in,r_out,alpha_c,h,rho,m.atan2(yp,-zp),planet_pos[p,2])/cell_area
            n_touch+=1
        if n_touch>1: #two planets or more in the cell: where they overlap each other, counted once
            apl-=_planets_overlap_excess_nb(r_in,r_out,alpha_c,h,planet_pos,planet_on)/cell_area
        return apl
    if central:
        width2=2.0*m.sin(half)
    else:
        width2=mu*width #projected width of the cell in the radial direction
    apl=0.0
    for p in range(planet_pos.shape[0]):
        if not planet_on[p]:
            continue
        dist=m.sqrt((planet_pos[p,0]*m.cos(planet_pos[p,1])-vec_grid[c,1])**2+(planet_pos[p,0]*m.sin(planet_pos[p,1])-vec_grid[c,2])**2) #grid-planet distance
        if dist>width2/2+planet_pos[p,2]:
            continue
        elif dist<planet_pos[p,2]-width2/2:
            apl+=1.0
        else:
            apl+=-(dist-planet_pos[p,2]-width2/2)/width2
    return apl


@nb.njit(cache=True,error_model='numpy')
def generate_ff_core(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,vis,region_type,planet_exact=True):
    """Filling factors of every cell at one epoch: fraction of the cell that is quiet photosphere, spot, facula and planet.
    N, Ngrid_in_ring, pare, amu : grid (generate_grid_coordinates_nb); pare = projected area of one cell of each ring
    spot_pos   : (n_regions, 3) colatitude, longitude and angular radius [rad] of every region
    vec_grid   : (n_cells, 3) cartesian coordinates of the cells
    vec_spot   : (n_regions, 3) cartesian coordinates of the centres of the regions
    simulate_planet : False -> the planets are ignored
    planet_pos : (n_planets, 3) rho, theta (polar coordinates on the disc: rho in stellar radii, theta in rad) and the
                 radius [Rstar] of every planet
    vis        : n_regions + n_planets values, 1.0 if the region / planet can be visible at this epoch (regions first)
    region_type: int array, 1 = spot, 2 = facula (numba-friendly version of the list of 'sp' / 'fc')
    planet_exact: True -> exact area of the planets in every cell, False -> ramp of the old StarSim (_planet_cover_nb)
    Returns ff_quiet, ff_sp, ff_fc, ff_planet (n_cells each; they add up to 1 in every cell) and the projected areas
    covered by each kind of surface (sums over the cells of ff * pare).

    Order of the regions in a cell (the result does not depend on the order in which the regions are given):
      1. spots: the fractions of all the spots are added;
      2. faculae, below the spots. The visible part of a facula depends on how its disc and the disc of each spot are
         placed (decided once per epoch, array `rel`):
            spot inside the facula   -> facula - spot            (exact)
            facula inside the spot   -> nothing                  (exact)
            discs do not overlap     -> facula                   (exact)
            partial overlap          -> facula*(1-spot)          (approximation: spot and facula independent in the cell)
      3. planets, on top of everything: the fraction covered by the planets (at most 1; exact projected area of the
         union of the planet discs in the cell, _planet_cover_nb) is removed from the spot and facula fractions. The
         planets are dark discs in the plane of the sky; where two planets overlap each other, the shared part is
         counted once (with planet_exact=False, the old ramp, it is counted twice)."""
    ncell=vec_grid.shape[0]
    npl=planet_pos.shape[0]
    nreg=len(vis)-npl
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    half=width/2.0

    #visible regions with radius>0, spots and faculae apart
    sidx=np.empty(nreg,dtype=np.int64)
    fidx=np.empty(nreg,dtype=np.int64)
    ns=0
    nf=0
    for l in range(nreg):
        if vis[l]==1.0 and spot_pos[l][2]>0.0:
            if region_type[l]==1:
                sidx[ns]=l
                ns+=1
            elif region_type[l]==2:
                fidx[nf]=l
                nf+=1

    #cosine of the reach of every region: cells further away are not touched (and need no acos)
    cos_reach=np.empty(nreg)
    for l in range(nreg):
        reach=half+spot_pos[l][2]
        if reach<m.pi:
            cos_reach[l]=m.cos(reach)
        else:
            cos_reach[l]=-1.0

    #relation between every facula b and every spot a: 0 do not overlap, 1 spot inside facula, 2 facula inside spot, 3 partial
    rel=np.zeros((nf,ns),dtype=np.int64)
    for b in range(nf):
        lf=fidx[b]
        for a in range(ns):
            ls=sidx[a]
            #angular distance between the two centres; atan2 is exact for identical positions (acos is not: acos(1-1e-16)=1.5e-8)
            cx=vec_spot[lf][1]*vec_spot[ls][2]-vec_spot[lf][2]*vec_spot[ls][1]
            cy=vec_spot[lf][2]*vec_spot[ls][0]-vec_spot[lf][0]*vec_spot[ls][2]
            cz=vec_spot[lf][0]*vec_spot[ls][1]-vec_spot[lf][1]*vec_spot[ls][0]
            d=m.atan2(m.sqrt(cx*cx+cy*cy+cz*cz),np.dot(vec_spot[lf],vec_spot[ls]))
            eps=1e-9 #radians
            if d+spot_pos[ls][2]<=spot_pos[lf][2]+eps:
                rel[b,a]=1
            elif d+spot_pos[lf][2]<=spot_pos[ls][2]+eps:
                rel[b,a]=2
            elif d>=spot_pos[ls][2]+spot_pos[lf][2]:
                rel[b,a]=0
            else:
                rel[b,a]=3

    planet_on=np.zeros(npl,dtype=np.bool_) #planets in front of the disc
    any_planet=False
    if simulate_planet:
        for p in range(npl):
            if vis[nreg+p]==1.0:
                planet_on[p]=True
                any_planet=True

    ds=np.zeros(ns) #coverage of the cell by each spot
    df=np.zeros(nf) #and by each facula
    ff_quiet=np.empty(ncell)
    ff_sp=np.empty(ncell)
    ff_fc=np.empty(ncell)
    ff_p=np.empty(ncell)
    Aph=0.0
    Asp=0.0
    Afc=0.0
    Apl=0.0

    c=0
    for i in range(N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            central=(c==0)
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            #SPOTS first
            for a in range(ns):
                ds[a]=0.0
                l=sidx[a]
                dot=vec_grid[c,0]*vec_spot[l,0]+vec_grid[c,1]*vec_spot[l,1]+vec_grid[c,2]*vec_spot[l,2]
                if dot<cos_reach[l]: #grid not covered
                    continue
                dist=m.acos(min(1.0,max(-1.0,dot)))
                ds[a]=overlap_fraction(dist,spot_pos[l][2],width,central)
                asp+=ds[a]

            #FACULAE afterwards, the spots are on top of them
            for b in range(nf):
                df[b]=0.0
                l=fidx[b]
                dot=vec_grid[c,0]*vec_spot[l,0]+vec_grid[c,1]*vec_spot[l,1]+vec_grid[c,2]*vec_spot[l,2]
                if dot<cos_reach[l]:
                    continue
                dist=m.acos(min(1.0,max(-1.0,dot)))
                df[b]=overlap_fraction(dist,spot_pos[l][2],width,central)
            for b in range(nf):
                if df[b]<=0.0:
                    continue
                sub=0.0
                mult=1.0
                hidden=False
                for a in range(ns):
                    if ds[a]<=0.0:
                        continue
                    if rel[b,a]==1:
                        sub+=ds[a]
                    elif rel[b,a]==2:
                        hidden=True
                    elif rel[b,a]==3:
                        mult*=(1.0-min(ds[a],1.0))
                if not hidden:
                    v=df[b]-sub
                    if v>0.0:
                        afc+=v*mult

            #PLANETS
            if any_planet:
                apl=_planet_cover_nb(c,i,Ngrid_in_ring[i],amu[i],central,width,half,vec_grid,planet_pos,planet_on,planet_exact)

            if afc<0:
                afc=0.0
            if afc>1.0:
                afc=1.0
            if asp>1.0:
                asp=1.0
                afc=0.0
            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0
            if afc+asp>1.0:
                afc=1.0-asp
            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl
            ff_quiet[c]=aph
            ff_sp[c]=asp
            ff_fc[c]=afc
            ff_p[c]=apl
            Aph+=aph*pare[i]
            Asp+=asp*pare[i]
            Afc+=afc*pare[i]
            Apl+=apl*pare[i]
            c+=1

    return ff_quiet, ff_sp, ff_fc, ff_p, Aph, Asp, Afc, Apl


#######################################################################################################
# CCF of every cell: the ring CCF placed on its velocity axis, shifted by the rotation velocity of the cell
# (kernels copied from the old nbspectra)
#######################################################################################################

@nb.njit(cache=True,error_model='numpy')
def interpolation_nb(xp,x,y,left=0,right=0):
    """Linear interpolation of y(x) at the points xp (x ascending); `left` / `right` outside [x[0], x[-1]].
    Every search starts from the previous index, so it is fast when xp is sorted."""

    # Create result array
    yp=np.zeros(len(xp))
    minx=x[0]
    maxx=x[-1]
    lastidx=1 

    for i,xi in enumerate(xp):
        if xi<minx: #extrapolate left
            yp[i]=left
        elif xi>maxx: #extrapolate right
            yp[i]=right
        else:
            for j in range(lastidx,len(x)): #per no fer el loop sobre tota la x, ja que esta sorted sempre comenso amb lanterior.
                if x[j]>xi:
                    #Trobo el primer x mes gran que xj. llavors utilitzo x[j] i x[j-1] per interpolar
                    yp[i]=y[j-1]+(xi-x[j-1])*(y[j]-y[j-1])/(x[j]-x[j-1])
                    lastidx=j
                    break
                elif x[j] == xi:
                    yp[i] = y[j]
                    break
    return yp


@nb.njit(cache=True,error_model='numpy')
def _interp_shifted_nb(xp,x,shift,y,left,right,out):
    """out = interpolation_nb(xp, x + shift, y, left, right), with exactly the same operations, written in the buffer
    `out` (no array allocated) and with 0.0 + value, as the cells of loop_compute_immaculate_nb (accumulated on 0)."""
    minx=x[0]+shift
    maxx=x[-1]+shift
    lastidx=1
    for i in range(len(xp)):
        xi=xp[i]
        v=0.0
        if xi<minx:
            v=left
        elif xi>maxx:
            v=right
        else:
            for j in range(lastidx,len(x)):
                xj=x[j]+shift
                if xj>xi:
                    xj1=x[j-1]+shift
                    v=y[j-1]+(xi-xj1)*(y[j]-y[j-1])/(xj-xj1)
                    lastidx=j
                    break
                elif xj==xi:
                    v=y[j]
                    break
        out[i]=0.0+v


@nb.njit(cache=True,error_model='numpy')
def _quiet_ccf_total_nb(N,Ngrid_in_ring,rvel,rv,rvs_ring,ccf_ring):
    """CCF of the whole quiet disc: sum over the cells (in their order) of the CCF of their ring shifted by their
    rotation velocity (loop_compute_immaculate_nb), without storing the CCF of every cell."""
    total=np.zeros(len(rv))
    buf=np.empty(len(rv))
    c=0
    for i in range(N):
        for j in range(Ngrid_in_ring[i]):
            _interp_shifted_nb(rv,rvs_ring[i,:],rvel[c],ccf_ring[i,:],ccf_ring[i,0],ccf_ring[i,-1],buf)
            for w in range(len(rv)):
                total[w]+=buf[w]
            c+=1
    return total


@nb.njit(cache=True,error_model='numpy',parallel=True)
def loop_compute_immaculate_nb(N,Ngrid_in_ring,ccf_tot,rvel,rv,rvs_ring,ccf_ring):
    """CCF of every cell: the CCF of its ring (ccf_ring, already weighted by the cell area) placed on the velocity
    axis of the ring (rvs_ring) shifted by the rotation velocity of the cell (rvel), and interpolated on the common
    velocity grid rv (the end values are kept outside). Accumulated in ccf_tot (n_cells, len(rv)), which is returned.
    The cells are done in parallel (each one only writes its own row)."""
    ring_of_cell=np.empty(ccf_tot.shape[0],dtype=np.int64)
    c=0
    for i in range(N):
        for j in range(Ngrid_in_ring[i]):
            ring_of_cell[c]=i
            c+=1
    for c in prange(ccf_tot.shape[0]):
        i=ring_of_cell[c]
        ccf_tot[c,:]+=interpolation_nb(rv,rvs_ring[i,:]+rvel[c],ccf_ring[i,:],ccf_ring[i,0],ccf_ring[i,-1])
    return ccf_tot


@nb.njit(cache=True,error_model='numpy',parallel=True)
def cross_correlation_mask(rv,wv,f,wvm,fm, phoenix_resolution, ccf_norm):
    """
    Cross-correlation of the spectrum (wv, f) with a mask of lines (wvm, weights fm) at the velocities rv [m/s].
    For every velocity the mask is Doppler shifted, wvm * (1 + rv/c), and every line takes the flux of the spectrum
    over a box of one pixel centred on it (split between the two pixels it overlaps, with linear interpolation).
    The CCF is minus the weighted sum, so it is a peak at the velocity of the lines.
    phoenix_resolution : True -> the pixel of every line is found directly from the known steps of the Phoenix
                         wavelength grid (0.1 A below 3000 A, 0.006 A up to 5000 A, 0.01 A up to 10000 A, 0.02 A up
                         to 15000 A, 0.03 A above): only for spectra on the Phoenix grid. False -> search in wv (any grid).
    ccf_norm           : True -> (ccf - min) / max(ccf - min), between 0 and 1;
                         False -> -ccf / min(ccf) + 1;
                         2 -> no normalisation: the raw CCF, minus the weighted sum of the fluxes (in the units of the
                         spectrum: proportional to it, so the CCFs of several spectra can be added as the spectra).
    The first two normalisations do not change when the spectrum is multiplied by a constant.
    The velocities are done in parallel (every value of the CCF is independent).
    """
    ccf = np.zeros(len(rv))
    lenm = len(wvm)
    wvmin=wv[0]

    if phoenix_resolution:

        for i in prange(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #mask lines Doppler shifted by rv[i] [m/s]
            #for each mask line
            for j in range(lenm):
                #find wavelengths right and left of the line.
                wvline=wvshift[j]

                if wvline<3000.0:
                    idxlf = int((wvline-wvmin)/0.1)

                elif wvline<4999.986:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((wvline-3000.0)/0.006) #pixels of the 0.1 A part + pixels of the 0.006 A part
                    else:
                        idxlf = int((wvline-wvmin)/0.006)

                elif wvline<5000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1
                    else:
                        idxlf = int((4999.986-wvmin)/0.006) + 1

                elif wvline<10000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((wvline-5000.0)/0.01)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((wvline-5000.0)/0.01)
                    else:
                        idxlf = int((wvline-wvmin)/0.01) 

                elif wvline<15000.0:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                    elif wvmin<10000.0:
                        idxlf = int((10000.0-wvmin)/0.01) + int((wvline-10000.0)/0.02)
                    else:
                        idxlf = int((wvline-wvmin)/0.02)

                else:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((15000.0-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<4999.986:
                        idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<10000.0:
                        idxlf = int((10000.0-wvmin)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                    elif wvmin<15000.0:
                        idxlf = int((15000-wvmin)/0.02) + int((wvline-15000.0)/0.03)
                    else:
                        idxlf = int((wvline-wvmin)/0.03)

                idxrg = idxlf + 1

                diffwv=wv[idxrg]-wv[idxlf] #pixel size in wavelength
                midpix=(wv[idxrg]+wv[idxlf])/2 #wavelength between the two pixels
                leftmask = wvline - diffwv/2 #left edge of the mask
                rightmask = wvline + diffwv/2 #right edge of the mask
                frac1 = (midpix - leftmask)/diffwv #fraction of the mask ovelapping the left pixel
                frac2 = (rightmask - midpix)/diffwv #fraction of the mask overlapping the right pixel
                midleft = (leftmask + midpix)/2 #central left overlapp
                midright = (rightmask + midpix)/2 #central wv right overlap
                f1 = f[idxlf] + (midleft-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
                f2 = f[idxlf] + (midright-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)

                ccf[i]=ccf[i] - f1*fm[j]*frac1 - f2*fm[j]*frac2
                


    else:
        for i in prange(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #mask lines Doppler shifted by rv[i] [m/s]
            #for each mask line
            for j in range(lenm):
                wvline=wvshift[j]
                
                if wvline > wv[-1]:
                    pass
                else:
                    #first pixel to the right of the line (wv[k] > wvline; k >= 1 if wvline >= wv[0]): binary search
                    #instead of a linear search from the start of the spectrum, same pixel
                    k = np.searchsorted(wv, wvline, side='right')

                    idxlf = int(k-1)

                    idxrg = idxlf + 1

                    diffwv=wv[idxrg]-wv[idxlf] #pixel size in wavelength
                    midpix=(wv[idxrg]+wv[idxlf])/2 #wavelength between the two pixels
                    leftmask = wvline - diffwv/2 #left edge of the mask
                    rightmask = wvline + diffwv/2 #right edge of the mask
                    frac1 = (midpix - leftmask)/diffwv #fraction of the mask ovelapping the left pixel
                    frac2 = (rightmask - midpix)/diffwv #fraction of the mask overlapping the right pixel
                    midleft = (leftmask + midpix)/2 #central left overlapp
                    midright = (rightmask + midpix)/2 #central wv right overlap
                    f1 = f[idxlf] + (midleft-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
                    f2 = f[idxlf] + (midright-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
                    # print(ccf[i], f1, fm[j], frac1, f2, frac2)
                    ccf[i]=ccf[i] - f1*fm[j]*frac1 - f2*fm[j]*frac2
                    
    if ccf_norm == 2: #no normalisation
        return ccf
    if ccf_norm:
        return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))
    else:
        return - ccf / np.min(ccf) + 1 #ccf < 0, most negative in the continuum: ~0 in the wings, relative line depth at the line velocity


@nb.njit(cache=True,error_model='numpy')
def speed_bisector_nb(rv,ccf,integrated_bis):
    '''Bisector of a CCF (a peak: maximum at the line centre), normalised to its maximum.
    The wings are cut where the CCF starts to rise again on each side of the maximum, below 80% of the maximum
    (cutleft, cutright). The bisector is computed at 50 heights ybis, from 10% of the depth above the higher of the two
    wing minima up to 99% (integrated_bis=True) or 99.9% of the maximum; at each height, xbis is the mid-point of the
    two velocities where the CCF crosses it (linear interpolation).
    Returns cutleft, cutright, xbis (50 velocities), ybis (50 heights).
    '''
    idxmax=ccf.argmax()
    maxccf=ccf[idxmax]
    maxrv=rv[idxmax]

    xnew = rv
    ynew = ccf


    cutleft=0
    cutright=len(ynew)-1
    # if not integrated_bis: #cut the CCF at the minimum of the wings only for reference CCF, if not there are errors.
    for i in range(len(ynew)):
        if xnew[i]>maxrv:
            if ynew[i]>ynew[i-1] and ynew[i]<0.8*maxccf:
                cutright=i
                break

    for i in range(len(ynew)):
        if xnew[-1-i]<maxrv:
            #the point examined (-1-i) must rise again towards the wing and be below 80% of the maximum
            #(the same test as the right wing; it tested the mirror point ynew[i] before)
            if ynew[-1-i]>ynew[-i] and ynew[-1-i]<0.8*maxccf:
                cutleft=len(ynew)-i
                break

    #keep the CCF between the wings

    xnew=xnew[cutleft:cutright]
    ynew=ynew[cutleft:cutright]
    
    minright=np.min(ynew[xnew>maxrv])
    minleft=np.min(ynew[xnew<maxrv])
    minccf=np.max(np.array([minright,minleft]))
    
    if integrated_bis:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.99*maxccf,50) #from 10% of the depth above the wings to 99% of the maximum
    else:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.999*maxccf,50) #from 10% of the depth above the wings to 99.9% of the maximum
    xbis=np.zeros(len(ybis))


    for i in range(len(ybis)):
        for j in range(len(ynew)-1):
            if ynew[j]<ybis[i] and ynew[j+1]>ybis[i] and xnew[j]<maxrv:
                rv1=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
            if ynew[j]>ybis[i] and ynew[j+1]<ybis[i] and xnew[j+1]>maxrv:
                rv2=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
        xbis[i]=(rv1+rv2)/2.0 #bisector
    # xbis[-1]=maxrv #at the top should be max RV

    return cutleft,cutright,xbis,ybis


@nb.njit(cache=True)
def polyval_mapped_nb(x, off, scl, coef):
    """Value of the polynomial sum_k coef[k] t^k at t = off + scl*x (numpy Polynomial with its domain mapped to its
    window: Polynomial(...)(x)), in one pass over x, with the same operations as numpy (Horner) for every point."""
    out = np.empty(len(x))
    n = len(coef)
    for i in range(len(x)):
        t = off + scl*x[i]
        v = coef[n-1]
        for k in range(n-2, -1, -1):
            v = coef[k] + v*t
        out[i] = v
    return out


@nb.njit(cache=True,error_model='numpy')
def gaussian2(x, amplitude, mean, stddev,C):
    """Gaussian with an offset: C + amplitude * exp(-(x-mean)^2 / (2 stddev^2)). Model of the CCF fits."""
    return C + amplitude * np.exp(-(x-mean)**2/(2*stddev**2))



#######################################################################################################
# Time series: the whole loop over the epochs in one call
#######################################################################################################

@nb.njit(cache=True)
def true_anomaly_scalar_nb(x,period,ecc,tperi):
    """sin and cos of the true anomaly f at time x, for an orbit of period `period`, eccentricity ecc and time of
    periastron tperi. The Kepler equation E - e sin E = M is solved with Newton's method (tolerance 1e-6 rad)."""
    fmean=2.0*np.pi*(x-tperi)/period
    fecc=fmean
    diff=1.0
    while(diff>1.0E-6):
        fecc_0=fecc
        fecc=fecc_0-(fecc_0-ecc*m.sin(fecc_0)-fmean)/(1.0-ecc*m.cos(fecc_0))
        diff=abs(fecc-fecc_0)
    sinf=m.sqrt(1.0-ecc*ecc)*m.sin(fecc)/(1.0-ecc*m.cos(fecc))
    cosf=(m.cos(fecc)-ecc)/(1.0-ecc*m.cos(fecc))
    return sinf,cosf


@nb.njit(cache=True)
def true_anomaly_nb(x,period,ecc,tperi):
    """true_anomaly_scalar_nb at every time of the array x: returns the arrays sin(f) and cos(f)."""
    sinf=np.empty(len(x))
    cosf=np.empty(len(x))
    for i in range(len(x)):
        sinf[i],cosf[i]=true_anomaly_scalar_nb(x[i],period,ecc,tperi)
    return sinf,cosf


@nb.njit(cache=True)
def ttrans_2_tperi_nb(T0,P,e,w):
    """Time of periastron from the time of (primary) transit T0, period P, eccentricity e and argument of periastron w.
    At the transit the true anomaly is f = pi/2 - w."""
    f=np.pi/2-w
    E=2*m.atan(m.tan(f/2)*m.sqrt((1-e)/(1+e))) #eccentric anomaly
    return T0-P/(2*np.pi)*(E-e*m.sin(E)) #time of periastron


@nb.njit(cache=True)
def _cell_ring_nb(N,Ngrid_in_ring,pare):
    """Ring index of every cell, and the total projected area of the disc."""
    ncell=0
    for i in range(N):
        ncell+=Ngrid_in_ring[i]
    cell_ring=np.empty(ncell,dtype=np.int64)
    area_tot=0.0
    c=0
    for i in range(N):
        area_tot+=Ngrid_in_ring[i]*pare[i]
        for j in range(Ngrid_in_ring[i]):
            cell_ring[c]=i
            c+=1
    return cell_ring,area_tot


@nb.njit(cache=True)
def _add_signals_nb(k,ff_quiet,ff_sp,ff_fc,cell_ring,gq_ph,gs_ph,gf_ph,quiet_ph,gq_sp,gs_sp,gf_sp,quiet_sp,sp_per_ring,
                    gq_cc,gs_cc,gf_cc,quiet_cc,flux,spec,ccf):
    """Signals of epoch k (written in flux[k], spec[k], ccf[k]) from the filling factors of the cells:
    signal = quiet_total + sum over the cells that are not entirely quiet of (ff_quiet-1)*grid_quiet + ff_sp*grid_sp +
    ff_fc*grid_fc. Photometry grids are per ring (cell_ring gives the ring of every cell); spectroscopy grids per ring
    (sp_per_ring: the fractions of the cells are first added per ring) or per cell; a signal with an empty grid
    is skipped (see timeseries_nb)."""
    do_ph=len(gq_ph)>0
    nsp=gq_sp.shape[1]
    ncc=gq_cc.shape[1]
    nr=gq_sp.shape[0] if (sp_per_ring and nsp>0) else 0
    rq=np.zeros(nr)
    rs=np.zeros(nr)
    rf=np.zeros(nr)
    if do_ph:
        flux[k]=quiet_ph[0]
    for w in range(nsp):
        spec[k,w]=quiet_sp[w]
    for w in range(ncc):
        ccf[k,w]=quiet_cc[w]
    for q in range(len(ff_quiet)):
        dq=ff_quiet[q]-1.0
        fs=ff_sp[q]
        ff=ff_fc[q]
        if dq==0.0 and fs==0.0 and ff==0.0: #untouched cell: quiet, already in the total
            continue
        if do_ph:
            r=cell_ring[q]
            flux[k]+=dq*gq_ph[r]+fs*gs_ph[r]+ff*gf_ph[r]
        if nr>0:
            r=cell_ring[q]
            rq[r]+=dq
            rs[r]+=fs
            rf[r]+=ff
        else:
            for w in range(nsp):
                spec[k,w]+=dq*gq_sp[q,w]+fs*gs_sp[q,w]+ff*gf_sp[q,w]
        for w in range(ncc):
            ccf[k,w]+=dq*gq_cc[q,w]+fs*gs_cc[q,w]+ff*gf_cc[q,w]
    _add_rings_nb(k,rq,rs,rf,gq_sp,gs_sp,gf_sp,spec)


@nb.njit(cache=True)
def _add_rings_nb(k,rq,rs,rf,gq_sp,gs_sp,gf_sp,spec):
    """Spectroscopy grids per ring: spec[k] += sum over the rings of rq*grid_quiet + rs*grid_sp + rf*grid_fc, where
    rq, rs, rf are the sums over the touched cells of each ring of (ff_quiet-1), ff_sp and ff_fc."""
    for r in range(len(rq)):
        a=rq[r]
        b=rs[r]
        c=rf[r]
        if a==0.0 and b==0.0 and c==0.0:
            continue
        for w in range(gq_sp.shape[1]):
            spec[k,w]+=a*gq_sp[r,w]+b*gs_sp[r,w]+c*gf_sp[r,w]


@nb.njit(cache=True)
def _grow_touched_nb(t_cell,t_frac,cap):
    """Touched-cell arrays of timeseries_nb copied into arrays of capacity cap."""
    c2=np.empty(cap,dtype=np.int64)
    f2=np.empty((cap,3))
    n=len(t_cell)
    c2[:n]=t_cell
    f2[:n]=t_frac
    return c2,f2


@nb.njit(cache=True,error_model='numpy')
def _touched_cells_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,vis,region_type,planet_exact):
    """generate_ff_core, but only the cells touched by a region or a planet are returned: t_cell (their index), t_frac
    (ff_quiet - 1, ff_sp, ff_fc), nt of them in the order of the cells (arrays that grow when needed), and the projected
    areas covered by each kind of surface (Aph, Asp, Afc, Apl). Same operations as generate_ff_core (the loop over the
    cells is written out: a function per cell compiles ~3x slower), and memory that does not grow with the cells."""
    ncell=vec_grid.shape[0]
    npl=planet_pos.shape[0]
    nreg=len(vis)-npl
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    half=width/2.0

    #visible regions with radius>0, spots and faculae apart
    sidx=np.empty(nreg,dtype=np.int64)
    fidx=np.empty(nreg,dtype=np.int64)
    ns=0
    nf=0
    for l in range(nreg):
        if vis[l]==1.0 and spot_pos[l][2]>0.0:
            if region_type[l]==1:
                sidx[ns]=l
                ns+=1
            elif region_type[l]==2:
                fidx[nf]=l
                nf+=1

    #cosine of the reach of every region: cells further away are not touched (and need no acos)
    cos_reach=np.empty(nreg)
    for l in range(nreg):
        reach=half+spot_pos[l][2]
        if reach<m.pi:
            cos_reach[l]=m.cos(reach)
        else:
            cos_reach[l]=-1.0

    #relation between every facula b and every spot a: 0 do not overlap, 1 spot inside facula, 2 facula inside spot, 3 partial
    rel=np.zeros((nf,ns),dtype=np.int64)
    for b in range(nf):
        lf=fidx[b]
        for a in range(ns):
            ls=sidx[a]
            #angular distance between the two centres; atan2 is exact for identical positions (acos is not: acos(1-1e-16)=1.5e-8)
            cx=vec_spot[lf][1]*vec_spot[ls][2]-vec_spot[lf][2]*vec_spot[ls][1]
            cy=vec_spot[lf][2]*vec_spot[ls][0]-vec_spot[lf][0]*vec_spot[ls][2]
            cz=vec_spot[lf][0]*vec_spot[ls][1]-vec_spot[lf][1]*vec_spot[ls][0]
            d=m.atan2(m.sqrt(cx*cx+cy*cy+cz*cz),np.dot(vec_spot[lf],vec_spot[ls]))
            eps=1e-9 #radians
            if d+spot_pos[ls][2]<=spot_pos[lf][2]+eps:
                rel[b,a]=1
            elif d+spot_pos[lf][2]<=spot_pos[ls][2]+eps:
                rel[b,a]=2
            elif d>=spot_pos[ls][2]+spot_pos[lf][2]:
                rel[b,a]=0
            else:
                rel[b,a]=3

    planet_on=np.zeros(npl,dtype=np.bool_) #planets in front of the disc
    any_planet=False
    if simulate_planet:
        for p in range(npl):
            if vis[nreg+p]==1.0:
                planet_on[p]=True
                any_planet=True

    ds=np.zeros(ns) #coverage of the cell by each spot
    df=np.zeros(nf) #and by each facula
    cap=256
    t_cell=np.empty(cap,dtype=np.int64)
    t_frac=np.empty((cap,3)) #dq = ff_quiet - 1, ff_sp, ff_fc
    nt=0
    Aph=0.0
    Asp=0.0
    Afc=0.0
    Apl=0.0

    c=0
    for i in range(N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            central=(c==0)
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            #SPOTS first
            for a in range(ns):
                ds[a]=0.0
                l=sidx[a]
                dot=vec_grid[c,0]*vec_spot[l,0]+vec_grid[c,1]*vec_spot[l,1]+vec_grid[c,2]*vec_spot[l,2]
                if dot<cos_reach[l]: #grid not covered
                    continue
                dist=m.acos(min(1.0,max(-1.0,dot)))
                ds[a]=overlap_fraction(dist,spot_pos[l][2],width,central)
                asp+=ds[a]

            #FACULAE afterwards, the spots are on top of them
            for b in range(nf):
                df[b]=0.0
                l=fidx[b]
                dot=vec_grid[c,0]*vec_spot[l,0]+vec_grid[c,1]*vec_spot[l,1]+vec_grid[c,2]*vec_spot[l,2]
                if dot<cos_reach[l]:
                    continue
                dist=m.acos(min(1.0,max(-1.0,dot)))
                df[b]=overlap_fraction(dist,spot_pos[l][2],width,central)
            for b in range(nf):
                if df[b]<=0.0:
                    continue
                sub=0.0
                mult=1.0
                hidden=False
                for a in range(ns):
                    if ds[a]<=0.0:
                        continue
                    if rel[b,a]==1:
                        sub+=ds[a]
                    elif rel[b,a]==2:
                        hidden=True
                    elif rel[b,a]==3:
                        mult*=(1.0-min(ds[a],1.0))
                if not hidden:
                    v=df[b]-sub
                    if v>0.0:
                        afc+=v*mult

            #PLANETS
            if any_planet:
                apl=_planet_cover_nb(c,i,Ngrid_in_ring[i],amu[i],central,width,half,vec_grid,planet_pos,planet_on,planet_exact)

            if afc<0:
                afc=0.0
            if afc>1.0:
                afc=1.0
            if asp>1.0:
                asp=1.0
                afc=0.0
            if apl>1.0:
                apl=1.0
                asp=0.0
                afc=0.0
            if afc+asp>1.0:
                afc=1.0-asp
            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl
            Aph+=aph*pare[i]
            Asp+=asp*pare[i]
            Afc+=afc*pare[i]
            Apl+=apl*pare[i]
            dq=aph-1.0
            if not (dq==0.0 and asp==0.0 and afc==0.0): #touched cell (the others are quiet, already in the total)
                if nt==cap:
                    cap*=2
                    t_cell,t_frac=_grow_touched_nb(t_cell,t_frac,cap)
                t_cell[nt]=c
                t_frac[nt,0]=dq
                t_frac[nt,1]=asp
                t_frac[nt,2]=afc
                nt+=1
            c+=1

    return t_cell,t_frac,nt,Aph,Asp,Afc,Apl


@nb.njit(cache=True,parallel=True)
def timeseries_nb(N,Ngrid_in_ring,pare,amu,vec_grid,spot_pos_all,vec_spot_all,vis_all,region_type,simulate_planet,planet_pos_all,
                  gq_ph,gs_ph,gf_ph,quiet_ph,gq_sp,gs_sp,gf_sp,quiet_sp,sp_per_ring,gq_cc,gs_cc,gf_cc,quiet_cc,
                  cc_on_the_fly,rv_cc,rvel,rvs_ring_cc,ccf_ring_cc,planet_exact,
                  sp_on_the_fly,wv_sp,new_wv_sp,rings_sp,vrel_sp):
    """Signals of all the epochs in one call (parallel over the epochs).
    The geometry is computed before, vectorised, for all the epochs (StarSim.generate_timeseries):
       spot_pos_all, vec_spot_all (n_times, n_regions, 3), vis_all (n_times, n_regions+n_planets; regions first),
       planet_pos_all (n_times, n_planets, 3); region_type: 1 = spot, 2 = facula.
    For every epoch the filling factors of every cell are computed (generate_ff_core) and the signals are
       signal = quiet_total + sum over the cells touched by a region or a planet of
                (ff_quiet-1)*grid_quiet + ff_sp*grid_sp + ff_fc*grid_fc
    which is equal to the sum over all the cells of ff_quiet*grid_quiet + ff_sp*grid_sp + ff_fc*grid_fc (the cells that
    are not touched are quiet), but only loops over the few touched cells.
    Grids gq_*, gs_*, gf_* (quiet / spot / facula) and quiet_* (signal of the quiet star), for
       photometry   (ph): one value per RING (N,); the cells are summed with the index of their ring
       spectroscopy (sp): one spectrum per RING (N, n_wavelengths) if sp_per_ring (no rotation: the fractions of the
                          touched cells are added per ring, then one spectrum per touched ring), otherwise one
                          spectrum per cell (n_cells, n_wavelengths)
       ccf          (cc): one CCF per cell (n_cells, n_rv)
    A signal with an empty grid (0 rings / 0 columns) is not computed.
    cc_on_the_fly: the CCF grids of the cells are not given (gq_cc, gs_cc, gf_cc empty): the CCF of every touched cell
    is computed when it is needed, from the CCF of its ring (ccf_ring_cc[s], already weighted by the cell area) on the
    velocity axis of the ring (rvs_ring_cc[s]) shifted by the rotation velocity of the cell (rvel), interpolated on
    rv_cc (_interp_shifted_nb: the same values as the grids of loop_compute_immaculate_nb); s = 0, 1, 2 for the quiet
    photosphere, the spots and the faculae. quiet_cc is then the total of _quiet_ccf_total_nb.
    sp_on_the_fly: the same for the spectroscopy with rotation: the grids of the cells (gq_sp, gs_sp, gf_sp) are not given;
    the spectrum of every touched cell is the spectrum of its ring (rings_sp[s], on the wavelengths wv_sp) Doppler shifted
    by the velocity of the cell (vrel_sp) on new_wv_sp (_interp_doppler_into_nb: the values of doppler_shift_grid_nb);
    quiet_sp is then the total of doppler_disc_total_nb.
    Returns flux (n_times,), spec (n_times, n_wv), ccf (n_times, n_rv), and the filling factors [% of the disc]
    (4, n_times): quiet, spots, faculae, planets."""
    n_times=vis_all.shape[0]
    ncell=vec_grid.shape[0]
    do_ph=len(gq_ph)>0
    nsp=len(new_wv_sp) if sp_on_the_fly else gq_sp.shape[1]
    ncc=len(rv_cc) if cc_on_the_fly else gq_cc.shape[1]
    nr=N if (sp_per_ring and nsp>0 and not sp_on_the_fly) else 0
    c_light=2.99792458e8

    cell_ring,area_tot=_cell_ring_nb(N,Ngrid_in_ring,pare)

    flux=np.zeros(n_times)
    spec=np.zeros((n_times,nsp))
    ccf=np.zeros((n_times,ncc))
    filling=np.zeros((4,n_times))

    for k in prange(n_times):
        vis=vis_all[k]
        spot_pos=spot_pos_all[k]
        vec_spot=vec_spot_all[k]
        planet_pos=planet_pos_all[k]

        if np.sum(vis)==0.0:
            #nothing visible: the star is the quiet photosphere
            filling[0,k]=100.0
            if do_ph:
                flux[k]=quiet_ph[0]
            for w in range(nsp):
                spec[k,w]=quiet_sp[w]
            for w in range(ncc):
                ccf[k,w]=quiet_cc[w]
            continue

        #1. filling factors of every cell (generate_ff_core), only the touched cells are kept: no array of n_cells per epoch
        t_cell,t_frac,nt,Aph,Asp,Afc,Apl=_touched_cells_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,
                                                            simulate_planet,planet_pos,vis,region_type,planet_exact)

        #2. signals: quiet total + the touched cells, in their order (as _add_signals_nb)
        if do_ph:
            flux[k]=quiet_ph[0]
        for w in range(nsp):
            spec[k,w]=quiet_sp[w]
        for w in range(ncc):
            ccf[k,w]=quiet_cc[w]
        nbuf=ncc if cc_on_the_fly else 0 #CCF of a touched cell (on the fly only; defined in all the cases for prange)
        bq=np.zeros(nbuf)
        bs=np.zeros(nbuf)
        bf=np.zeros(nbuf)
        nbsp=nsp if sp_on_the_fly else 0 #spectrum of a touched cell (on the fly only)
        sq=np.zeros(nbsp)
        ss_=np.zeros(nbsp)
        sf=np.zeros(nbsp)
        rq=np.zeros(nr) #spectroscopy per ring: fractions of the touched cells added per ring
        rs=np.zeros(nr)
        rf=np.zeros(nr)
        for t in range(nt):
            q=t_cell[t]
            i=cell_ring[q]
            dq=t_frac[t,0]
            fs=t_frac[t,1]
            ff=t_frac[t,2]
            if do_ph:
                flux[k]+=dq*gq_ph[i]+fs*gs_ph[i]+ff*gf_ph[i]
            if nr>0:
                rq[i]+=dq
                rs[i]+=fs
                rf[i]+=ff
            elif sp_on_the_fly:
                #spectrum of this cell for the surfaces it has (a surface with fraction 0 adds 0, as with the grids)
                fac=1.0+vrel_sp[q]/c_light
                if dq!=0.0:
                    _interp_doppler_into_nb(wv_sp,rings_sp[0,i,:],new_wv_sp,fac,sq,0,nsp)
                else:
                    sq[:]=0.0
                if fs!=0.0:
                    _interp_doppler_into_nb(wv_sp,rings_sp[1,i,:],new_wv_sp,fac,ss_,0,nsp)
                else:
                    ss_[:]=0.0
                if ff!=0.0:
                    _interp_doppler_into_nb(wv_sp,rings_sp[2,i,:],new_wv_sp,fac,sf,0,nsp)
                else:
                    sf[:]=0.0
                for w in range(nsp):
                    spec[k,w]+=dq*sq[w]+fs*ss_[w]+ff*sf[w]
            else:
                for w in range(nsp):
                    spec[k,w]+=dq*gq_sp[q,w]+fs*gs_sp[q,w]+ff*gf_sp[q,w]
            if cc_on_the_fly:
                #CCF of this cell for the surfaces it has (a surface with fraction 0 adds 0, as with the grids)
                if dq!=0.0:
                    _interp_shifted_nb(rv_cc,rvs_ring_cc[0,i,:],rvel[q],ccf_ring_cc[0,i,:],ccf_ring_cc[0,i,0],ccf_ring_cc[0,i,-1],bq)
                else:
                    bq[:]=0.0
                if fs!=0.0:
                    _interp_shifted_nb(rv_cc,rvs_ring_cc[1,i,:],rvel[q],ccf_ring_cc[1,i,:],ccf_ring_cc[1,i,0],ccf_ring_cc[1,i,-1],bs)
                else:
                    bs[:]=0.0
                if ff!=0.0:
                    _interp_shifted_nb(rv_cc,rvs_ring_cc[2,i,:],rvel[q],ccf_ring_cc[2,i,:],ccf_ring_cc[2,i,0],ccf_ring_cc[2,i,-1],bf)
                else:
                    bf[:]=0.0
                for w in range(ncc):
                    ccf[k,w]+=dq*bq[w]+fs*bs[w]+ff*bf[w]
            else:
                for w in range(ncc):
                    ccf[k,w]+=dq*gq_cc[q,w]+fs*gs_cc[q,w]+ff*gf_cc[q,w]
        if nr>0:
            _add_rings_nb(k,rq,rs,rf,gq_sp,gs_sp,gf_sp,spec)
        filling[0,k]=100*Aph/area_tot
        filling[1,k]=100*Asp/area_tot
        filling[2,k]=100*Afc/area_tot
        filling[3,k]=100*Apl/area_tot

    return flux,spec,ccf,filling


#######################################################################################################
# Parameters of the CCFs: gaussian fit (Levenberg-Marquardt, analytic Jacobian) and bisector, all CCFs in one call
#######################################################################################################

@nb.njit(cache=True)
def _gaussian_residuals_nb(x,y,p,r,J):
    """Residuals r = C + A exp(-(x-mu)^2/(2 s^2)) - y of the parameters p = (A, mu, s, C) and their Jacobian J (n, 4),
    written in place; returns the cost sum(r^2)."""
    A=p[0]
    mu=p[1]
    s=p[2]
    C=p[3]
    cost=0.0
    for j in range(len(x)):
        d=x[j]-mu
        e=m.exp(-d*d/(2.0*s*s))
        r[j]=C+A*e-y[j]
        cost+=r[j]*r[j]
        J[j,0]=e
        J[j,1]=A*e*d/(s*s)
        J[j,2]=A*e*d*d/(s*s*s)
        J[j,3]=1.0
    return cost


@nb.njit(cache=True)
def _solve4_nb(M,b,x):
    """Solve the 4x4 system M x = b (Gaussian elimination with partial pivoting), x written in place.
    Returns False if the matrix is singular. M and b are modified."""
    n=4
    for k in range(n):
        piv=k
        for i in range(k+1,n):
            if abs(M[i,k])>abs(M[piv,k]):
                piv=i
        if M[piv,k]==0.0 or not np.isfinite(M[piv,k]):
            return False
        if piv!=k:
            for j in range(n):
                M[k,j],M[piv,j]=M[piv,j],M[k,j]
            b[k],b[piv]=b[piv],b[k]
        for i in range(k+1,n):
            f=M[i,k]/M[k,k]
            for j in range(k,n):
                M[i,j]-=f*M[k,j]
            b[i]-=f*b[k]
    for i in range(n-1,-1,-1):
        v=b[i]
        for j in range(i+1,n):
            v-=M[i,j]*x[j]
        x[i]=v/M[i,i]
    return True


@nb.njit(cache=True)
def fit_gaussian_nb(x,y,p0,max_iter=500):
    """Least-squares fit of C + A exp(-(x-mu)^2/(2 s^2)) to y, starting from p0 = (A, mu, s, C).
    Levenberg-Marquardt with the analytic Jacobian (Marquardt damping lam * diag(J^T J)). A step is accepted if it does
    not increase the cost, and the damping is then decreased (x0.3); otherwise it is increased (x10) and the step retried.
    Converged when the step is below 1e-12 of the parameters, when the cost decreases by less than 1e-15 of itself, or
    when no step can decrease the cost any more (minimum reached at the precision of the floating point numbers).
    Returns the parameters (A, mu, s, C) and True if the fit converged (False after max_iter iterations or for a bad fit)."""
    n=len(x)
    p=p0.copy()
    r=np.empty(n)
    J=np.empty((n,4))
    rn=np.empty(n)
    Jn=np.empty((n,4))
    JTJ=np.empty((4,4))
    g=np.empty(4)
    M=np.empty((4,4))
    b=np.empty(4)
    delta=np.empty(4)
    pn=np.empty(4)
    cost=_gaussian_residuals_nb(x,y,p,r,J)
    if not np.isfinite(cost):
        return p,False
    lam=1e-3
    for it in range(max_iter):
        for a in range(4):
            g[a]=0.0
            for k in range(4):
                JTJ[a,k]=0.0
        for j in range(n):
            for a in range(4):
                g[a]+=J[j,a]*r[j]
                for k in range(a,4):
                    JTJ[a,k]+=J[j,a]*J[j,k]
        for a in range(4):
            for k in range(a):
                JTJ[a,k]=JTJ[k,a]
        if cost==0.0:
            return p,True
        accepted=False
        while lam<1e16:
            for a in range(4):
                for k in range(4):
                    M[a,k]=JTJ[a,k]
                M[a,a]+=lam*JTJ[a,a]
                b[a]=-g[a]
            if _solve4_nb(M,b,delta):
                for a in range(4):
                    pn[a]=p[a]+delta[a]
                costn=_gaussian_residuals_nb(x,y,pn,rn,Jn)
                if np.isfinite(costn) and costn<=cost:
                    accepted=True
                    break
            lam*=10.0
        if not accepted:
            #no step decreases the cost: the minimum is reached (to the precision of the floating point numbers)
            return p,np.all(np.isfinite(p)) and p[2]!=0.0
        small=True
        for a in range(4):
            if abs(delta[a])>1e-12*(abs(pn[a])+1e-300):
                small=False
        converged=small or (cost-costn)<=1e-15*cost
        for a in range(4):
            p[a]=pn[a]
        for j in range(n):
            r[j]=rn[j]
            for a in range(4):
                J[j,a]=Jn[j,a]
        cost=costn
        lam=max(lam*0.3,1e-12)
        if converged:
            return p,p[2]!=0.0
    return p,False


@nb.njit(cache=True,parallel=True)
def ccf_params_nb(rv,ccfs,cutleft,cutright,vsini):
    """Parameters of every CCF (rows of ccfs), in parallel. Every CCF is shifted to a minimum of ~0, c = ccf - min + 1e-6:
      bisector of c/max(c) (speed_bisector_nb) and BIS = mean(bisector at 10-40% of the height) - mean(at 60-90%);
      gaussian fit of c on rv[cutleft:cutright] (between the wings of the reference CCF), starting from
      (max(c), rv of the maximum + 100 m/s, 1.5 vsini + 1000 m/s, 1e-6): contrast (amplitude), rv (mean) and
      fwhm = 2 sqrt(2 ln 2) |sigma|.
    Returns rvs, contrast, fwhm, BIS, xbis (n, 50), ybis (n, 50) and failed (bool: the gaussian fit did not converge; the
    CCF then has the placeholder values rv = 100000 m/s, contrast = 1, fwhm = 2.35 m/s, as with a failed scipy fit)."""
    n=ccfs.shape[0]
    rvs=np.zeros(n)
    contrast=np.zeros(n)
    fwhm=np.zeros(n)
    BIS=np.zeros(n)
    xbis_all=np.zeros((n,50))
    ybis_all=np.zeros((n,50))
    failed=np.zeros(n,dtype=np.bool_)
    xs=rv[cutleft:cutright]
    for i in prange(n):
        c=ccfs[i]-np.min(ccfs[i])+0.000001
        _,_,xbis,ybis=speed_bisector_nb(rv,c/np.max(c),True)
        xbis_all[i]=xbis
        ybis_all[i]=ybis
        s1=0.0
        n1=0
        s2=0.0
        n2=0
        for k in range(len(ybis)):
            if ybis[k]>=0.1 and ybis[k]<=0.4:
                s1+=xbis[k]
                n1+=1
            if ybis[k]>=0.6 and ybis[k]<=0.9:
                s2+=xbis[k]
                n2+=1
        BIS[i]=s1/n1-s2/n2
        ys=c[cutleft:cutright]
        k0=np.argmax(ys)
        p0=np.array([ys[k0],xs[k0]+100.0,1.5*vsini+1000.0,0.000001])
        p,ok=fit_gaussian_nb(xs,ys,p0,500)
        if ok:
            contrast[i]=p[0]
            rvs[i]=p[1]
            fwhm[i]=2*m.sqrt(2*m.log(2))*abs(p[2])
        else:
            #placeholder values, as for a failed scipy fit
            contrast[i]=1.0
            rvs[i]=100000.0
            fwhm[i]=2*m.sqrt(2*m.log(2))*1.0
            failed[i]=True
    return rvs,contrast,fwhm,BIS,xbis_all,ybis_all,failed


#######################################################################################################
# 2D maps of the disc (e.g. SDO): projection of the pixels on the grid, filling factors, time series
#######################################################################################################

@nb.njit(cache=True,parallel=True)
def pixel_to_cell_nb(n_rows,n_cols,N,Ngrid_in_ring,origin_upper,flip_horizontal):
    """Grid cell of every pixel of an image of the disc (n_rows x n_cols, the disc fills the image: its diameter is
    the width and the height of the image). Returns an int32 array (n_rows*n_cols,) in the order of the flattened
    image (row by row), with -1 for the pixels outside the disc.
    Orientation (frame of the grid): the horizontal axis of the image is y (the projected equator: columns from left
    to right = y from -1 to 1, i.e. the regions move from left to right with the rotation; flip_horizontal reverses
    it) and the vertical axis is z (the projected rotation axis, north up): origin_upper=True -> row 0 is the top of
    the image (z = +1, as np.loadtxt / imshow(origin='upper')), False -> row 0 is the bottom.
    A pixel belongs to the ring that contains its projected distance from the disc centre (ring i extends from
    sin((i-0.5) w) to sin((i+0.5) w), with w = pi/(2N-1) the angular width of the rings), and to the cell of that ring
    whose position angle around the disc centre is the closest (cells at alpha_j = 2 pi j / n, alpha = atan2(y, -z))."""
    width=np.pi/(2*N-1)
    r_edge=np.empty(N) #outer projected radius of every ring
    for i in range(N):
        r_edge[i]=m.sin(min((i+0.5)*width,np.pi/2))
    r_edge[N-1]=1.0
    ring_start=np.empty(N,dtype=np.int64)
    acc=0
    for i in range(N):
        ring_start[i]=acc
        acc+=Ngrid_in_ring[i]
    out=np.full(n_rows*n_cols,-1,dtype=np.int32)
    for a in prange(n_rows):
        if origin_upper:
            z=1.0-(2.0*a+1.0)/n_rows
        else:
            z=-1.0+(2.0*a+1.0)/n_rows
        for b in range(n_cols):
            y=-1.0+(2.0*b+1.0)/n_cols
            if flip_horizontal:
                y=-y
            r2=y*y+z*z
            if r2>1.0:
                continue
            r=m.sqrt(r2)
            lo=0 #binary search: first ring with r <= r_edge
            hi=N-1
            while lo<hi:
                mid=(lo+hi)//2
                if r_edge[mid]<r:
                    lo=mid+1
                else:
                    hi=mid
            n=Ngrid_in_ring[lo]
            j=0
            if n>1:
                alpha=m.atan2(y,-z)
                if alpha<0.0:
                    alpha+=2.0*np.pi
                j=int(m.floor(alpha/(2.0*np.pi/n)+0.5))%n
            out[a*n_cols+b]=ring_start[lo]+j
    return out


@nb.njit(cache=True,nogil=True)
def maps_to_cells_nb(cell_of_pixel,spot_map,facula_map,npix_cell,spots_over_faculae=True):
    """Mean of the spot map and of the facula map over the pixels of every cell: ff_sp, ff_fc (n_cells,).
    spots_over_faculae: in every pixel the spot is on top of the facula, the facula counts only where there is no spot
    (min(fc, 1 - sp); 0 for a binary pixel flagged in both maps, as the SDO facula maps that include the spots). False:
    the two maps are averaged independently (the old code: a pixel in both maps counts twice, the faculae are capped
    only per cell, afc <= 1 - asp).
    cell_of_pixel: cell of every pixel of the flattened maps (pixel_to_cell_nb), npix_cell: number of pixels of every
    cell (computed once with the projection). Only the pixels that are not 0 in one of the maps are visited in the
    index, so sparse maps (a few % of active pixels) are fast. NaN pixels count as 0 (quiet). A cell without any pixel
    (map too small for the grid) gets 0. nogil: several epochs can be reduced in parallel threads.
    The maps can be of any numeric type (float32, uint8, ...)."""
    ncell=len(npix_cell)
    s_sp=np.zeros(ncell)
    s_fc=np.zeros(ncell)
    for k in range(len(cell_of_pixel)):
        a=spot_map[k]
        b=facula_map[k]
        if a==0 and b==0: #quiet pixel (most of them): nothing to add
            continue
        c=cell_of_pixel[k]
        if c<0: #outside the disc
            continue
        if a!=a: #NaN
            a=0
        if b!=b:
            b=0
        s_sp[c]+=a
        if spots_over_faculae:
            s_fc[c]+=min(b,1-a)
        else:
            s_fc[c]+=b
    ff_sp=np.zeros(ncell)
    ff_fc=np.zeros(ncell)
    for c in range(ncell):
        if npix_cell[c]>0:
            ff_sp[c]=s_sp[c]/npix_cell[c]
            ff_fc[c]=s_fc[c]/npix_cell[c]
    return ff_sp,ff_fc


@nb.njit(cache=True,parallel=True)
def timeseries_maps_nb(N,Ngrid_in_ring,pare,amu,vec_grid,ff_sp_all,ff_fc_all,simulate_planet,planet_pos_all,planet_vis_all,
                       gq_ph,gs_ph,gf_ph,quiet_ph,gq_sp,gs_sp,gf_sp,quiet_sp,sp_per_ring,gq_cc,gs_cc,gf_cc,quiet_cc,planet_exact):
    """Same as timeseries_nb, but the spot and facula filling factors of every cell come from 2D maps:
    ff_sp_all, ff_fc_all (n_times, n_cells) (maps_to_cells_nb, active_region_mask.filling_factors). In every cell the spots are on top of the faculae
    (asp <= 1, afc <= 1 - asp) and the planets (planet_pos_all (n_times, n_planets, 3), planet_vis_all (n_times,
    n_planets)) on top of both, as in generate_ff_core. Returns flux, spec, ccf and the filling factors (4, n_times)."""
    n_times=ff_sp_all.shape[0]
    ncell=vec_grid.shape[0]
    nsp=gq_sp.shape[1]
    ncc=gq_cc.shape[1]
    width=np.pi/(2*N-1)
    half=width/2.0
    cell_ring,area_tot=_cell_ring_nb(N,Ngrid_in_ring,pare)

    flux=np.zeros(n_times)
    spec=np.zeros((n_times,nsp))
    ccf=np.zeros((n_times,ncc))
    filling=np.zeros((4,n_times))

    for k in prange(n_times):
        planet_pos=planet_pos_all[k]
        planet_on=np.zeros(planet_pos.shape[0],dtype=np.bool_)
        any_planet=False
        if simulate_planet:
            for p in range(planet_pos.shape[0]):
                if planet_vis_all[k,p]:
                    planet_on[p]=True
                    any_planet=True
        ff_quiet=np.empty(ncell)
        ff_sp=np.empty(ncell)
        ff_fc=np.empty(ncell)
        Aph=0.0
        Asp=0.0
        Afc=0.0
        Apl=0.0
        for c in range(ncell):
            i=cell_ring[c]
            asp=min(max(ff_sp_all[k,c],0.0),1.0)
            afc=min(max(ff_fc_all[k,c],0.0),1.0)
            if afc+asp>1.0: #the spots are on top of the faculae
                afc=1.0-asp
            apl=0.0
            if any_planet:
                apl=min(_planet_cover_nb(c,i,Ngrid_in_ring[i],amu[i],c==0,width,half,vec_grid,planet_pos,planet_on,planet_exact),1.0)
                asp=asp*(1-apl)
                afc=afc*(1-apl)
            aph=1-asp-afc-apl
            ff_quiet[c]=aph
            ff_sp[c]=asp
            ff_fc[c]=afc
            Aph+=aph*pare[i]
            Asp+=asp*pare[i]
            Afc+=afc*pare[i]
            Apl+=apl*pare[i]
        filling[0,k]=100*Aph/area_tot
        filling[1,k]=100*Asp/area_tot
        filling[2,k]=100*Afc/area_tot
        filling[3,k]=100*Apl/area_tot
        _add_signals_nb(k,ff_quiet,ff_sp,ff_fc,cell_ring,gq_ph,gs_ph,gf_ph,quiet_ph,gq_sp,gs_sp,gf_sp,quiet_sp,sp_per_ring,
                        gq_cc,gs_cc,gf_cc,quiet_cc,flux,spec,ccf)
    return flux,spec,ccf,filling


# =============================================================================================
# flux-conserving resampling (instrument orders)
# =============================================================================================
@nb.njit(cache=True)
def bin_edges_nb(c):
    """Edges of the bins of the centres c: halfway between the centres, the two outer ones extrapolated (as
    specutils.SpectralAxis.bin_edges)."""
    n=len(c)
    e=np.empty(n+1)
    e[0]=((2*c[0]-c[1])+c[0])/2
    for i in range(1,n):
        e[i]=(c[i-1]+c[i])/2
    e[n]=(c[n-1]+(2*c[n-1]-c[n-2]))/2
    return e


@nb.njit(cache=True)
def flux_conserving_resample_nb(in_edges,flux,out_edges):
    """Flux-conserving resampling of flux (bins of edges in_edges) on the bins of edges out_edges (n_out + 1 values,
    bin_edges_nb of the centres): the mean of the input fluxes over every output bin, weighted by the overlap of the
    bins. The output bins beyond the input range are NaN. The same algorithm and the same numbers as
    specutils.manipulation.FluxConservingResampler (extrapolation_treatment='nan_fill'; float64 wavelengths), ~300x
    faster."""
    n_out=len(out_edges)-1
    out=np.zeros(n_out)
    min_idx=0
    for k in range(n_out+1): #last output edge <= the first input edge
        if out_edges[k]<=in_edges[0]:
            min_idx=k
    for k in range(min_idx):
        out[k]=np.nan
    max_idx=n_out
    for k in range(n_out+1): #first output edge > the last input edge
        if out_edges[k]>in_edges[-1]:
            max_idx=k-1
            break
    for k in range(max_idx if max_idx>=0 else 0,n_out):
        out[k]=np.nan
    if max_idx<0:
        max_idx=n_out
    if min_idx>=max_idx: #no output bin inside the input range
        for k in range(n_out):
            out[k]=np.nan
        return out
    #first input bin that intersects the first output bin in range
    idx=0
    while in_edges[idx+1]<=out_edges[min_idx]:
        idx+=1
    for i in range(min_idx,max_idx):
        bin_start=out_edges[i]
        bin_stop=out_edges[i+1]
        first=idx
        while in_edges[idx+1]<bin_stop:
            idx+=1
        final=idx
        if in_edges[idx+1]<=bin_stop:
            idx+=1
        if final==first:
            out[i]=flux[first]
        else:
            num=0.0
            den=0.0
            for j in range(first,final+1):
                lo=bin_start if j==first else in_edges[j]
                hi=bin_stop if j==final else in_edges[j+1]
                num+=flux[j]*(hi-lo)
                den+=hi-lo
            out[i]=num/den
    return out
