#NUMBA ############################################
import numba as nb
import numpy as np
import math as m

from numba import njit, prange



@njit(cache=True)
def interp_linear_nb(x, xp, fp):

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
def interp_doppler_nb(wv, flux, new_wv, velocity):

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

    Nt=2*N-1 #N is number of concentric rings. Nt is counting them two times minus the center one.
    width=180.0/(2*N-1) #width of one grid element.

    centres=np.append(0,np.linspace(width,90-width/2,N-1)) #colatitudes of the concentric grids. The pole of the grid faces the observer.
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
            ts.append(centres[i]) #latitude
            alphas.append(angles[j]) #longitude


    alphas=np.array(alphas) #longitude of grid (pole faces observer)
    ts=np.array(ts) #colatitude of grid
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
    #fraction of a grid cell (width, angular) covered by a circular region of radius rad at angular distance dist.
    #Same formulas as generate_ff; central=True for the central (circular) cell.
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
def generate_ff(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,vis, active_region_types):
    #Filling factors do not depend on the order of the
    #regions: in every cell the spots are computed first, and the faculae afterwards, with the spots on top of them.
    #The visible part of a facula depends on how its disc and the disc of each spot are placed (decided once per epoch):
    #   spot inside the facula   -> facula - spot            (exact)
    #   facula inside the spot   -> nothing                  (exact)
    #   discs do not overlap     -> facula                   (exact)
    #   partial overlap          -> facula*(1-spot)          (independent approximation)
    ncell=vec_grid.shape[0]
    nreg=len(vis)-1
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    half=width/2.0

    #visible regions with radius>0, spots and faculae apart
    sidx=np.empty(nreg,dtype=np.int64)
    fidx=np.empty(nreg,dtype=np.int64)
    ns=0
    nf=0
    for l in range(nreg):
        if vis[l]==1.0 and spot_pos[l][2]>0.0:
            if active_region_types[l]=='sp':
                sidx[ns]=l
                ns+=1
            elif active_region_types[l]=='fc':
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

    planet_on=False
    if simulate_planet:
        if vis[-1]==1.0:
            planet_on=True

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

            #PLANET
            if planet_on:
                dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1])-vec_grid[c,1])**2+(planet_pos[0]*m.sin(planet_pos[1])-vec_grid[c,2])**2) #grid-planet distance
                if central:
                    width2=2.0*m.sin(half)
                else:
                    width2=amu[i]*width
                if dist>width2/2+planet_pos[2]:
                    apl=0.0
                elif dist<planet_pos[2]-width2/2:
                    apl=1.0
                else:
                    apl=-(dist-planet_pos[2]-width2/2)/width2

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


@nb.njit(cache=True,error_model='numpy')
def generate_ff_2d_mask(N,Ngrid_in_ring,pare,rs,sdo_input):
    #Filling factors from 2D maps of the disc (e.g. SDO): every pixel is assigned to a grid cell,
    #and the filling factor of the cell is the mean of the spot/facula maps over its pixels.
    array_sp=sdo_input[0]
    array_fc=sdo_input[1]
    n_pxls=len(array_sp)
    typ_cell,_,_=projection_pxl_to_ss_grid(Ngrid_in_ring,rs,n_pxls)

    typ_flat=typ_cell.ravel()
    sp_flat=array_sp.ravel()
    fc_flat=array_fc.ravel()

    ncell=0
    for i in range(N):
        ncell+=Ngrid_in_ring[i]
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
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            idx=typ_flat==c
            n_tot=np.sum(idx)

            if n_tot>0:
                asp=np.nansum(sp_flat[idx])/n_tot
                afc=np.nansum(fc_flat[idx])/n_tot

            if afc<0.0:
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
def loop_compute_immaculate_nb(N,Ngrid_in_ring,ccf_tot,rvel,rv,rvs_ring,ccf_ring):
    #CCF of each pixel, adding doppler and interpolating
    iteration=0
    #Compute the position of the grid projected on the sphere and its radial velocity.
    for i in range(0,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #loop for each grid in the ring
            ccf_tot[iteration,:]=ccf_tot[iteration,:]+interpolation_nb(rv,rvs_ring[i,:] + rvel[iteration],ccf_ring[i,:],ccf_ring[i,0],ccf_ring[i,-1]) 
            iteration=iteration+1

    return ccf_tot


@nb.njit(cache=True,error_model='numpy')
def cross_correlation_mask(rv,wv,f,wvm,fm, phoenix_resolution, ccf_norm):
    """
    Function to compute CCF against Phoenix-spectra. The steps used
    are specific to Phoenix spectra, do not use other spectra.
    
    """
    ccf = np.zeros(len(rv))
    lenm = len(wvm)
    wvmin=wv[0]

    if phoenix_resolution:

        for i in range(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
            #for each mask line
            for j in range(lenm):
                #find wavelengths right and left of the line.
                wvline=wvshift[j]

                if wvline<3000.0:
                    idxlf = int((wvline-wvmin)/0.1)

                elif wvline<4999.986:
                    if wvmin<3000.0:
                        idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((wvline-3000.0)/0.006) 
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
        for i in range(len(rv)):
            wvshift=wvm*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
            #for each mask line
            for j in range(lenm):
                wvline=wvshift[j]
                
                if wvline > wv[-1]:
                    pass
                else:
                    dist = wvline - wvmin
                    k = 0

                    while dist >= 0.0:
                        k += 1
                        dist = wvline - wv[k]

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
                    
    if ccf_norm:
        return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))
    else:
        return - ccf / np.min(ccf) + 1 #TEST


@nb.njit(cache=True,error_model='numpy')
def speed_bisector_nb(rv,ccf,integrated_bis):
    ''' Fit the bisector of the CCF with a 5th deg polynomial
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
            if ynew[-1-i]>ynew[-i] and ynew[i]<0.8*maxccf:
                cutleft=len(ynew)-i
                break

    #TEST

    xnew=xnew[cutleft:cutright]
    ynew=ynew[cutleft:cutright]
    
    minright=np.min(ynew[xnew>maxrv])
    minleft=np.min(ynew[xnew<maxrv])
    minccf=np.max(np.array([minright,minleft]))
    
    if integrated_bis:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.99*maxccf,50) #from 5% to maximum
    else:
        ybis=np.linspace(minccf+0.1*(maxccf-minccf),0.999*maxccf,50) #from 5% to maximum
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


@nb.njit(cache=True,error_model='numpy')
def gaussian2(x, amplitude, mean, stddev,C):
    return C + amplitude * np.exp(-(x-mean)**2/(2*stddev**2))



@nb.njit(cache=True,error_model='numpy')
def projection_pxl_to_ss_grid(Ngrid_in_ring, rs, n_pxls):
    N = len(Ngrid_in_ring)
    x = np.linspace(-1.0, 1.0, n_pxls)
    two_pi = 2.0 * np.pi
    half_pi = np.pi / 2.0

    # Ring radius limits (midpoints between unique rounded radii)
    r_sorted = np.sort(rs.astype(np.float64))
    r_unique = np.empty_like(r_sorted)
    count = 0
    for i in range(r_sorted.shape[0]):
        val = np.round(r_sorted[i], 6)
        if count == 0 or val != r_unique[count - 1]:
            r_unique[count] = val
            count += 1
    lim_r = np.empty(N)
    for i in range(N - 1):
        lim_r[i] = (r_unique[i + 1] + r_unique[i]) / 2.0
    lim_r[N - 1] = 1.0

    # Cumulative cell offsets and angular widths per ring
    offsets = np.empty(N, dtype=np.int64)
    acc = 0
    for k in range(N):
        offsets[k] = acc
        acc += Ngrid_in_ring[k]
    dtheta = np.empty(N)
    for k in range(N):
        dtheta[k] = two_pi / Ngrid_in_ring[k]

    xg = np.empty((n_pxls, n_pxls))
    yg = np.empty((n_pxls, n_pxls))
    typ_cell = np.empty((n_pxls, n_pxls))

    for i in range(n_pxls):
        y_val = x[i]
        for j in range(n_pxls):
            x_val = x[j]
            jf = n_pxls - 1 - j  # horizontal flip applied directly
            r_val = np.sqrt(x_val * x_val + y_val * y_val)
            if r_val > 1.0:
                xg[i, j] = np.nan
                yg[i, j] = np.nan
                typ_cell[i, jf] = np.nan
                continue
            xg[i, j] = x_val
            yg[i, j] = y_val

            angle = np.arctan2(x_val, y_val) + half_pi
            if angle < 0:
                angle += two_pi

            # Binary search: first ring with lim_r[ring] >= r_val
            lo = 0
            hi = N
            while lo < hi:
                mid = (lo + hi) >> 1
                if lim_r[mid] < r_val:
                    lo = mid + 1
                else:
                    hi = mid
            ring = lo if lo < N else N - 1

            n_cells = Ngrid_in_ring[ring]
            idx = int(np.round(((angle + half_pi) % two_pi) / dtheta[ring])) % n_cells
            typ_cell[i, jf] = offsets[ring] + idx

    return typ_cell, xg, yg
