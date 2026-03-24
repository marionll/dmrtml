"""
DMRT-ML (Dense Media Radiative Transfer - Multi-Layer)
Copyright (c), all rights reserved, 2007-2012,  Ghislain Picard
email: ghislain.picard@univ-grenoble-alpes.fr

Main contributors: Ludovic Brucker, Alexandre Roy, Florent Dupont

Licensed under the terms of the GNU General Public License version 3:
http://www.opensource.org/licenses/gpl-3.0.html

dmrtml'url: https://snow.univ-grenoble-alpes.fr/dmrtml

Recommended citation:
   Picard, G., Brucker, L., Roy, A., Dupont, F., Fily, M., Royer, A., and Harlow, C.,
   Simulation of the microwave emission of multi-layered snowpacks using the Dense Media Radiative Transfer theory: the DMRT-ML model,
   Geoscientific Model Development, 6, 1061-1078, 2013, https://doi.org/10.5194/gmd-6-1061-2013
"""

from dmrtml_for import (
    dmrtml_pywrapper,
    albedobeta_pywrapper,
    icedielectric_pywrapper,
    compute_streams_pywrapper,
)
import numpy as np

NONSTICKY = 1e6


def run(
    freq,
    n,
    depth,
    density,
    radius,
    temp,
    tau=NONSTICKY,
    fwetness=0,
    medium="S",
    dist=False,
    soilp=None,
    tbatmodown=0,
    eps_ice=(0.0, 0.0),
):
    """
    Run dmrtml and return a DMRTMLResult object.
    Main routine to call dmrtml with Python.
    """
    if soilp is None:
        soilp = SoilParams()

    eps_ice_r, eps_ice_i = eps_ice

    depth, density, radius, temp, tau, fwetness, medium, eps_ice_r, eps_ice_i = (
        _clean_parameters(
            depth, density, radius, temp, tau, fwetness, medium, eps_ice_r, eps_ice_i
        )
    )

    restuple = dmrtml_pywrapper(
        freq,
        n,
        depth,
        density,
        radius,
        temp,
        tau,
        fwetness,
        medium,
        dist,
        soilp.imodel,
        soilp.temp,
        soilp.eps[0],
        soilp.eps[1],
        soilp.sigma,
        soilp.SM,
        soilp.sand,
        soilp.clay,
        soilp.dm_rho,
        soilp.Q,
        soilp.N,
        tbatmodown,
        # eps_ice[0], eps_ice[1])
        eps_ice_r,
        eps_ice_i,
    )

    return DMRTMLResult(restuple)


def dmrtml(*args, **kwargs):
    """
    A synonym for run.
    """
    return run(*args, **kwargs)


class DMRTMLResult:
    """
    Class returned by the dmrtml routine.
    Results from the DMRTML model.
    """

    def __init__(self, restuple):
        """Create a DMRTMLResult from a tuple returned by dmrtml_wrapper."""
        self.TbVarr, self.TbHarr, self.mhu = restuple
        mask = self.mhu > 0
        self.TbVarr = self.TbVarr[mask]
        self.TbHarr = self.TbHarr[mask]
        self.mhu = self.mhu[mask]

    def TbH(self, theta=None):
        """Return TbH for the angle(s) theta)."""
        if theta is None:
            return self.TbHarr
        else:
            return np.interp(theta, self.thetas(), self.TbHarr)

    def TbV(self, theta=None):
        """Return TbV for the angle(s) theta)."""
        if theta is None:
            return self.TbVarr
        else:
            return np.interp(theta, self.thetas(), self.TbVarr)

    def thetas(self):
        """Return the actual zenith angles of the streams used for the calculation."""
        return np.arccos(self.mhu) * 180 / np.pi


def albedobeta(
    frequency,
    density,
    radius,
    temp,
    tau=NONSTICKY,
    fwetness=0,
    medium="S",
    dist=False,
    grodyapproach=True,
):
    """Compute the albedo and extinction with the DMRT theory."""
    return albedobeta_pywrapper(
        frequency, density, radius, temp, tau, fwetness, medium, dist, grodyapproach
    )


def compute_streams(
    freq,
    n,
    density,
    radius=0,
    temp=273,
    tau=NONSTICKY,
    fwetness=0,
    medium="S",
    dist=False,
):
    """Compute the streams angles used by DISORT."""

    depth = np.ones_like(density)
    depth, density, radius, temp, tau, fwetness, medium = _clean_parameters(
        depth, density, radius, temp, tau, fwetness, medium
    )

    restuple = compute_streams_pywrapper(
        freq, n, density, radius, temp, tau, fwetness, medium, dist
    )
    return Streams(restuple)  # ! return mhu,weight,ns,outmhu,ns0


class Streams:
    def __init__(self, restuple):
        """Create a Streams from a tuple returned by compute_streams_pywrapper."""
        self.mhu, self.weight, self.ns, self.outmhu, self.ns0 = restuple


def ice_dielectric_constant(frequency, temperature):
    return icedielectric_pywrapper(frequency, temperature)


class SoilParams:
    """Soil parameters"""

    def __init__(self):
        self.imodel = 0  # no soil
        self.temp = 273
        self.eps = 1, 0
        self.sigma = 0.005
        self.SM = 0.2
        self.sand = 0.4
        self.clay = 0.3
        self.dm_rho = 1100.0

        self.Q = 0
        self.N = 0

    def set_temperature(self, temp):
        self.temp = temp

    def set_soilmoisture(self, SM):
        if SM is not None:
            self.SM = SM

    def set_soiltexture(self, sand, clay, drymatter_density):
        if sand is not None:
            self.sand = sand
        if clay is not None:
            self.clay = clay
        if drymatter_density is not None:
            self.dm_rho = drymatter_density

    def set_roughness(self, sigma):
        if sigma is not None:
            self.sigma = sigma

    def set_QNH(self, q, n, h):
        if q is not None:
            self.Q = q
        if n is not None:
            self.N = n
        if h is not None:
            self.sigma = h

    def set_dielectricconstant(self, epsreal, epsimag=None):
        try:
            self.eps = epsreal.real, epsreal.imag
        except:
            if epsimag is not None:
                self.eps = epsreal, epsimag
            else:
                raise Exception("Invalid value type for epsreal and/or epsimag.")


class NoSoilParams(SoilParams):
    """No soil"""

    def __init__(self, temperature=273):
        SoilParams.__init__(self)
        self.imodel = 0  # imodel=0 no soil (rh=rv=0)
        self.set_temperature(temperature)


class FlatSoilParams(SoilParams):
    """Flat soil"""

    def __init__(
        self,
        temperature=273,
        eps="epspulliainen",
        sand=None,
        clay=None,
        drymatter_density=None,
        SM=None,
    ):
        SoilParams.__init__(self)
        self.set_temperature(temperature)

        if eps == "epspulliainen" or eps == "epsmodel":
            # imodel=2 flat surface, fresnel coefficient with ESS epsilon
            self.imodel = 2
            self.set_soilmoisture(SM)
            self.set_soiltexture(sand, clay, drymatter_density)
            self.eps = 0, 0

        elif eps == "epsdobson":
            # imodel=3 flat surface, fresnel coefficient with Dobson epsilon
            self.imodel = 3
            self.set_soilmoisture(SM)
            self.set_soiltexture(sand, clay, drymatter_density)
            self.eps = 0, 0

        elif eps == "epsmironov":
            # imodel=4 flat surface, fresnel coefficient with Mironov epsilon
            self.imodel = 4
            self.set_soilmoisture(SM)
            self.set_soiltexture(sand, clay, drymatter_density)
            self.eps = 0, 0

        else:
            # imodel=1 flat surface, fresnel coefficient with prescribed eps
            self.imodel = 1
            self.eps = eps


class HUTRoughSoilParams(FlatSoilParams):
    """Rough soil"""

    def __init__(
        self,
        temperature=273,
        eps="epspulliainen",
        sand=None,
        clay=None,
        drymatter_density=None,
        SM=None,
        sigma=None,
    ):
        FlatSoilParams.__init__(
            self, temperature, eps, sand, clay, drymatter_density, SM
        )
        #        print "HUTRoughSoilParams",self.eps
        # imodel=101 HUT roughsoil reflectivity for rough surface with prescribed eps
        # imodel=102 HUT roughsoil reflectivity for rough surface with EPSS eps
        # imodel=103 HUT roughsoil reflectivity for rough surface with Dobson eps
        # imodel=104 HUT roughsoil reflectivity for rough surface with Mironov eps
        self.imodel += 100
        self.set_roughness(sigma)


class QNHRoughSoilParams(FlatSoilParams):
    """Rough soil"""

    def __init__(
        self,
        temperature=273,
        eps="epspulliainen",
        sand=None,
        clay=None,
        drymatter_density=None,
        SM=None,
        Q=None,
        N=None,
        H=None,
    ):
        FlatSoilParams.__init__(
            self, temperature, eps, sand, clay, drymatter_density, SM
        )

        # imodel=301 HUT roughsoil reflectivity for rough surface with prescribed eps
        # imodel=302 HUT roughsoil reflectivity for rough surface with EPSS eps

        # imodel=303 HUT roughsoil reflectivity for rough surface with Dobson eps
        # imodel=304 HUT roughsoil reflectivity for rough surface with Mironov eps
        self.set_QNH(Q, N, H)

        self.imodel += 300


class FlatIceParams(SoilParams):
    """Flat ice"""

    def __init__(self, temperature=273, eps="epsmodel"):

        SoilParams.__init__(self)
        if eps == "epsmodel":
            # imodel=202 ice flat surface, fresnel coefficient with eps model
            self.imodel = 202
        else:
            # imodel=201 ice flat surface, fresnel coefficient with prescribed eps (same as imodel=1)
            self.imodel = 201
            self.eps = eps


def _clean_parameters(
    depth, density, radius, temp, tau, fwetness, medium, eps_ice_r, eps_ice_i
):

    depth = np.asarray(depth, dtype=float)

    density = np.asarray(density, dtype=float)
    if density.size == 1:
        density = density * np.ones(depth.size)

    radius = np.asarray(radius, dtype=float)
    if radius.size == 1:
        radius = radius * np.ones(depth.size)

    temp = np.asarray(temp, dtype=float)
    if temp.size == 1:
        temp = temp * np.ones(depth.size)

    tau = np.asarray(tau, dtype=float)
    if tau.size == 1:
        tau = tau * np.ones(depth.size)

    fwetness = np.asarray(fwetness, dtype=float)
    if fwetness.size == 1:
        fwetness = fwetness * np.ones(depth.size)

    medium = np.asarray(medium, dtype=object)
    if medium.size == 1:
        medium = np.full(depth.size, medium, dtype=object)
        # m = np.chararray(depth.size)
        # m[:] = medium
        # medium = m

    eps_ice_r = np.asarray(eps_ice_r, dtype=float)
    if eps_ice_r.size == 1:
        eps_ice_r = eps_ice_r * np.ones(depth.size)

    eps_ice_i = np.asarray(eps_ice_i, dtype=float)
    if eps_ice_i.size == 1:
        eps_ice_i = eps_ice_i * np.ones(depth.size)

    return depth, density, radius, temp, tau, fwetness, medium, eps_ice_r, eps_ice_i
