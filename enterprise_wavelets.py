from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.stats

#import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const

from enterprise_extensions import model_utils

#import numexpr as ne

def create_gw_antenna_pattern(theta, phi, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians
    
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi), 
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), 
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])

    phat = np.array([np.sin(theta)*np.cos(phi), 
                     np.sin(theta)*np.sin(phi), 
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu

@signal_base.function
def wavelet_delay(toas, theta, phi,
             cos_gwtheta=0, gwphi=0,
             log10_h = None, #seconds
             log10_h2 = None, #seconds
             tau = 0.5, #years
             log10_f0 = -8,
             t0 = 1.0, #years
             phase0 = 0,
             phase02 = 0,
             epsilon = None,
             psi = 0.0,
             tref=0):
    """
    Function to create GW incuced residuals from an elliptically polarized Morlet-Gabor wavelet.
    :param toas:
        Pular toas in seconds
    :param theta:
        Polar angle of pulsar location.
    :param phi:
        Azimuthal angle of pulsar location.
    :param cos_gwtheta:
        Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi:
        Azimuthal angle of GW source in celestial coords [radians]
    :param log10_h:
        log10 of induced residual amplitude [seconds]
    :param tau:
        width of sine-Gaussian [years]
    :param log10_f0:
        log10 of central frequency [Hz]
    :param t0:
        central time of wavelet [years],
    :param phase0:
        Phase at maximum [radians]
    :param epsilon:
        Ellipticity parameter such that h_cross (f) = i*epsilon*h_plus (f)
    :param psi:
        Polarization angle [radians]. Mixes h+ and hx corresponding to rotation along the propagation direction (see eq. (7.24-25) in Maggiore Vol1, 2008).
    :param tref:
        Reference time for phase and frequency [s]
    :return: Vector of induced residuals
    """

    # convert units
    toas -= tref #substract reference
    if log10_h is None:
        Amp = 0.0
    else:
        Amp = 10**log10_h
    t0 *= 365.25*24*3600
    tau *= 365.25*24*3600
    f0 = 10**log10_f0
    gwtheta = np.arccos(cos_gwtheta)

    # calculate residuals for a Morlet-Gabor wavelet
    hplus = Amp * np.exp(-(toas-t0)**2/tau**2) * np.cos(2*np.pi*f0*(toas-t0) + phase0)
    #pi = np.pi
    #hplus = ne.evaluate('Amp * exp(-(toas-t0)**2/tau**2 * cos(2*pi*f0*(toas-t0) + phase0))')
    if epsilon is None:
        if log10_h2 is None:
            Amp_cross = 0.0
        else:
            Amp_cross = 10**log10_h2
        phase0_cross = phase02
    else:
        Amp_cross = epsilon * Amp
        phase0_cross = phase0+np.pi/2
    hcross = Amp_cross * np.exp(-(toas-t0)**2/tau**2) * np.cos(2*np.pi*f0*(toas-t0) + phase0_cross)

    #apply rotation by psi angle (see e.g. eq. (7.24-25) in Maggiore Vol1, 2008)
    rplus = hplus*np.cos(2*psi) - hcross*np.sin(2*psi)
    rcross = hplus*np.sin(2*psi) + hcross*np.cos(2*psi)

    # get antenna pattern funcs and cosMu
    #fplus, fcross, cosMu = utils.create_gw_antenna_pattern(theta, phi, gwtheta, gwphi)
    #print(theta, phi, gwtheta, gwphi)
    fplus, fcross, cosMu = create_gw_antenna_pattern(theta, phi, gwtheta, gwphi)

    res = -fplus*rplus - fcross*rcross #minus sign comes from the fact that what appeas is [pulsar term] - [Earth term] but we don't care about the pulsar term here

    return res


@signal_base.function
def glitch_delay(toas, theta, phi,
             log10_h = None, #seconds
             tau = 0.5, #years
             log10_f0 = -8,
             t0 = 1.0, #years
             phase0 = 0,
             psr_float_idx = 1.0,
             pulsars = None,
             tref=0):
    """
    Function to create non-coherent Morlet-Gabor wavelet glitch.
    :param toas:
        Pular toas in seconds
    :param theta:
        Polar angle of pulsar location - not really needed.
    :param phi:
        Azimuthal angle of pulsar location - not really needed.
    :param log10_h:
        log10 of induced residual amplitude [seconds]
    :param tau:
        width of sine-Gaussian [years]
    :param log10_f0:
        log10 of central frequency [Hz]
    :param t0:
        central time of wavelet [years],
    :param phase0:
        Phase at maximum [radians]
    :param tref:
        Reference time for phase and frequency [s]
    :return: Vector of induced residuals
    """

    #check if this is the right pulsar by checking its sky location - super-kludgy solution :)
    #print(psr_float_idx)
    #print(int(np.round(psr_float_idx)))
    rounded_idx = int(np.round(psr_float_idx))
    if rounded_idx<0 or rounded_idx>=len(pulsars): #this is an error of being outside prior range, but prior will handle that, we just need to avoid an indexing error here
        return toas*0.0
    else: #go ahead as usual
        desired_theta = pulsars[rounded_idx].theta
        desired_phi = pulsars[rounded_idx].phi

        if theta==desired_theta and phi==desired_phi: #it is the selected pulsar
            # convert units
            toas -= tref #substract reference
            if log10_h is None:
                Amp = 0.0
            else:
                Amp = 10**log10_h
            t0 *= 365.25*24*3600
            tau *= 365.25*24*3600
            f0 = 10**log10_f0

            res = Amp * np.exp(-(toas-t0)**2/tau**2) * np.cos(2*np.pi*f0*(toas-t0) + phase0)

            return res
        else: #it is not the selected pulsar
            return toas*0.0


