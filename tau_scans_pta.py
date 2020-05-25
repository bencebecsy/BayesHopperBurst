################################################################################
#
#TauScanPTA -- Produce Tau-scan time-frequency maps for PTA burst search
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2020
################################################################################

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl
import json

import enterprise_cw_funcs_from_git as models

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

class TauScan(object):
    """
    Class for the Tau scans.
    :param psrs: List of `enterprise` Pulsar instances.
    :param params: Dictionary of noise parameters.
    """
    
    def __init__(self, psrs, params=None):
        
        print('Initializing the model...')

        efac = parameter.Constant() 
        equad = parameter.Constant() 
        ef = white_signals.MeasurementNoise(efac=efac)
        eq = white_signals.EquadNoise(log10_equad=equad)

        tm = gp_signals.TimingModel(use_svd=True)

        s = eq + ef + tm

        model = []
        for p in psrs:
            model.append(s(p))
        self.pta = signal_base.PTA(model)  

        # set white noise parameters
        if params is None:
            print('No noise dictionary provided!...')
        else:
            self.pta.set_default_params(params)

        self.psrs = psrs
        self.params = params
                                   
        self.Nmats = None


    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.pta.get_TNT(self.params)
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        #Get noise parameters for pta toaerr**2
        Nvecs = self.pta.get_ndiag(self.params)
        #Get the basis matrix
        Ts = self.pta.get_basis(self.params)
        
        Nmats = [ make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]
        
        return Nmats

    def compute_TauScan(self, tau, t0, f0):
        """
        Computes the Tau-scans.
        :param tau: tau of wavelet to use
        :param t0: central time of wavelet to use
        :param f0: central frequency of wavelet to use
        :returns:
        tau_scan: value of tau-scan at given tau, t0, f0 (map can be produced by looping over these
        """

        tref=53000*86400
        
        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        
        if self.Nmats == None:
            self.Nmats = self.get_Nmats()
        
        n_psr = len(self.psrs)

       
        cos_norm = 0
        sin_norm = 0
        for idx, (psr, Nmat, TNT, phiinv, T) in enumerate(zip(self.psrs, self.Nmats,
                                             TNTs, phiinvs, Ts)):
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            ntoa = len(psr.toas)
 
            wavelet_cos = MorletGaborWavelet(psr.toas-tref, 1.0, tau, f0, t0, 0.0)
            wavelet_sin = MorletGaborWavelet(psr.toas-tref, 1.0, tau, f0, t0, np.pi/2)

            cos_norm += innerProduct_rr(wavelet_cos, wavelet_cos, Nmat, T, Sigma)
            sin_norm += innerProduct_rr(wavelet_sin, wavelet_sin, Nmat, T, Sigma)

        cos_norm = np.sqrt(cos_norm)
        sin_norm = np.sqrt(sin_norm)
        #print(cos_norm, sin_norm)

        tau_scan_cos = 0
        tau_scan_sin = 0
        
        for idx, (psr, Nmat, TNT, phiinv, T) in enumerate(zip(self.psrs, self.Nmats,
                                             TNTs, phiinvs, Ts)):
            
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            
            ntoa = len(psr.toas)
        
            wavelet_cos = MorletGaborWavelet(psr.toas-tref, 1.0/cos_norm, tau, f0, t0, 0.0)
            wavelet_sin = MorletGaborWavelet(psr.toas-tref, 1.0/sin_norm, tau, f0, t0, np.pi/2)
        
            tau_scan_cos += innerProduct_rr(wavelet_cos, psr.residuals, Nmat, T, Sigma)
            tau_scan_sin += innerProduct_rr(wavelet_sin, psr.residuals, Nmat, T, Sigma)
            
        return tau_scan_cos, tau_scan_sin
        #return cos_norm, sin_norm

def MorletGaborWavelet(t, A, tau, f0, t0, phi0):
    return A * np.exp(-(t-t0)**2/tau**2) * np.cos(2*np.pi*f0*(t-t0) + phi0)


def innerProduct_rr(x, y, Nmat, Tmat, Sigma, TNx=None, TNy=None, brave=False):
    """
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y
        
        :param x: vector timeseries 1
        :param y: vector timeseries 2
        :param Nmat: white noise matrix
        :param Tmat: Modified design matrix including red noise/jitter
        :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
        :param TNx: T^T N^{-1} x precomputed
        :param TNy: T^T N^{-1} y precomputed
        :return: inner product (x|y)
        """
    
    # white noise term
    Ni = Nmat
    xNy = np.dot(np.dot(x, Ni), y)
    Nx, Ny = np.dot(Ni, x), np.dot(Ni, y)
    
    if TNx == None and TNy == None:
        TNx = np.dot(Tmat.T, Nx)
        TNy = np.dot(Tmat.T, Ny)
    
    if brave:
        cf = sl.cho_factor(Sigma, check_finite=False)
        SigmaTNy = sl.cho_solve(cf, TNy, check_finite=False)
    else:
        cf = sl.cho_factor(Sigma)
        SigmaTNy = sl.cho_solve(cf, TNy)

    ret = xNy - np.dot(TNx, SigmaTNy)

    return ret

def make_Nmat(phiinv, TNT, Nvec, T):
    
    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)   
    cf = sl.cho_factor(Sigma)
    Nshape = np.shape(T)[0]

    TtN = Nvec.solve(other = np.eye(Nshape),left_array = T)
    
    #Put pulsar's autoerrors in a diagonal matrix
    Ndiag = Nvec.solve(other = np.eye(Nshape),left_array = np.eye(Nshape))
    
    expval2 = sl.cho_solve(cf,TtN)
    #TtNt = np.transpose(TtN)
    
    #An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T,expval2)
