################################################################################
#
#BayesWavePTA -- Bayesian search for burst GW signals in PTA data based on the BayesWave algorithm
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2020
################################################################################

import numpy as np
import matplotlib.pyplot as plt

import enterprise
import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils
from enterprise.signals import deterministic_signals

from enterprise_extensions.frequentist import Fe_statistic
import enterprise_wavelets as models

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_bw_pta(N, T_max, n_chain, pulsars, max_n_wavelet=1, n_wavelet_prior='flat', n_wavelet_start='random', RJ_weight=0,
               regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder = None,
               tau_scan_proposal_weight=0, tau_scan_file=None, draw_from_prior_weight=0,
               de_weight=0, prior_recovery=False, wavelet_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform',
               gwb_log_amp_range=[-18,-11], rn_log_amp_range=[-18,-11], wavelet_log_amp_range=[-18,-11],
               vary_white_noise=False, efac_start=1.0,
               include_gwb=False, gwb_switch_weight=0,
               include_rn=False, vary_rn=False, rn_params=[-13.0,1.0], jupyter_notebook=False,
               gwb_on_prior=0.5):
    
    ptas = get_ptas(pulsars, vary_white_noise=vary_white_noise, include_rn=include_rn, vary_rn=vary_rn, include_gwb=include_gwb, max_n_wavelet=max_n_wavelet, efac_start=efac_start, rn_amp_prior=rn_amp_prior, rn_log_amp_range=rn_log_amp_range, rn_params=rn_params, gwb_amp_prior=gwb_amp_prior, gwb_log_amp_range=gwb_log_amp_range, wavelet_amp_prior=wavelet_amp_prior, wavelet_log_amp_range=wavelet_log_amp_range, prior_recovery=prior_recovery)

    print(ptas)
    for i, PTA in enumerate(ptas):
        print(i)
        for j, pta in enumerate(PTA):
            print(j)
            print(ptas[i][j].params)
            #point_to_test = np.tile(np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5]),i+1)
            #print(PTA.summary())


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################

def get_ptas(pulsars, vary_white_noise=True, include_rn=True, vary_rn=True, include_gwb=True, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-13.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], prior_recovery=False):
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        #equad = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant(efac_start)
        #equad = parameter.Constant(wn_params[1])
    
    ef = white_signals.MeasurementNoise(efac=efac)
    #eq = white_signals.EquadNoise(log10_equad=equad)
    tm = gp_signals.TimingModel(use_svd=True)

    base_model = ef + tm
    
    #adding red noise if included
    if include_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        
        if vary_rn:
            #rn = ext_models.common_red_noise_block(prior='uniform', Tspan=Tspan, name='com_rn')
            amp_name = 'com_rn_log10_A'
            if rn_amp_prior == 'uniform':
                log10_Arn = parameter.LinearExp(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            elif rn_amp_prior == 'log-uniform':
                log10_Arn = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            gam_name = 'com_rn_gamma'
            gamma_rn = parameter.Uniform(0, 7)(gam_name)
            pl = utils.powerlaw(log10_A=log10_Arn, gamma=gamma_rn)
            rn = gp_signals.FourierBasisGP(spectrum=pl, coefficients=False, components=30, Tspan=Tspan,
                                           modes=None, name='com_rn')
        else:
            log10_A = parameter.Constant(rn_params[0])
            gamma = parameter.Constant(rn_params[1])
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
        
        base_model += rn 

    #make base models including GWB
    if include_gwb:
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        amp_name = 'gw_log10_A'
        if gwb_amp_prior == 'uniform':
            log10_Agw = parameter.LinearExp(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)
        elif gwb_amp_prior == 'log-uniform':
            log10_Agw = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)
        
        gam_name = 'gw_gamma'
        gamma_val = 13.0/3
        gamma_gw = parameter.Constant(gamma_val)(gam_name)

        cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        gwb = gp_signals.FourierBasisCommonGP(cpl, utils.hd_orf(), coefficients=False,
                                              components=30, Tspan=Tspan,
                                              modes=None, name='gw')

        base_model_gwb = base_model + gwb

    #setting up the pta object
    wavelets = []
    for i in range(max_n_wavelet):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), -7)(str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)(str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'gwphi')
        phase0 = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'phase0')
        epsilon = parameter.Uniform(0, 1.0)(str(i)+'_'+'epsilon')
        tau = parameter.Uniform(0.1, 5)(str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, 10.0)(str(i)+'_'+'t0')
        if wavelet_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h')
        elif wavelet_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h')
        else:
            print("CW amplitude prior of {0} not available".format(cw_amp_prior))
        wavelet_wf = models.wavelet_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_h = log10_h,
                                          tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0,
                                          epsilon = epsilon, tref=53000*86400)
        wavelets.append(deterministic_signals.Deterministic(wavelet_wf, name='wavelet'+str(i)))
    
    ptas = []
    for n_wavelet in range(max_n_wavelet+1):
        PTA = []
        s = base_model
        for i in range(n_wavelet):
            s = s + wavelets[i]

        model = []
        for p in pulsars:
            model.append(s(p))
        
        #set the likelihood to unity if we are in prior recovery mode
        if prior_recovery:
            PTA.append(get_prior_recovery_pta(signal_base.PTA(model)))
        else:
            PTA.append(signal_base.PTA(model))

        if include_gwb:
            s_gwb = base_model_gwb
            for i in range(n_wavelet):
                s_gwb = s_gwb + wavelets[i]
        
            model_gwb = []
            for p in pulsars:
                model_gwb.append(s_gwb(p))
            
            #set the likelihood to unity if we are in prior recovery mode
            if prior_recovery:
                PTA.append(get_prior_recovery_pta(signal_base.PTA(model_gwb)))
            else:
                PTA.append(signal_base.PTA(model_gwb))
        ptas.append(PTA)

    return ptas

################################################################################
#
#MAKE PTA OBJECT FOR PRIOR RECOVERY
#
################################################################################

def get_prior_recovery_pta(pta):
    class prior_recovery_pta:
        def __init__(self, pta):
            self.pta = pta
            self.params = pta.params
            self.pulsars = pta.pulsars
        def get_lnlikelihood(self, x):
            return 0.0
        def get_lnprior(self, x):
            return self.pta.get_lnprior(x)

    return prior_recovery_pta(pta)
