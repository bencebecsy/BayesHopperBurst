import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import scipy.linalg as sl

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

import enterprise_extensions

import corner

#import pdb
#pdb.set_trace()
import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP

import BayesHopperBurst

import sys


parfiles = sorted(glob.glob('../12p5yr-like_data/real_4/*.par'))
timfiles = sorted(glob.glob('../12p5yr-like_data/real_4/*.tim'))

print(parfiles)
print(timfiles)

print(len(parfiles))

tmin = 53216.1516728
stop_time = 57781.7766728 #for 12.5yr slice

psrs = []
for i, (p, t) in enumerate(zip(parfiles[::], timfiles[::])):
    print(p)
    psr = Pulsar(p, t, ephem='DE436', clk=None)
    psr.filter_data(start_time = 0.0, end_time = stop_time)
    if (psr.toas.size == 0) or (enterprise_extensions.model_utils.get_tspan([psr]) < 3*365.25*24*3600):
        print("Meh")
        continue
    else:
        print("yeah")
        psrs.append(psr)

#Test whether the time-span for the new dataset is really 12.5-yr
tspan = enterprise_extensions.model_utils.get_tspan(psrs)
print(tspan / (365.25 * 24 * 3600))
print(len(psrs))


#N=int(5e5)
#N=int(1e6)
N=int(1e2)
T_max = 8
n_chain = 6

ts_file = "../12p5yr-like_data/TauScans/tauscan_12p5yr-like_real4_RN.pkl"
glitch_ts_file = "../12p5yr-like_data/TauScans/glitch_tauscan_12p5yr-like_real4_RN.pkl.pkl"

noisedict_file = "../12p5yr-like_data/noisedict_12p5yr-like_efac_rn.json"
RN_start_file = "../12p5yr-like_data/RN_start_values.npz"

savefile = "../Results/samples_12p5yr-like_real4.npz"

results = BayesHopperBurst.run_bhb(N, T_max, n_chain, psrs,
                                   max_n_wavelet=5,
                                   min_n_wavelet=0,
                                   n_wavelet_start=1,
                                   RJ_weight=4,
                                   glitch_RJ_weight=4,
                                   regular_weight=4,
                                   noise_jump_weight=0,
                                   PT_swap_weight=4,
                                   tau_scan_proposal_weight=3,
                                   glitch_tau_scan_proposal_weight=3,
                                   gwb_switch_weight=0,
                                   tau_scan_file=ts_file,
                                   glitch_tau_scan_file=glitch_ts_file,
                                   gwb_log_amp_range=[-18,-11],
                                   rn_log_amp_range=[-18,-11],
                                   wavelet_log_amp_range=[-10.0,-4.0],
                                   per_psr_rn_log_amp_range=[-18,-11],
                                   prior_recovery=False,
                                   gwb_amp_prior='log-uniform',
                                   rn_amp_prior='log-uniform',
                                   wavelet_amp_prior='uniform',
                                   per_psr_rn_amp_prior='log-uniform',
                                   #gwb_on_prior=0.975,
                                   max_n_glitch=5,
                                   n_glitch_start=1,
                                   glitch_log_amp_range=[-10.0,-4.0],
                                   glitch_amp_prior='uniform',
                                   t0_max=12.5,
                                   draw_from_prior_weight=0, de_weight=0,
                                   vary_white_noise=False,
                                   include_gwb=False,
                                   include_rn=True, vary_rn=True,
                                   include_equad_ecorr=False,
                                   wn_backend_selection=False,
                                   noisedict_file=noisedict_file,
                                   include_per_psr_rn=True,
                                   vary_per_psr_rn=False,
                                   #resume_from=None,
                                   per_psr_rn_start_file=RN_start_file,
                                   savefile=savefile, save_every_n=1000)

samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood, betas, PT_acc = results

np.savez(savefile, samples=samples, acc_fraction=acc_fraction, swap_record=swap_record,
                   log_likelihood=log_likelihood, betas=betas, PT_acc=PT_acc)
