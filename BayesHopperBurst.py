################################################################################
#
#BayesWavePTA -- Bayesian search for burst GW signals in PTA data based on the BayesWave algorithm
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2020
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import json

import enterprise
import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import selections
from enterprise.signals.selections import Selection

import enterprise_wavelets as models
import pickle

import multiprocessing
from joblib import Parallel, delayed

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_bw_pta(N, T_max, n_chain, pulsars, max_n_wavelet=1, min_n_wavelet=0, n_wavelet_prior='flat', n_wavelet_start='random', RJ_weight=0, glitch_RJ_weight=0,
               regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder = None,
               tau_scan_proposal_weight=0, tau_scan_file=None, draw_from_prior_weight=0,
               de_weight=0, prior_recovery=False, wavelet_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform', per_psr_rn_amp_prior='uniform',
               gwb_log_amp_range=[-18,-11], rn_log_amp_range=[-18,-11], per_psr_rn_log_amp_range=[-18,-11], wavelet_log_amp_range=[-18,-11],
               vary_white_noise=False, efac_start=1.0, include_equad_ecorr=False, wn_backend_selection=False, noisedict_file=None,
               include_gwb=False, gwb_switch_weight=0,
               include_rn=False, vary_rn=False, num_wn_params=1, rn_params=[-13.0,1.0], include_per_psr_rn=False, vary_per_psr_rn=False, per_psr_rn_start_file=None,
               jupyter_notebook=False, gwb_on_prior=0.5,
               max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], n_glitch_prior='flat', n_glitch_start='random', t0_max=10.0, tref=53000*86400,
               glitch_tau_scan_proposal_weight=0, glitch_tau_scan_file=None, TF_prior_file=None,
               save_every_n=10000, savefile=None, resume_from=None, parallel=False):

    if parallel:
        print("Using parallelization -------------------------------------------------------------------------------------")
        print(multiprocessing.cpu_count())

    if TF_prior_file is None:
        TF_prior = None
    else:
        with open(TF_prior_file, 'rb') as f:
            TF_prior = pickle.load(f)
    
    ptas = get_ptas(pulsars, vary_white_noise=vary_white_noise, include_equad_ecorr=include_equad_ecorr, wn_backend_selection=wn_backend_selection, noisedict_file=noisedict_file, include_rn=include_rn, vary_rn=vary_rn, include_per_psr_rn=include_per_psr_rn, vary_per_psr_rn=vary_per_psr_rn, include_gwb=include_gwb, max_n_wavelet=max_n_wavelet, efac_start=efac_start, rn_amp_prior=rn_amp_prior, rn_log_amp_range=rn_log_amp_range, rn_params=rn_params, per_psr_rn_amp_prior=per_psr_rn_amp_prior, per_psr_rn_log_amp_range=per_psr_rn_log_amp_range, gwb_amp_prior=gwb_amp_prior, gwb_log_amp_range=gwb_log_amp_range, wavelet_amp_prior=wavelet_amp_prior, wavelet_log_amp_range=wavelet_log_amp_range, prior_recovery=prior_recovery, max_n_glitch=max_n_glitch, glitch_amp_prior=glitch_amp_prior, glitch_log_amp_range=glitch_log_amp_range, t0_max=t0_max, TF_prior=TF_prior, tref=tref)

    print(ptas)
    for i in range(len(ptas)):
        for j in range(len(ptas[i])):
            for k in range(len(ptas[i][j])):
                print(i,j,k)
                print(ptas[i][j][k].params)
                #point_to_test = np.tile(np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5]),i+1)
    
    print(ptas[-1][-1][-1].summary())

    #fisher updating every n_fish_update step
    n_fish_update = 200 #50
    #print out status every n_status_update step
    n_status_update = 10

    #setting up temperature ladder
    if T_ladder is None:
        #using geometric spacing
        c = T_max**(1.0/(n_chain-1))
        Ts = c**np.arange(n_chain)
        print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
 Temperature ladder is:\n".format(n_chain,c),Ts)
    else:
        Ts = np.array(T_ladder)
        n_chain = Ts.size
        print("Using {0} temperature chains with custom spacing: ".format(n_chain),Ts)

    #printitng out the prior used on GWB on/off
    if include_gwb:
        print("Prior on GWB on/off: {0}%".format(gwb_on_prior*100))

    #set up and print out prior on number of wavelets
    if max_n_wavelet!=0:
        if n_wavelet_prior=='flat':
            n_wavelet_prior = np.ones(max_n_wavelet+1)/(max_n_wavelet+1-min_n_wavelet)
            for i in range(min_n_wavelet):
                n_wavelet_prior[i] = 0.0
        else:
            n_wavelet_prior = np.array(n_wavelet_prior)
            n_wavelet_norm = np.sum(n_wavelet_prior)
            n_wavelet_prior *= 1.0/n_wavelet_norm
        print("Prior on number of wavelets: ", n_wavelet_prior)

    #set up and print out prior on number of glitches
    if max_n_glitch!=0:
        if n_glitch_prior=='flat':
            n_glitch_prior = np.ones(max_n_glitch+1)/(max_n_glitch+1)
        else:
            n_glitch_prior = np.array(n_glitch_prior)
            n_glitch_norm = np.sum(n_glitch_prior)
            n_glitch_prior *= 1.0/n_glitch_norm
        print("Prior on number of glitches: ", n_glitch_prior)

    #setting up array for the samples
    num_params = max_n_wavelet*10+max_n_glitch*6
    num_params += 2 #for keepeng a record of number of wavelets and glitches
    if include_gwb:
        num_params += 1

    num_per_psr_params = 0
    num_noise_params = 0
    if vary_white_noise:
        num_per_psr_params += len(pulsars)*num_wn_params
        num_noise_params += len(pulsars)*num_wn_params
    if vary_rn:
        num_noise_params += 2
    if vary_per_psr_rn:
        num_per_psr_params += 2*len(pulsars)
        num_noise_params += 2*len(pulsars)

    num_params += num_noise_params
    print('-'*5)
    print(num_params)
    print(num_noise_params)
    print(num_per_psr_params)
    print('-'*5)

    if resume_from is not None:
        print("Resuming from file: " + resume_from)
        npzfile = np.load(resume_from)
        swap_record = list(npzfile['swap_record'])
        log_likelihood_resume = npzfile['log_likelihood']
        samples_resume = npzfile['samples']

        N_resume = samples_resume.shape[1]
        print("# of samples sucessfully read in: " + str(N_resume))

        samples = np.zeros((n_chain, N_resume+N, num_params))
        samples[:,:N_resume,:] = np.copy(samples_resume)

        log_likelihood = np.zeros((n_chain,N_resume+N))
        log_likelihood[:,:N_resume] = np.copy(log_likelihood_resume)
    else:
        samples = np.zeros((n_chain, N, num_params))
    
        #set up log_likelihood array
        log_likelihood = np.zeros((n_chain,N))

        #filling first sample with random draw
        for j in range(n_chain):
            #set up n_wavelet
            if n_wavelet_start is 'random':
                n_wavelet = np.random.choice( np.arange(min_n_wavelet,max_n_wavelet+1) )
            else:
                n_wavelet = n_wavelet_start
            #set up n_glitch
            if n_glitch_start is 'random':
                n_glitch = np.random.choice(max_n_glitch+1)
            else:
                n_glitch = n_glitch_start

            samples[j,0,0] = n_wavelet
            samples[j,0,1] = n_glitch
            if j==0:
                print("Starting with n_wavelet=",n_wavelet)
                print("Starting with n_glitch=",n_glitch)

            if n_wavelet!=0:
                #making sure all wavelets get the same sky location and ellipticity
                init_cos_gwtheta = ptas[n_wavelet][0][0].params[0].sample()
                init_psi = ptas[n_wavelet][0][0].params[1].sample()
                init_gwphi = ptas[n_wavelet][0][0].params[2].sample()
                for which_wavelet in range(n_wavelet):
                    samples[j,0,2+0+which_wavelet*10] = init_cos_gwtheta
                    samples[j,0,2+1+which_wavelet*10] = init_psi
                    samples[j,0,2+2+which_wavelet*10] = init_gwphi
                    #randomly pick other wavelet parameters separately fo each wavelet
                    samples[j,0,2+3+which_wavelet*10:2+10+which_wavelet*10] = np.hstack(p.sample() for p in ptas[n_wavelet][0][0].params[3:10])

            if n_glitch!=0:
                for which_glitch in range(n_glitch):
                    samples[j,0,2+10*max_n_wavelet+which_glitch*6:2+10*max_n_wavelet+6+which_glitch*6] = np.hstack(p.sample() for p in ptas[0][n_glitch][0].params[:6])

            if vary_white_noise and not vary_per_psr_rn:
                samples[j,0,2+max_n_wavelet*10+max_n_glitch*6:2+max_n_wavelet*10+max_n_glitch*6+len(pulsars)*num_wn_params] = np.ones(len(pulsars)*num_wn_params)*efac_start
            elif vary_per_psr_rn and not vary_white_noise:
                if per_psr_rn_start_file==None:
                    samples[j,0,2+max_n_wavelet*10+max_n_glitch*6:2+max_n_wavelet*10+max_n_glitch*6+2*len(pulsars)] = np.hstack(p.sample() for p in ptas[n_wavelet][0][0].params[n_wavelet*10:n_wavelet*10+2*len(pulsars)])
                else:
                    RN_noise_data = np.load(per_psr_rn_start_file)
                    samples[j,0,2+max_n_wavelet*10+max_n_glitch*6:2+max_n_wavelet*10+max_n_glitch*6+2*len(pulsars)] = RN_noise_data['RN_start']
            elif vary_per_psr_rn and vary_white_noise: #vary both per psr RN and WN
                samples[j,0,2+max_n_wavelet*10+max_n_glitch*6:2+max_n_wavelet*10+max_n_glitch*6+(2+num_wn_params)*len(pulsars)] = np.hstack(p.sample() for p in ptas[n_wavelet][0][0].params[n_wavelet*10:n_wavelet*10+(2+num_wn_params)*len(pulsars)])
            if vary_rn:
                samples[j,0,2+max_n_wavelet*10+max_n_glitch*6+num_per_psr_params:2+max_n_wavelet*10+max_n_glitch*6+num_noise_params] = np.array([ptas[n_wavelet][0][0].params[n_wavelet*10+num_noise_params-2].sample(), ptas[n_wavelet][0][0].params[n_wavelet*10+num_noise_params-1].sample()])
            if include_gwb:
                samples[j,0,2+max_n_wavelet*10+max_n_glitch*6+num_noise_params] = ptas[n_wavelet][0][1].params[n_wavelet*10+num_noise_params].sample()
        print(samples[0,0,:])
        n_wavelet = get_n_wavelet(samples, 0, 0)
        n_glitch = get_n_glitch(samples, 0, 0)
        first_sample = strip_samples(samples, 0, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)  
        print(first_sample)
        log_likelihood[0,0] = ptas[n_wavelet][n_glitch][0].get_lnlikelihood(first_sample)
        print(log_likelihood[0,0])
        print(ptas[n_wavelet][n_glitch][0].get_lnprior(first_sample))

    #setting up array for the fisher eigenvalues
    #one for wavelet parameters which we will keep updating
    eig = np.ones((n_chain, max_n_wavelet, 10, 10))*0.1

    #also one for the glitch parameters
    eig_glitch = np.ones((n_chain, max_n_glitch, 6, 6))*0.03

    #one for GWB and common rn parameters, which we will keep updating
    if include_gwb:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0], [0,0.3,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
    else:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()

    #and one for white noise parameters, which we will not update
    if vary_white_noise and not vary_per_psr_rn:
        eig_per_psr = np.broadcast_to(np.eye(len(pulsars)*num_wn_params)*0.1, (n_chain, len(pulsars)*num_wn_params, len(pulsars)*num_wn_params) ).copy()
        #calculate wn eigenvectors
        for j in range(n_chain):
            n_wavelet = get_n_wavelet(samples, j, 0)
            n_glitch = get_n_glitch(samples, j, 0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][0][0], T_chain=Ts[j], n_wavelet=1, dim=len(pulsars)*num_wn_params, offset=n_wavelet*10+n_glitch*6, parallel=parallel)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]
    elif vary_per_psr_rn and not vary_white_noise:
        eig_per_psr = np.broadcast_to(np.eye(2*len(pulsars))*0.1, (n_chain, 2*len(pulsars), 2*len(pulsars)) ).copy()
        for j in range(n_chain):
            n_wavelet = get_n_wavelet(samples, j, 0)
            n_glitch = get_n_glitch(samples, j, 0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][0][0], T_chain=Ts[j], n_wavelet=1, dim=2*len(pulsars), offset=n_wavelet*10+n_glitch*6, parallel=parallel)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]
    elif vary_per_psr_rn and vary_white_noise: #vary both per psr RN and WN
        eig_per_psr = np.broadcast_to(np.eye((2+num_wn_params)*len(pulsars))*0.1, (n_chain, (2+num_wn_params)*len(pulsars), (2+num_wn_params)*len(pulsars)) ).copy()
        for j in range(n_chain):
            n_wavelet = get_n_wavelet(samples, j, 0)
            n_glitch = get_n_glitch(samples, j, 0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][0][0], T_chain=Ts[j], n_wavelet=1, dim=(2+num_wn_params)*len(pulsars), offset=n_wavelet*10+n_glitch*6, parallel=parallel)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]

    #read in tau_scan data if we will need it
    if tau_scan_proposal_weight+RJ_weight>0:
        if tau_scan_file==None:
            raise Exception("tau-scan data file is needed for tau-scan global propsals")
        with open(tau_scan_file, 'rb') as f:
            tau_scan_data = pickle.load(f)
            print("Tau-scan data read in successfully!")
        
        tau_scan = tau_scan_data['tau_scan']
        print(len(tau_scan))
        
        TAU_list = list(tau_scan_data['tau_edges'])
        F0_list = tau_scan_data['f0_edges']
        T0_list = tau_scan_data['t0_edges']

        #check if same prior range was used
        log_f0_max = float(ptas[-1][-1][-1].params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[-1][-1][-1].params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[-1][-1][-1].params[8]._typename.split('=')[2][:-1])
        t0_min = float(ptas[-1][-1][-1].params[8]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[-1][-1][-1].params[9]._typename.split('=')[2][:-1])
        tau_min = float(ptas[-1][-1][-1].params[9]._typename.split('=')[1].split(',')[0])

        print("#"*70)
        print("Tau-scan and MCMC prior range check (they must be the same)")
        print("tau_min: ", TAU_list[0], tau_min)
        print("tau_max: ", TAU_list[-1], tau_max)
        print("t0_min: ", T0_list[0][0]/3600/24/365.25, t0_min)
        print("t0_max: ", T0_list[0][-1]/3600/24/365.25, t0_max)
        print("f0_min: ", F0_list[0][0], 10**log_f0_min)
        print("f0_max: ", F0_list[0][-1], 10**log_f0_max)
        print("#"*70)
        
        #normalization
        norm = 0.0
        for idx, TTT in enumerate(tau_scan):
            for kk in range(TTT.shape[0]):
                for ll in range(TTT.shape[1]):
                    df = np.log10(F0_list[idx][kk+1]/F0_list[idx][kk])
                    dt = (T0_list[idx][ll+1]-T0_list[idx][ll])/3600/24/365.25
                    dtau = (TAU_list[idx+1]-TAU_list[idx])
                    norm += TTT[kk,ll]*df*dt*dtau
        tau_scan_data['norm'] = norm #TODO: Implement some check to make sure this is normalized over the same range as the prior range used in the MCMC
        print(norm)

    #read in glitch_tau_scan data if we will need it
    if glitch_tau_scan_proposal_weight+glitch_RJ_weight>0:
        if glitch_tau_scan_file==None:
            raise Exception("glitch-tau-scan data file is needed for glitch model tau-scan global propsals")
        with open(glitch_tau_scan_file, 'rb') as f:
            glitch_tau_scan_data = pickle.load(f)
            print("Glitch tau-scan data read in successfully!")
        
        TAU_list = list(glitch_tau_scan_data['tau_edges'])
        F0_list = glitch_tau_scan_data['f0_edges']
        T0_list = glitch_tau_scan_data['t0_edges']

        #check if same prior range was used
        log_f0_max = float(ptas[-1][-1][-1].params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[-1][-1][-1].params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[-1][-1][-1].params[8]._typename.split('=')[2][:-1])
        t0_min = float(ptas[-1][-1][-1].params[8]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[-1][-1][-1].params[9]._typename.split('=')[2][:-1])
        tau_min = float(ptas[-1][-1][-1].params[9]._typename.split('=')[1].split(',')[0])

        print("#"*70)
        print("Glitch tau--scan and MCMC prior range check (they must be the same)")
        print("tau_min: ", TAU_list[0], tau_min)
        print("tau_max: ", TAU_list[-1], tau_max)
        print("t0_min: ", T0_list[0][0]/3600/24/365.25, t0_min)
        print("t0_max: ", T0_list[0][-1]/3600/24/365.25, t0_max)
        print("f0_min: ", F0_list[0][0], 10**log_f0_min)
        print("f0_max: ", F0_list[0][-1], 10**log_f0_max)
        print("#"*70)
        
        #normalization
        glitch_tau_scan_data['psr_idx_proposal'] = np.ones(len(pulsars))
        for i in range(len(pulsars)):
            print(i)
            glitch_tau_scan = glitch_tau_scan_data['tau_scan'+str(i)]
            print(len(glitch_tau_scan))

            norm = 0.0
            for idx, TTT in enumerate(glitch_tau_scan):
                for kk in range(TTT.shape[0]):
                    for ll in range(TTT.shape[1]):
                        df = np.log10(F0_list[idx][kk+1]/F0_list[idx][kk])
                        dt = (T0_list[idx][ll+1]-T0_list[idx][ll])/3600/24/365.25
                        dtau = (TAU_list[idx+1]-TAU_list[idx])
                        norm += TTT[kk,ll]*df*dt*dtau
            glitch_tau_scan_data['norm'+str(i)] = norm #TODO: Implement some check to make sure this is normalized over the same range as the prior range used in the MCMC
            print(norm)

            tau_scan_limit = 0.1#0 #--start form 1 to avoid having zeros in the proposal
            #check if we've read in a tau-scan file
            if tau_scan_proposal_weight+RJ_weight<=0:
                #make fake tau_scan_data to use in next step
                tau_scan_data = {}
                tau_scan_data['tau_scan'] = [ggg*0.0 for ggg in glitch_tau_scan]
            for g_TS, TS in zip(glitch_tau_scan, tau_scan_data['tau_scan']):
                TS_max = np.max( g_TS - TS/np.sqrt(float(len(pulsars))) )
                if TS_max>tau_scan_limit:
                    tau_scan_limit = TS_max
            print(tau_scan_limit)
            glitch_tau_scan_data['psr_idx_proposal'][i] = tau_scan_limit

        glitch_tau_scan_data['psr_idx_proposal'] = glitch_tau_scan_data['psr_idx_proposal']/np.sum(glitch_tau_scan_data['psr_idx_proposal'])
        print('-'*20)
        print("Glitch psr index proposal:")
        print(glitch_tau_scan_data['psr_idx_proposal'])
        print(np.sum(glitch_tau_scan_data['psr_idx_proposal']))
        print('-'*20) 

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((7, n_chain)) #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, PT, fisher, noise_jump)
    a_no=np.zeros((7, n_chain))
    acc_fraction = a_yes/(a_no+a_yes)
    if resume_from is None:
        swap_record = []
    rj_record = []

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + tau_scan_proposal_weight +
                    RJ_weight + gwb_switch_weight + noise_jump_weight + glitch_tau_scan_proposal_weight + glitch_RJ_weight)
    swap_probability = PT_swap_weight/total_weight
    tau_scan_proposal_probability = tau_scan_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    glitch_tau_scan_proposal_probability = glitch_tau_scan_proposal_weight/total_weight
    glitch_RJ_probability = glitch_RJ_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {3:.2f}%\nGlitch RJ moves: {7:.2f}%\nGWB-switches: {4:.2f}%\n\
Tau-scan-proposals: {1:.2f}%\nGlitch tau-scan-proposals: {6:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\nNoise jump: {5:.2f}%".format(swap_probability*100,
          tau_scan_proposal_probability*100, regular_probability*100,
          RJ_probability*100, gwb_switch_probability*100, noise_jump_probability*100, glitch_tau_scan_proposal_probability*100, glitch_RJ_probability*100))

    if resume_from is None:
        start_iter = 0
        stop_iter = N
    else:
        start_iter = N_resume-1 #-1 because if only 1 sample is read in that's the same as having a different starting point and start_iter should still be 0
        stop_iter = N_resume-1+N

    for i in range(int(start_iter), int(stop_iter-1)): #-1 because ith step here produces (i+1)th sample based on ith sample
        ########################################################
        #
        #write results to file every save_every_n iterations
        #
        ########################################################
        if savefile is not None and i%save_every_n==0 and i!=start_iter:
            np.savez(savefile, samples=samples[:,:i,:], acc_fraction=acc_fraction, swap_record=swap_record, log_likelihood=log_likelihood[:,:i])
        ########################################################
        #
        #print out run state every n_status_update iterations
        #
        ########################################################
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            if jupyter_notebook:
                print('Progress: {0:2.2f}% '.format(i/N*100) + '\r',end='')
            else:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                        'Acceptance fraction #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, PT, fisher, noise_jump):')
                print(acc_fraction)
        #################################################################################
        #
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        #
        #################################################################################
        if i%n_fish_update==0:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                for j in range(n_chain):
                    n_wavelet = get_n_wavelet(samples, j, i)
                    n_glitch = get_n_glitch(samples, j, i)
                    if include_gwb:
                        gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
                    else:
                        gwb_on = 0

                    #wavelet eigenvectors
                    if n_wavelet!=0:
                        eigenvectors = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][n_glitch][gwb_on], T_chain=Ts[j], n_wavelet=n_wavelet, parallel=parallel)
                        #print("Eigen wavelet")
                        if np.all(eigenvectors):
                            #print("+")
                            #print(eigenvectors)
                            eig[j,:n_wavelet,:,:] = eigenvectors
                        #else:
                        #    print("-")
                    #glitch eigenvectors
                    if n_glitch!=0:
                        eigen_glitch = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][n_glitch][gwb_on], T_chain=Ts[j], n_wavelet=n_glitch, dim=6, offset=n_wavelet*10, parallel=parallel)
                        #print("Eigen glitch")
                        if np.all(eigen_glitch):
                            #print("+")
                            #print(eigen_glitch)
                            eig_glitch[j,:n_glitch,:,:] = eigen_glitch
                        #else:
                        #    print("-")
                        #    print(eigen_glitch)
                    #RN+GWB eigenvectors
                    if include_gwb:
                        eigvec_rn = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][n_glitch][gwb_on], T_chain=Ts[j], n_wavelet=1, dim=3, offset=n_wavelet*10+n_glitch*6+num_per_psr_params, parallel=parallel)
                    else:
                        eigvec_rn = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][n_glitch][0], T_chain=Ts[j], n_wavelet=1, dim=2, offset=n_wavelet*10+n_glitch*6+num_per_psr_params, parallel=parallel)
                    #print("Eigen RN+GWB")
                    if np.all(eigvec_rn):
                        #print("+")
                        #print(eigvec_rn)
                        eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]
                    #else:
                    #    print("-")

            elif samples[0,i,0]!=0:
                n_wavelet = get_n_wavelet(samples, 0, i)
                n_glitch = get_n_glitch(samples, 0, i)
                if include_gwb:
                    gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
                else:
                    gwb_on = 0
                eigenvectors = get_fisher_eigenvectors(strip_samples(samples, 0, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), ptas[n_wavelet][n_glitch][gwb_on], T_chain=Ts[0], n_wavelet=n_wavelet, parallel=parallel)
                #check if eigenvector calculation was succesful
                #if not, we just keep the initialized eig full of 0.1 values              
                if np.all(eigenvectors):
                    eig[0,:n_wavelet,:,:] = eigenvectors
        ###########################################################
        #
        #Do the actual MCMC step
        #
        ###########################################################
        #draw a random number to decide which jump to do
        jump_decide = np.random.uniform()
        #print("-"*50)
        #print(samples[0,i,:])
        #PT swap move
        if jump_decide<swap_probability:
            do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params, log_likelihood)
        #global proposal based on tau_scan
        elif jump_decide<swap_probability+tau_scan_proposal_probability:
            do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, log_likelihood, parallel)
        #do RJ move
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability):
            do_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, log_likelihood, parallel)
        #do GWB switch move
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability):
            gwb_switch_move(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, gwb_on_prior, gwb_log_amp_range, log_likelihood)
        #do noise jump
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability):
            noise_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig_per_psr, include_gwb, num_noise_params, vary_white_noise, log_likelihood, parallel)
        #glitch model global proposal based on tau_scan
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability+glitch_tau_scan_proposal_probability):
            do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, glitch_tau_scan_data, log_likelihood)
        #do glitch RJ move
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability+glitch_tau_scan_proposal_probability+glitch_RJ_probability):
            do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, glitch_tau_scan_data, log_likelihood)
        #regular step
        else:
            regular_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, log_likelihood, parallel)
        #print(samples[0,i+1,:])
        #print("-"*50)

    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood

################################################################################
#
#GLITCH REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a glitch wavelet
#
################################################################################
def do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, glitch_tau_scan_data, log_likelihood):
    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']

    for j in range(n_chain):
        #print("-- ", j)
        n_wavelet = get_n_wavelet(samples, j, i)
        n_glitch = get_n_glitch(samples, j, i)

        if include_gwb:
            gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
        else:
            gwb_on = 0

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_glitch==0 or (direction_decide<add_prob and n_glitch!=max_n_glitch): #adding a wavelet------------------------------------------------------
            #pick which pulsar to add a glitch to
            psr_idx = np.random.choice(len(ptas[n_wavelet][0][gwb_on].pulsars), p=glitch_tau_scan_data['psr_idx_proposal'])

            #load in the appropriate tau-scan
            tau_scan = glitch_tau_scan_data['tau_scan'+str(psr_idx)]
            #print(i)
            tau_scan_limit = 0
            for TS in tau_scan:
                TS_max = np.max(TS)
                if TS_max>tau_scan_limit:
                    tau_scan_limit = TS_max
            #print(tau_scan_limit)

            log_f0_max = float(ptas[0][1][gwb_on].params[0]._typename.split('=')[2][:-1])
            log_f0_min = float(ptas[0][1][gwb_on].params[0]._typename.split('=')[1].split(',')[0])
            t0_max = float(ptas[0][1][gwb_on].params[4]._typename.split('=')[2][:-1])
            t0_min = float(ptas[0][1][gwb_on].params[4]._typename.split('=')[1].split(',')[0])
            tau_max = float(ptas[0][1][gwb_on].params[5]._typename.split('=')[2][:-1])
            tau_min = float(ptas[0][1][gwb_on].params[5]._typename.split('=')[1].split(',')[0])

            accepted = False
            while accepted==False:
                log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
                t0_new = np.random.uniform(low=t0_min, high=t0_max)
                tau_new = np.random.uniform(low=tau_min, high=tau_max)

                tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
                f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
                t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

                #print(tau_new, t0_new, 10**log_f0_new)
                #print(tau_idx, t0_idx, f0_idx)

                tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
                #print(tau_scan_new_point/tau_scan_limit)
                if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                    accepted = True
                    #print("Yeeeh!")

            #randomly select phase and amplitude
            phase0_new = ptas[0][1][gwb_on].params[2].sample()
            log10_h_new = ptas[0][1][gwb_on].params[1].sample()

            prior_ext = ptas[0][1][gwb_on].params[2].get_pdf(phase0_new) * ptas[0][1][gwb_on].params[1].get_pdf(log10_h_new)# * ptas[0][1][gwb_on].params[3].get_pdf(float(psr_idx))

            samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            new_point = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch+1, max_n_glitch)
            new_glitch = np.array([log_f0_new, log10_h_new, phase0_new, float(psr_idx), t0_new, tau_new])
            new_point[n_wavelet*10+n_glitch*6:n_wavelet*10+(n_glitch+1)*6] = new_glitch

            log_acc_ratio = ptas[n_wavelet][(n_glitch+1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_L = log_acc_ratio
            log_acc_ratio += ptas[n_wavelet][(n_glitch+1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

            #apply normalization
            tau_scan_new_point_normalized = tau_scan_new_point/glitch_tau_scan_data['norm'+str(psr_idx)]

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized/glitch_tau_scan_data['psr_idx_proposal'][psr_idx]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==0:
                acc_ratio *= 0.5
            if n_glitch==max_n_glitch-1:
                acc_ratio *= 2.0
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)+1]/n_glitch_prior[int(n_glitch)]

            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Yeeeh")
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch+1
                samples[j,i+1,2:2+n_wavelet*10] = new_point[:n_wavelet*10]
                samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+(n_glitch+1)*6] = new_point[n_wavelet*10:n_wavelet*10+(n_glitch+1)*6]
                samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+(n_glitch+1)*6:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

        elif n_glitch==max_n_glitch or (direction_decide>add_prob and n_glitch!=0):   #removing a wavelet----------------------------------------------------------
            #choose which glitch to remove
            remove_index = np.random.randint(n_glitch)

            samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            new_point = np.delete(samples_current, range(n_wavelet*10+remove_index*6,n_wavelet*10+(remove_index+1)*6))

            log_acc_ratio = ptas[n_wavelet][(n_glitch-1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_L = log_acc_ratio
            log_acc_ratio += ptas[n_wavelet][(n_glitch-1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

            #getting old parameters
            tau_old = samples[j,i,2+max_n_wavelet*10+remove_index*6+5]
            f0_old = 10**samples[j,i,2+max_n_wavelet*10+remove_index*6+0]
            t0_old = samples[j,i,2+max_n_wavelet*10+remove_index*6+4]
            log10_h_old = samples[j,i,2+max_n_wavelet*10+remove_index*6+1]
            phase0_old = samples[j,i,2+max_n_wavelet*10+remove_index*6+2]
           
            #get old psr index and load in appropriate tau scan
            psr_idx_old = int(np.round(samples[j,i,2+max_n_wavelet*10+remove_index*6+3]))
            tau_scan_old = glitch_tau_scan_data['tau_scan'+str(psr_idx_old)]
            tau_scan_limit_old = 0
            for TS in tau_scan_old:
                TS_max = np.max(TS)
                if TS_max>tau_scan_limit_old:
                    tau_scan_limit_old = TS_max
            #print(tau_scan_limit_old)

            #getting tau_scan at old point
            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

            #print(tau_old, TAU_list)
            #print(tau_idx_old, f0_idx_old, t0_idx_old)

            tau_scan_old_point = tau_scan_old[tau_idx_old][f0_idx_old, t0_idx_old]
            
            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/glitch_tau_scan_data['norm'+str(psr_idx_old)]

            prior_ext = ptas[0][1][gwb_on].params[2].get_pdf(phase0_old) * ptas[0][1][gwb_on].params[1].get_pdf(log10_h_old)# * ptas[0][1][gwb_on].params[3].get_pdf(psr_idx_old)

            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized*glitch_tau_scan_data['psr_idx_proposal'][psr_idx_old]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==1:
                acc_ratio *= 2.0
            if n_glitch==max_n_glitch:
                acc_ratio *= 0.5
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)-1]/n_glitch_prior[int(n_glitch)]

            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Ohhhhhhhhhhhhh")
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch-1
                samples[j,i+1,2:2+n_wavelet*10] = new_point[:n_wavelet*10]
                samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+(n_glitch-1)*6] = new_point[n_wavelet*10:n_wavelet*10+(n_glitch-1)*6]
                samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+(n_glitch-1)*6:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]


################################################################################
#
#GLITCH MODEL GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, glitch_tau_scan_data, log_likelihood):
    #print("GLITCH TAU-GLOBAL")

    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']

    for j in range(n_chain):
        #print("-- ", j)
        #check if there's any wavelet -- stay at given point if not
        n_wavelet = get_n_wavelet(samples, j, i)
        n_glitch = get_n_glitch(samples, j, i)
        #if j==0:
        #    print(n_wavelet)
        #    print(n_glitch)
        #    print(samples[j,i,:])
        if n_glitch==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            #print("No glitch to vary!")
            continue

        if include_gwb:
            gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
        else:
            gwb_on = 0

        #select which glitch to change
        glitch_select = np.random.randint(n_glitch)
        #if j==0: print(glitch_select)

        #pick which pulsar to move the glitch to (stay at where we are in 50% of the time) -- might be an issue with detailed balance
        if np.random.uniform()<=0.5:
            psr_idx = np.random.choice(len(ptas[n_wavelet][0][gwb_on].pulsars))
        else:
            psr_idx = int(np.round(samples[j,i,2+max_n_wavelet*10+glitch_select*6+3]))
        #if j==0: print(psr_idx)

        #load in the appropriate tau-scan
        tau_scan = glitch_tau_scan_data['tau_scan'+str(psr_idx)]
        #print(i)
        tau_scan_limit = 0
        for TS in tau_scan:
            TS_max = np.max(TS)
            if TS_max>tau_scan_limit:
                tau_scan_limit = TS_max
        #print(tau_scan_limit)

        log_f0_max = float(ptas[0][1][gwb_on].params[0]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[0][1][gwb_on].params[0]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[0][1][gwb_on].params[4]._typename.split('=')[2][:-1])
        t0_min = float(ptas[0][1][gwb_on].params[4]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[0][1][gwb_on].params[5]._typename.split('=')[2][:-1])
        tau_min = float(ptas[0][1][gwb_on].params[5]._typename.split('=')[1].split(',')[0])

        accepted = False
        while accepted==False:
            log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
            t0_new = np.random.uniform(low=t0_min, high=t0_max)
            tau_new = np.random.uniform(low=tau_min, high=tau_max)

            tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
            f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
            t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

            #print(tau_new, t0_new, 10**log_f0_new)
            #print(tau_idx, t0_idx, f0_idx)

            tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
            #print(tau_scan_new_point/tau_scan_limit)
            if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                accepted = True
                #print("Yeeeh!")

        #randomly select phase and amplitude
        phase0_new = ptas[0][1][gwb_on].params[2].sample()
        log10_h_new = ptas[0][1][gwb_on].params[1].sample()

        samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
        new_point = np.copy(samples_current)
        new_point[n_wavelet*10+glitch_select*6:n_wavelet*10+(glitch_select+1)*6] = np.array([log_f0_new, log10_h_new, phase0_new,
                                                                                           float(psr_idx), t0_new, tau_new])

        #if j==0:
        #    print(samples_current)
        #    print(new_point)

        log_acc_ratio = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_L = log_acc_ratio
        log_acc_ratio += ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

        #getting ratio of proposal densities!
        tau_old = samples[j,i,2+max_n_wavelet*10+glitch_select*6+5]
        f0_old = 10**samples[j,i,2+max_n_wavelet*10+glitch_select*6+0]
        t0_old = samples[j,i,2+max_n_wavelet*10+glitch_select*6+4]

        #get old psr index and load in appropriate tau scan
        psr_idx_old = int(np.round(samples[j,i,2+max_n_wavelet*10+glitch_select*6+3]))
        tau_scan_old = glitch_tau_scan_data['tau_scan'+str(psr_idx_old)]
        tau_scan_limit_old = 0
        for TS in tau_scan_old:
            TS_max = np.max(TS)
            if TS_max>tau_scan_limit_old:
                tau_scan_limit_old = TS_max
        #print(tau_scan_limit_old)

        tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
        f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
        t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

        #print(tau_old, TAU_list)
        #print(tau_idx_old, f0_idx_old, t0_idx_old)

        tau_scan_old_point = tau_scan_old[tau_idx_old][f0_idx_old, t0_idx_old]
        
        log10_h_old = samples[j,i,2+max_n_wavelet*10+glitch_select*6+1]
        hastings_extra_factor = ptas[0][-1][gwb_on].params[1].get_pdf(log10_h_old) / ptas[0][-1][gwb_on].params[1].get_pdf(log10_h_new)
    
        acc_ratio = np.exp(log_acc_ratio)*(tau_scan_old_point/tau_scan_new_point) * (tau_scan_limit/tau_scan_limit_old) * hastings_extra_factor
        #if j==0:
        #    print(acc_ratio)
        #    print(np.exp(log_acc_ratio))
        #    print(tau_scan_old_point/tau_scan_new_point)
        #    print(tau_scan_limit/tau_scan_limit_old)
        #    print(hastings_extra_factor)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Ohh jeez")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:2+n_wavelet*10] = new_point[:n_wavelet*10]
            samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
            samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
            a_yes[1,j]+=1
            log_likelihood[j,i+1] = log_L
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]


################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a wavelet
#
################################################################################
#wrapper
def do_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, log_likelihood, parallel):
    #print("RJ")
    tau_scan = tau_scan_data['tau_scan']

    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max
    
    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']
    
    #print(samples[0,i,:])
    #print(i)

    if parallel:
        num_cores = multiprocessing.cpu_count()
        rrr = Parallel(n_jobs=num_cores)(delayed(rj_move_core)(j, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood) for j in range(n_chain))
        samples_new, log_L, yes_or_no = zip(*rrr)
        #print(samples_new)
    else:
        rrr = [rj_move_core(j, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood) for j in range(n_chain)]
        samples_new, log_L, yes_or_no = zip(*rrr)

    for j in range(n_chain):
        samples[j,i+1,:] = samples_new[j]
        log_likelihood[j,i+1] = log_L[j]
        if yes_or_no[j]==1:
            a_yes[2,j] += 1
        else:
            a_no[2,j] += 1

#core jump function
def rj_move_core(j, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood):

    n_wavelet = get_n_wavelet(samples, j, i)
    n_glitch = get_n_glitch(samples, j, i)
    #if j==0:
        #print(n_wavelet)
        #print(n_glitch)
        #print(samples[j,i,:])
    
    if include_gwb:
        gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
    else:
        gwb_on = 0

    add_prob = 0.5 #same propability of addind and removing
    #decide if we add or remove a signal
    direction_decide = np.random.uniform()
    if n_wavelet==min_n_wavelet or (direction_decide<add_prob and n_wavelet!=max_n_wavelet): #adding a wavelet------------------------------------------------------
        if j==0: rj_record.append(1)
        #if j==0: print("Propose to add a wavelet")

        log_f0_max = float(ptas[-1][0][gwb_on].params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[-1][0][gwb_on].params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[-1][0][gwb_on].params[8]._typename.split('=')[2][:-1])
        t0_min = float(ptas[-1][0][gwb_on].params[8]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[-1][0][gwb_on].params[9]._typename.split('=')[2][:-1])
        tau_min = float(ptas[-1][0][gwb_on].params[9]._typename.split('=')[1].split(',')[0])

        accepted = False
        while accepted==False:
            log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
            t0_new = np.random.uniform(low=t0_min, high=t0_max)
            tau_new = np.random.uniform(low=tau_min, high=tau_max)

            tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
            f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
            t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

            #print(tau_new, t0_new, 10**log_f0_new)
            #print(tau_idx, t0_idx, f0_idx)

            tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
            #print(tau_scan_new_point/tau_scan_limit)
            if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                accepted = True
                #print("Yeeeh!")

        #randomly select other parameters
        log10_h_new = ptas[-1][0][gwb_on].params[4].sample()
        log10_h_cross_new = ptas[-1][0][gwb_on].params[5].sample()
        phase0_new = ptas[-1][0][gwb_on].params[6].sample()
        phase0_cross_new = ptas[-1][0][gwb_on].params[7].sample()
        #if this is the first wavelet, draw sky location and polarization angle too
        if n_wavelet==0:
            cos_gwtheta_new = ptas[-1][0][gwb_on].params[0].sample()
            gwphi_new = ptas[-1][0][gwb_on].params[2].sample()
            psi_new = ptas[-1][0][gwb_on].params[1].sample()
        #if this is not the first wavelet, copy sky location and ellipticity from existing wavelet(s)
        else:
            cos_gwtheta_new = np.copy(samples[j,i,2+0])
            gwphi_new = np.copy(samples[j,i,2+2])
            psi_new = np.copy(samples[j,i,2+1])

        prior_ext = (ptas[-1][0][gwb_on].params[0].get_pdf(cos_gwtheta_new) * ptas[-1][0][gwb_on].params[1].get_pdf(psi_new) *
                     ptas[-1][0][gwb_on].params[2].get_pdf(gwphi_new) *
                     ptas[-1][0][gwb_on].params[4].get_pdf(log10_h_new) * ptas[-1][0][gwb_on].params[5].get_pdf(log10_h_cross_new) *
                     ptas[-1][0][gwb_on].params[6].get_pdf(phase0_new) * ptas[-1][0][gwb_on].params[7].get_pdf(phase0_cross_new))

        samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

        new_point = strip_samples(samples, j, i, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch)
        new_wavelet = np.array([cos_gwtheta_new, psi_new, gwphi_new, log_f0_new, log10_h_new, log10_h_cross_new,
                                phase0_new, phase0_cross_new, t0_new, tau_new])
        new_point[n_wavelet*10:(n_wavelet+1)*10] = new_wavelet

        #if j==0:
        #    print(samples_current)
        #    print(new_point)


        log_acc_ratio = ptas[(n_wavelet+1)][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_L = log_acc_ratio
        log_acc_ratio += ptas[(n_wavelet+1)][n_glitch][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)
       
        
        #apply normalization
        tau_scan_new_point_normalized = tau_scan_new_point/tau_scan_data['norm']

        acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized
        #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
        if n_wavelet==min_n_wavelet:
            acc_ratio *= 0.5
        if n_wavelet==max_n_wavelet-1:
            acc_ratio *= 2.0
        #accounting for n_wavelet prior
        acc_ratio *= n_wavelet_prior[int(n_wavelet)+1]/n_wavelet_prior[int(n_wavelet)]
        #if j==0:
        #    print("Add wavelet")
        #    print(acc_ratio)
        #    print(np.exp(log_acc_ratio)/prior_ext/ptas[-1][gwb_on].params[3].get_pdf(log_f0_new)/ptas[-1][gwb_on].params[6].get_pdf(t0_new)/ptas[-1][gwb_on].params[7].get_pdf(tau_new))
        #    print(log_acc_ratio)
        #    print(ptas[(n_wavelet+1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]-ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j])
        #    print(ptas[(n_wavelet+1)][gwb_on].get_lnprior(new_point)-ptas[n_wavelet][gwb_on].get_lnprior(samples_current))
        #    print(1/prior_ext)
        #    print(1/tau_scan_new_point_normalized)
        #    print(n_wavelet_prior[int(n_wavelet)+1]/n_wavelet_prior[int(n_wavelet)])
        samples_new = np.zeros(samples.shape[2])
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Yeeeh")
            samples_new[0] = n_wavelet+1
            samples_new[1] = n_glitch
            samples_new[2:2+(n_wavelet+1)*10] = new_point[:(n_wavelet+1)*10]
            samples_new[2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[(n_wavelet+1)*10:(n_wavelet+1)*10+n_glitch*6]
            samples_new[2+max_n_wavelet*10+max_n_glitch*6:] = new_point[(n_wavelet+1)*10+n_glitch*6:]
            #a_yes[2,j] += 1
            #log_likelihood[j,i+1] = log_L
            yes_or_no = 1
        else:
            #samples[j,i+1,:] = samples[j,i,:]
            #a_no[2,j] += 1
            log_L = log_likelihood[j,i]
            samples_new = np.copy(samples[j,i,:])
            yes_or_no = 0
        
        return samples_new, log_L, yes_or_no

    elif n_wavelet==max_n_wavelet or (direction_decide>add_prob and n_wavelet!=min_n_wavelet):   #removing a wavelet----------------------------------------------------------
        if j==0: rj_record.append(-1)
        #if j==0: print("Propose to remove a wavelet")
        #choose which wavelet to remove
        remove_index = np.random.randint(n_wavelet)

        samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
        new_point = np.delete(samples_current, range(remove_index*10,(remove_index+1)*10))

        #if j==0:
        #    print(samples_current)
        #    print(new_point)

        log_acc_ratio = ptas[(n_wavelet-1)][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_L = log_acc_ratio
        log_acc_ratio += ptas[(n_wavelet-1)][n_glitch][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

        #getting tau_scan at old point
        tau_old = samples[j,i,2+9+remove_index*10]
        f0_old = 10**samples[j,i,2+3+remove_index*10]
        t0_old = samples[j,i,2+8+remove_index*10]

        tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
        f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
        t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

        tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]

        #apply normalization
        tau_scan_old_point_normalized = tau_scan_old_point/tau_scan_data['norm']

        #getting external parameter priors
        log10_h_old = np.copy(samples[j,i,2+4+remove_index*10])
        log10_h_cross_old = np.copy(samples[j,i,2+5+remove_index*10])
        phase0_old = np.copy(samples[j,i,2+6+remove_index*10])
        phase0_cross_old = np.copy(samples[j,i,2+7+remove_index*10])
        cos_gwtheta_old = np.copy(samples[j,i,2+0+remove_index*10])
        gwphi_old = np.copy(samples[j,i,2+2+remove_index*10])
        psi_old = np.copy(samples[j,i,2+1+remove_index*10])

        #if j==0:
        #    print("---")
        #    print("cos_theta: ", cos_gwtheta_old)
        #    print("epsilon: ", epsilon_old)
        #    print("phi: ", gwphi_old)
        #    print("f0: ", f0_old)
        #    print("log10h: ", log10_h_old)
        #    print("phase0: ", phase0_old)
        #    print("t0: ", t0_old)
        #    print("tau: ", tau_old)
        #    print(samples_current)
        #    print(new_point)
        #    print("---")
            

        prior_ext = (ptas[-1][0][gwb_on].params[0].get_pdf(cos_gwtheta_old) * ptas[-1][0][gwb_on].params[1].get_pdf(psi_old) *
                     ptas[-1][0][gwb_on].params[2].get_pdf(gwphi_old) *
                     ptas[-1][0][gwb_on].params[4].get_pdf(log10_h_old) * ptas[-1][0][gwb_on].params[5].get_pdf(log10_h_cross_old) *
                     ptas[-1][0][gwb_on].params[6].get_pdf(phase0_old) * ptas[-1][0][gwb_on].params[7].get_pdf(phase0_cross_old))

        acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized
        #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
        if n_wavelet==min_n_wavelet+1:
            acc_ratio *= 2.0
        if n_wavelet==max_n_wavelet:
            acc_ratio *= 0.5
        #accounting for n_wavelet prior
        acc_ratio *= n_wavelet_prior[int(n_wavelet)-1]/n_wavelet_prior[int(n_wavelet)]
        #if j==0:
        #    print("Remove wavelet")
        #    print(acc_ratio)
        #    print(np.exp(log_acc_ratio)*prior_ext*ptas[-1][gwb_on].params[3].get_pdf(np.log10(f0_old))*ptas[-1][gwb_on].params[6].get_pdf(t0_old)*ptas[-1][gwb_on].params[7].get_pdf(tau_old))
        #    print(log_acc_ratio)
        #    print(ptas[(n_wavelet-1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]-ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j])
        #    print(ptas[(n_wavelet-1)][gwb_on].get_lnprior(new_point)-ptas[n_wavelet][gwb_on].get_lnprior(samples_current))
        #    print(1/prior_ext)
        #    print(1/tau_scan_old_point_normalized)
        #    print(n_wavelet_prior[int(n_wavelet)-1]/n_wavelet_prior[int(n_wavelet)])
        samples_new = np.zeros(samples.shape[2])
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Ohhhhhhhhhhhhh")
            samples_new[0] = n_wavelet-1
            samples_new[1] = n_glitch
            samples_new[2:2+(n_wavelet-1)*10] = new_point[:(n_wavelet-1)*10]
            samples_new[2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[(n_wavelet-1)*10:(n_wavelet-1)*10+n_glitch*6]
            samples_new[2+max_n_wavelet*10+max_n_glitch*6:] = new_point[(n_wavelet-1)*10+n_glitch*6:]
            #a_yes[2,j] += 1
            #log_likelihood[j,i+1] = log_L
            yes_or_no = 1
        else:
            #samples[j,i+1,:] = samples[j,i,:]
            samples_new = np.copy(samples[j,i,:])
            #a_no[2,j] += 1
            log_L = log_likelihood[j,i]
            yes_or_no = 0

        return samples_new, log_L, yes_or_no


################################################################################
#
#GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, log_likelihood, parallel):
    #print("TAU-GLOBAL")
    tau_scan = tau_scan_data['tau_scan']
    #print(i)
    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max
    #print(tau_scan_limit)

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']

    #print(len(tau_scan))
    #print(taus/(365.25*24*3600))
    #print(np.array(TAU_list)/(365.25*24*3600))

    #print("------------")
    #print(tau_scan[0].shape)
    #print(F0_list[0])
    #print(T0_list[0])
    
    #print(samples[0,i,:])
    #print(i)

    if parallel:
        num_cores = multiprocessing.cpu_count()
        rrr = Parallel(n_jobs=num_cores)(delayed(tau_scan_move_core)(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood) for j in range(n_chain))
        samples_new, log_L, yes_or_no = zip(*rrr)
        #print(samples_new)
    else:
        rrr = [tau_scan_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood) for j in range(n_chain)]
        samples_new, log_L, yes_or_no = zip(*rrr)

    for j in range(n_chain):
        samples[j,i+1,:] = samples_new[j]
        log_likelihood[j,i+1] = log_L[j]
        if yes_or_no[j]==1:
            a_yes[3,j] += 1
        else:
            a_no[3,j] += 1

def tau_scan_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, tau_scan, tau_scan_limit, TAU_list, F0_list, T0_list, log_likelihood):
    #check if there's any wavelet -- stay at given point if not
    n_wavelet = get_n_wavelet(samples, j, i)
    n_glitch = get_n_glitch(samples, j, i)
    #if j==0:
    #    print(n_wavelet)
    #    print(n_glitch)
    #    print(samples[j,i,:])
    #print(n_wavelet)
    if n_wavelet==0:
        samples_new = np.copy(samples[j,i,:])
        #a_no[3,j]+=1
        #print("No source to vary!")
        yes_or_no = 0
        log_L = log_likelihood[j,i]

    if include_gwb:
        gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
    else:
        gwb_on = 0

    log_f0_max = float(ptas[n_wavelet][0][gwb_on].params[3]._typename.split('=')[2][:-1])
    log_f0_min = float(ptas[n_wavelet][0][gwb_on].params[3]._typename.split('=')[1].split(',')[0])
    t0_max = float(ptas[n_wavelet][0][gwb_on].params[8]._typename.split('=')[2][:-1])
    t0_min = float(ptas[n_wavelet][0][gwb_on].params[8]._typename.split('=')[1].split(',')[0])
    tau_max = float(ptas[n_wavelet][0][gwb_on].params[9]._typename.split('=')[2][:-1])
    tau_min = float(ptas[n_wavelet][0][gwb_on].params[9]._typename.split('=')[1].split(',')[0])

    accepted = False
    while accepted==False:
        log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
        t0_new = np.random.uniform(low=t0_min, high=t0_max)
        tau_new = np.random.uniform(low=tau_min, high=tau_max)

        tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
        f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
        t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

        #print(tau_new, t0_new, 10**log_f0_new)
        #print(tau_idx, t0_idx, f0_idx)

        tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
        #print(tau_scan_new_point/tau_scan_limit)
        if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
            accepted = True
            #print("Yeeeh!")

    #randomly select other parameters (except sky location and psi, which we won't change here)
    #cos_gwtheta_new = ptas[-1][gwb_on].params[0].sample()
    cos_gwtheta_old = np.copy(samples[j,i,2+0])
    gwphi_old = np.copy(samples[j,i,2+2])
    psi_old = np.copy(samples[j,i,2+1])
    log10_h_new = ptas[-1][0][gwb_on].params[4].sample()
    log10_h_cross_new = ptas[-1][0][gwb_on].params[5].sample()
    phase0_new = ptas[-1][0][gwb_on].params[6].sample()
    phase0_cross_new = ptas[-1][0][gwb_on].params[7].sample()

    wavelet_select = np.random.randint(n_wavelet)

    samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
    new_point = np.copy(samples_current)
    new_point[wavelet_select*10:(wavelet_select+1)*10] = np.array([cos_gwtheta_old, psi_old, gwphi_old, log_f0_new,
                                                                 log10_h_new, log10_h_cross_new, phase0_new, phase0_cross_new, t0_new, tau_new])

    log_acc_ratio = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
    log_L = log_acc_ratio
    log_acc_ratio += ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point)
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

    #getting ratio of proposal densities!
    tau_old = samples[j,i,2+9+wavelet_select*10]
    f0_old = 10**samples[j,i,2+3+wavelet_select*10]
    t0_old = samples[j,i,2+8+wavelet_select*10]

    tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
    f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
    t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

    tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]
    
    log10_h_old = samples[j,i,2+4+wavelet_select*10]
    log10_h_cross_old = samples[j,i,2+5+wavelet_select*10]
    hastings_extra_factor = ptas[-1][0][gwb_on].params[4].get_pdf(log10_h_old) / ptas[-1][0][gwb_on].params[4].get_pdf(log10_h_new)
    hastings_extra_factor *= ptas[-1][0][gwb_on].params[5].get_pdf(log10_h_cross_old) / ptas[-1][0][gwb_on].params[5].get_pdf(log10_h_cross_new)

    acc_ratio = np.exp(log_acc_ratio)*(tau_scan_old_point/tau_scan_new_point) * hastings_extra_factor
    samples_new = np.zeros(samples.shape[2])
    if np.random.uniform()<=acc_ratio:
        samples_new[0] = n_wavelet
        samples_new[1] = n_glitch
        samples_new[2:2+n_wavelet*10] = new_point[:n_wavelet*10]
        samples_new[2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
        samples_new[2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
        #a_yes[3,j]+=1
        #log_likelihood[j,i+1] = log_L
        yes_or_no = 1
    else:
        samples_new = np.copy(samples[j,i,:])
        #a_no[3,j]+=1
        log_L = log_likelihood[j,i]
        yes_or_no = 0

    return samples_new, log_L, yes_or_no


################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################
def regular_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, log_likelihood, parallel):
    #print("FISHER")

    if parallel:
        num_cores = multiprocessing.cpu_count()
        rrr = Parallel(n_jobs=num_cores)(delayed(regular_move_core)(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, log_likelihood) for j in range(n_chain))
        samples_new, log_L, yes_or_no = zip(*rrr)
        #print(samples_new)
    else:
        rrr = [regular_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, log_likelihood) for j in range(n_chain)]
        samples_new, log_L, yes_or_no = zip(*rrr)

    for j in range(n_chain):
        samples[j,i+1,:] = samples_new[j]
        log_likelihood[j,i+1] = log_L[j]
        if yes_or_no[j]==1:
            a_yes[5,j] += 1
        else:
            a_no[5,j] += 1

def regular_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, log_likelihood):
    n_wavelet = get_n_wavelet(samples, j, i)
    n_glitch = get_n_glitch(samples, j, i)
    #if j==0:
    #    print(n_wavelet)
    #    print(n_glitch)

    if include_gwb:
        gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
    else:
        gwb_on = 0

    samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

    #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
    #case #1: we can vary any of them
    if n_wavelet!=0 and n_glitch!=0 and (gwb_on==1 or vary_rn):
        vary_decide = np.random.uniform()
        if vary_decide <= 1.0/3.0:
            what_to_vary = 'WAVE'
        elif vary_decide <= 2.0/3.0:
            what_to_vary = 'GLITCH'
        else:
            what_to_vary = 'GWB'
    #case #2: whe can vary two of them
    elif n_glitch!=0 and (gwb_on==1 or vary_rn):
        vary_decide = np.random.uniform()
        if vary_decide <= 0.5:
            what_to_vary = 'GLITCH'
        else:
            what_to_vary = 'GWB'
    elif n_wavelet!=0 and (gwb_on==1 or vary_rn):
        vary_decide = np.random.uniform()
        if vary_decide <= 0.5:
            what_to_vary = 'WAVE'
        else:
            what_to_vary = 'GWB'
    elif n_wavelet!=0 and n_glitch!=0:
        vary_decide = np.random.uniform()
        if vary_decide <= 0.5:
            what_to_vary = 'GLITCH'
        else:
            what_to_vary = 'WAVE'
    #case #3: we can only vary one of them
    elif n_wavelet!=0:
        what_to_vary = 'WAVE'
    elif n_glitch!=0:
        what_to_vary = 'GLITCH'
    elif gwb_on==1 or vary_rn:
        what_to_vary = 'GWB'
    #case #4: nothing to vary
    else:
        samples_new = np.copy(samples[j,i,:])
        yes_or_no = 0
        log_L = log_likelihood[j,i]
        #print("Nothing to vary!")
        return samples_new, log_L, yes_or_no

    if what_to_vary == 'WAVE':
        wavelet_select = np.random.randint(n_wavelet)
        jump_select = np.random.randint(10)
        jump_1wavelet = eig[j,wavelet_select,jump_select,:]
        jump = np.zeros(samples_current.size)
        #change intrinsic (and extrinsic) parameters of selected wavelet
        jump[wavelet_select*10:(wavelet_select+1)*10] = jump_1wavelet
        #and change sky location and polarization angle of all wavelets
        for which_wavelet in range(n_wavelet):
            jump[which_wavelet*10:which_wavelet*10+3] = jump_1wavelet[:3]
        #print('cw')
        #print(jump)
    elif what_to_vary == 'GLITCH':
        glitch_select = np.random.randint(n_glitch)
        jump_select = np.random.randint(6)
        jump_1glitch = eig_glitch[j,glitch_select,jump_select,:]
        jump = np.zeros(samples_current.size)
        #print(jump.shape)
        #print(jump[n_wavelet*8+glitch_select*6:n_wavelet*8+(glitch_select+1)*6].shape)
        #print(jump_1glitch.shape)
        jump[n_wavelet*10+glitch_select*6:n_wavelet*10+(glitch_select+1)*6] = jump_1glitch
    elif what_to_vary == 'GWB':
        if include_gwb:
            jump_select = np.random.randint(3)
        else:
            jump_select = np.random.randint(2)
        jump_gwb = eig_gwb_rn[j,jump_select,:]
        if gwb_on==0 and include_gwb:
            jump_gwb[-1] = 0
        if include_gwb:
            jump = np.array([jump_gwb[int(i-n_wavelet*10-n_glitch*6-num_per_psr_params)] if i>=n_wavelet*10+n_glitch*6+num_per_psr_params and i<n_wavelet*10+n_glitch*6+num_noise_params+1 else 0.0 for i in range(samples_current.size)])
        else:
            jump = np.array([jump_gwb[int(i-n_wavelet*10-n_glitch*6-num_per_psr_params)] if i>=n_wavelet*10+n_glitch*6+num_per_psr_params and i<n_wavelet*10+n_glitch*6+num_noise_params else 0.0 for i in range(samples_current.size)])
        #if j==0: print('gwb+rn')
        #if j==0: print(i)
        #if j==0: print(jump)

    new_point = samples_current + jump*np.random.normal()

    #if j==0:
    #    print(samples_current)
    #    print(new_point)

    log_acc_ratio = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
    log_L = log_acc_ratio
    log_acc_ratio += ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point)
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

    acc_ratio = np.exp(log_acc_ratio)
    #if j==0: print(acc_ratio)
    samples_new = np.zeros(samples.shape[2])
    if np.random.uniform()<=acc_ratio:
        #if j==0: print("ohh jeez")
        samples_new[0] = n_wavelet
        samples_new[1] = n_glitch
        samples_new[2:2+n_wavelet*10] = new_point[:n_wavelet*10]
        samples_new[2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
        samples_new[2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
        #a_yes[5,j]+=1
        #log_likelihood[j,i+1] = log_L
        yes_or_no = 1
    else:
        samples_new = np.copy(samples[j,i,:])
        #a_no[5,j]+=1
        log_L = log_likelihood[j,i]
        yes_or_no = 0

    return samples_new, log_L, yes_or_no


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params, log_likelihood):
    #print("SWAP")
    swap_chain = np.random.randint(n_chain-1)

    n_wavelet1 = get_n_wavelet(samples, swap_chain, i)
    n_wavelet2 = get_n_wavelet(samples, swap_chain+1, i)

    n_glitch1 = get_n_glitch(samples, swap_chain, i)
    n_glitch2 = get_n_glitch(samples, swap_chain+1, i)

    if include_gwb:
        gwb_on1 = get_gwb_on(samples, swap_chain, i, max_n_wavelet, max_n_glitch, num_noise_params)
        gwb_on2 = get_gwb_on(samples, swap_chain+1, i, max_n_wavelet, max_n_glitch, num_noise_params)
    else:
        gwb_on1 = 0
        gwb_on2 = 0

    samples_current1 = strip_samples(samples, swap_chain, i, n_wavelet1, max_n_wavelet, n_glitch1, max_n_glitch)
    samples_current2 = strip_samples(samples, swap_chain+1, i, n_wavelet2, max_n_wavelet, n_glitch2, max_n_glitch)

    log_acc_ratio = -ptas[n_wavelet1][n_glitch1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain]
    log_acc_ratio += -ptas[n_wavelet2][n_glitch2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain+1]
    log_acc_ratio += ptas[n_wavelet2][n_glitch2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain]
    log_acc_ratio += ptas[n_wavelet1][n_glitch1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain+1]

    #print(samples_current1)
    #print(samples_current2)

    acc_ratio = np.exp(log_acc_ratio)
    if np.random.uniform()<=acc_ratio:
        for j in range(n_chain):
            if j==swap_chain:
                samples[j,i+1,:] = samples[j+1,i,:]
                log_likelihood[j,i+1] = ptas[n_wavelet2][n_glitch2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain]
            elif j==swap_chain+1:
                samples[j,i+1,:] = samples[j-1,i,:]
                log_likelihood[j,i+1] = ptas[n_wavelet1][n_glitch1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain+1]
            else:
                samples[j,i+1,:] = samples[j,i,:]
                log_likelihood[j,i+1] = log_likelihood[j,i]
        a_yes[4,swap_chain]+=1
        swap_record.append(swap_chain)
    else:
        for j in range(n_chain):
            samples[j,i+1,:] = samples[j,i,:]
            log_likelihood[j,i+1] = log_likelihood[j,i]
        a_no[4,swap_chain]+=1

################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################
def noise_jump(n_chain, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig_per_psr, include_gwb, num_noise_params, vary_white_noise, log_likelihood, parallel):
    print("NOISE")
    
    if parallel:
        num_cores = multiprocessing.cpu_count()
        rrr = Parallel(n_jobs=num_cores)(delayed(noise_move_core)(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig_per_psr, include_gwb, num_noise_params, vary_white_noise, log_likelihood) for j in range(n_chain))
        samples_new, log_L, yes_or_no = zip(*rrr)
        #print(samples_new)
    else:
        rrr = [noise_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig_per_psr, include_gwb, num_noise_params, vary_white_noise, log_likelihood) for j in range(n_chain)]
        samples_new, log_L, yes_or_no = zip(*rrr)

    for j in range(n_chain):
        samples[j,i+1,:] = samples_new[j]
        log_likelihood[j,i+1] = log_L[j]
        if yes_or_no[j]==1:
            a_yes[6,j] += 1
        else:
            a_no[6,j] += 1

def noise_move_core(j, max_n_wavelet, max_n_glitch, ptas, samples, i, Ts, a_yes, a_no, eig_per_psr, include_gwb, num_noise_params, vary_white_noise, log_likelihood):
    n_wavelet = get_n_wavelet(samples, j, i)
    n_glitch = get_n_glitch(samples, j, i)
    #if j==0:
    #    print(n_wavelet)
    #    print(n_glitch)

    if include_gwb:
        gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
    else:
        gwb_on = 0

    samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

    #do the wn jump
    jump_select = np.random.randint(eig_per_psr.shape[1])
    #print(jump_select)
    jump_wn = eig_per_psr[j,jump_select,:]
    jump = np.array([jump_wn[int(i-n_wavelet*10-n_glitch*6)] if i>=n_wavelet*10+n_glitch*6 and i<n_wavelet*10+n_glitch*6+eig_per_psr.shape[1] else 0.0 for i in range(samples_current.size)])
    #if j==0: print('noise')
    #if j==0: print(jump)

    new_point = samples_current + jump*np.random.normal()

    log_acc_ratio = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)/Ts[j]
    log_L = log_acc_ratio
    log_acc_ratio += ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point)
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
    log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

    acc_ratio = np.exp(log_acc_ratio)
    #if j==0: print(acc_ratio)
    samples_new = np.zeros(samples.shape[2])
    if np.random.uniform()<=acc_ratio:
        #if j==0: print("Ohhhh")
        samples_new[0] = n_wavelet
        samples_new[1] = n_glitch
        samples_new[2:2+n_wavelet*10] = new_point[:n_wavelet*10]
        samples_new[2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
        samples_new[2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
        #a_yes[6,j]+=1
        #log_likelihood[j,i+1] = log_L
        yes_or_no = 1
    else:
        samples_new = np.copy(samples[j,i,:])
        #a_no[6,j]+=1
        log_L = log_likelihood[j,i]
        yes_or_no = 0

    return samples_new, log_L, yes_or_no


################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, T_chain=1, epsilon=1e-4, n_wavelet=1, dim=10, offset=0, use_prior=False, parallel=False):
    n_source=n_wavelet
    fisher = np.zeros((n_source,dim,dim))
    eig = []

    #print(params)

    #lnlikelihood at specified point
    if use_prior:
        nn = pta.get_lnlikelihood(params) + pta.get_lnprior(params)
    else:
        nn = pta.get_lnlikelihood(params)
    
    
    for k in range(n_source):
        #print(k)
        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[offset+i+k*dim] += 2*epsilon
            paramsMM[offset+i+k*dim] -= 2*epsilon
            #print(params)
            #print(paramsPP)
            
            #lnlikelihood at +-epsilon positions
            if use_prior:
                pp = pta.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
            else:
                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)

            #print(pp, nn, mm)
            
            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            #print('diagonal')
            #print(pp,nn,mm)
            #print(-(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon))
            fisher[k,i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

        #calculate off-diagonal elements
        for i in range(dim-1):
            if parallel:
                num_cores = multiprocessing.cpu_count()
                qqq = Parallel(n_jobs=num_cores)(delayed(fisher_core)(params, pta, epsilon, offset, dim, use_prior, k, i, j) for j in range(i+1,dim))
                #print(qqq)
                pp, mp, pm, mm = zip(*qqq)
            else:
                qqq = [fisher_core(params, pta, epsilon, offset, dim, use_prior, k, i, j) for j in range(i+1,dim)]
                pp, mp, pm, mm = zip(*qqq)

            #print('off-diagonal')
            #print(i)
            #print(pp,mp,pm,mm)
            #print(-(pp - mp - pm + mm)/(4.0*epsilon*epsilon))
            for j in range(i+1,dim):
                #print(j)
                fisher[k,i,j] = -(pp[j-(i+1)] - mp[j-(i+1)] - pm[j-(i+1)] + mm[j-(i+1)])/(4.0*epsilon*epsilon)
                fisher[k,j,i] = -(pp[j-(i+1)] - mp[j-(i+1)] - pm[j-(i+1)] + mm[j-(i+1)])/(4.0*epsilon*epsilon)
        
        #print(fisher)
        #correct for the given temperature of the chain    
        fisher = fisher/T_chain
     
        #print("---")
        #print(fisher)

        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            #UPDATED so that 0s are also replaced with 1.0
            FISHER = np.where(np.isfinite(fisher[k,:,:]) * (fisher[k,:,:]!=0.0), fisher[k,:,:], 1.0)
            if not np.array_equal(FISHER, fisher[k,:,:]):
                print("Changed some nan elements in the Fisher matrix to 1.0")

            #Find eigenvalues and eigenvectors of the Fisher matrix
            w, v = np.linalg.eig(FISHER)

            #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
            eig_limit = 1.0

            W = np.where(np.abs(w)>eig_limit, w, eig_limit)
            #print(W)
            #print(np.sum(v**2, axis=0))
            #if T_chain==1.0: print(W)
            #if T_chain==1.0: print(v)

            eig.append( (np.sqrt(1.0/np.abs(W))*v).T )
            #print(np.sum(eig**2, axis=1))
            #if T_chain==1.0: print(eig)

        except:
            print("An Error occured in the eigenvalue calculation")
            eig.append( np.array(False) )

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(np.log10(np.abs(np.real(np.array(FISHER)))))
        #plt.imshow(np.real(np.array(FISHER)))
        #plt.colorbar()
        
        #plt.figure()
        #plt.imshow(np.log10(np.abs(np.real(np.array(eig)[0,:,:]))))
        #plt.imshow(np.real(np.array(eig)[0,:,:]))
        #plt.colorbar()
    
    return np.array(eig)

def fisher_core(params, pta, epsilon, offset, dim, use_prior, k, i, j):
    #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
    paramsPP = np.copy(params)
    paramsMM = np.copy(params)
    paramsPM = np.copy(params)
    paramsMP = np.copy(params)

    paramsPP[offset+i+k*dim] += epsilon
    paramsPP[offset+j+k*dim] += epsilon
    paramsMM[offset+i+k*dim] -= epsilon
    paramsMM[offset+j+k*dim] -= epsilon
    paramsPM[offset+i+k*dim] += epsilon
    paramsPM[offset+j+k*dim] -= epsilon
    paramsMP[offset+i+k*dim] -= epsilon
    paramsMP[offset+j+k*dim] += epsilon

    #lnlikelihood at those positions
    if use_prior:
        pp = pta.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
        mm = pta.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
        pm = pta.get_lnlikelihood(paramsPM) + pta.get_lnprior(paramsPM)
        mp = pta.get_lnlikelihood(paramsMP) + pta.get_lnprior(paramsMP)
    else:
        pp = pta.get_lnlikelihood(paramsPP)
        mm = pta.get_lnlikelihood(paramsMM)
        pm = pta.get_lnlikelihood(paramsPM)
        mp = pta.get_lnlikelihood(paramsMP)

    #calculate off-diagonal elements of the Hessian from a central finite element scheme
    #note the minus sign compared to the regular Hessian
    #print('off-diagonal')
    #print(pp,mp,pm,mm)
    #print(-(pp - mp - pm + mm)/(4.0*epsilon*epsilon))
    return pp, mp, pm, mm


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################
def get_ptas(pulsars, vary_white_noise=True, include_equad_ecorr=False, wn_backend_selection=False, noisedict_file=None, include_rn=True, vary_rn=True, include_per_psr_rn=False, vary_per_psr_rn=False, include_gwb=True, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-13.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], per_psr_rn_amp_prior='uniform', per_psr_rn_log_amp_range=[-18,-11], prior_recovery=False, max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], t0_max=10.0, TF_prior=None, tref=53000*86400):
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        if include_equad_ecorr:
            equad = parameter.Uniform(-8.5, -5)
            ecorr = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant(efac_start)
        if include_equad_ecorr:
            equad = parameter.Constant()
            ecorr = parameter.Constant()

    if wn_backend_selection:
        selection = selections.Selection(selections.by_backend)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        if include_equad_ecorr:
            eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    else:
        ef = white_signals.MeasurementNoise(efac=efac)
        if include_equad_ecorr:
            eq = white_signals.EquadNoise(log10_equad=equad)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr)

    tm = gp_signals.TimingModel(use_svd=True)

    base_model = ef + tm
    if include_equad_ecorr:
        base_model = base_model + eq + ec

    #adding per psr RN if included
    if include_per_psr_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)

        if vary_per_psr_rn:
            if per_psr_rn_amp_prior == 'uniform':
                log10_A = parameter.LinearExp(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])
            elif per_psr_rn_amp_prior == 'log-uniform':
                log10_A = parameter.Uniform(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])

            gamma = parameter.Uniform(0, 7)
        else:
            log10_A = parameter.Constant()
            gamma = parameter.Constant()
        
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        per_psr_rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

        base_model = base_model + per_psr_rn
    
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

        #base_model_gwb = base_model + gwb

    #wavelet models
    wavelets = []
    for i in range(max_n_wavelet):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), -7)(str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)(str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'gwphi')
        psi = parameter.Uniform(0, np.pi)(str(i)+'_'+'gw_psi')
        phase0 = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'phase0')
        phase0_cross = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'phase0_cross')
        tau = parameter.Uniform(0.2, 5)(str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, t0_max)(str(i)+'_'+'t0')
        if wavelet_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h')
            log10_h_cross = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h_cross')
        elif wavelet_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h')
            log10_h_cross = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])(str(i)+'_'+'log10_h_cross')
        else:
            print("CW amplitude prior of {0} not available".format(cw_amp_prior))
        wavelet_wf = models.wavelet_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_h = log10_h, log10_h2=log10_h_cross,
                                          tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, phase02=phase0_cross,
                                          epsilon = None, psi=psi, tref=tref)
        wavelets.append(deterministic_signals.Deterministic(wavelet_wf, name='wavelet'+str(i)))

    #glitch models
    glitches = []
    for i in range(max_n_glitch):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), -7)("Glitch_"+str(i)+'_'+'log10_f0')
        phase0 = parameter.Uniform(0, 2*np.pi)("Glitch_"+str(i)+'_'+'phase0')
        tau = parameter.Uniform(0.2, 5)("Glitch_"+str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, t0_max)("Glitch_"+str(i)+'_'+'t0')
        psr_idx = parameter.Uniform(-0.5, len(pulsars)-0.5)("Glitch_"+str(i)+'_'+'psr_idx')
        if glitch_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(glitch_log_amp_range[0], glitch_log_amp_range[1])("Glitch_"+str(i)+'_'+'log10_h')
        elif glitch_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(glitch_log_amp_range[0], glitch_log_amp_range[1])("Glitch_"+str(i)+'_'+'log10_h')
        else:
            print("Glitch amplitude prior of {0} not available".format(glitch_amp_prior))
        glitch_wf = models.glitch_delay(log10_h = log10_h, tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, tref=tref,
                                        psr_float_idx = psr_idx, pulsars=pulsars)
        glitches.append(deterministic_signals.Deterministic(glitch_wf, name='Glitch'+str(i) ))
    
    gwb_options = [False,]
    if include_gwb:
        gwb_options.append(True)

    ptas = []
    for n_wavelet in range(max_n_wavelet+1):
        glitch_sub_ptas = []
        for n_glitch in range(max_n_glitch+1):
            gwb_sub_ptas = []
            for gwb_o in gwb_options:
                #setting up the proper model
                s = base_model

                if gwb_o:
                    s += gwb
                for i in range(n_glitch):
                    s = s + glitches[i]
                for i in range(n_wavelet):
                    s = s + wavelets[i]

                model = []
                for p in pulsars:
                    model.append(s(p))

                #set the likelihood to unity if we are in prior recovery mode
                if prior_recovery:
                    if TF_prior is None:
                        gwb_sub_ptas.append(get_prior_recovery_pta(signal_base.PTA(model)))
                    else:
                        gwb_sub_ptas.append(get_tf_prior_pta(signal_base.PTA(model), TF_prior, n_wavelet, prior_recovery=True))
                elif noisedict_file is not None:
                    with open(noisedict_file, 'r') as fp:
                        noisedict = json.load(fp)
                        pta = signal_base.PTA(model)
                        pta.set_default_params(noisedict)
                        if TF_prior is None:
                            gwb_sub_ptas.append(pta)
                        else:
                            gwb_sub_ptas.append(get_tf_prior_pta(pta, TF_prior, n_wavelet))
                else:
                    if TF_prior is None:
                        gwb_sub_ptas.append(signal_base.PTA(model))
                    else:
                        gwb_sub_ptas.append(get_tf_prior_pta(signal_base.PTA(model), TF_prior, n_wavelet))

            glitch_sub_ptas.append(gwb_sub_ptas)

        ptas.append(glitch_sub_ptas)

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

################################################################################
#
#MAKE PTA OBJECT WITH CUSTOM T0-F0 PRIOR FOR ZOOM-IN RERUNS
#
################################################################################
def get_tf_prior_pta(pta, TF_prior, n_wavelet, prior_recovery=False):
    class tf_prior_pta:
        def __init__(self, pta):
            self.pta = pta
            self.params = pta.params
            self.pulsars = pta.pulsars
            self.summary = pta.summary
        def get_lnlikelihood(self, x):
            if prior_recovery:
                return 0.0
            else:
                return self.pta.get_lnlikelihood(x)
        def get_lnprior(self, x):
            within_prior = True
            for i in range(n_wavelet):
                t0 = x[8+10*i]
                log10_f0 = x[3+10*i]
                #print(t0, log10_f0)
                t_idx = int( np.digitize(t0, TF_prior['t_bins']) )
                f_idx = int( np.digitize(log10_f0, TF_prior['lf_bins']) )
                #print((t_idx, f_idx))
                if (t_idx, f_idx) not in TF_prior['on_idxs']:
                    within_prior = False
            if within_prior:
                return self.pta.get_lnprior(x)
            else:
                return -np.inf

    return tf_prior_pta(pta)

################################################################################
#
#SOME HELPER FUNCTIONS
#
################################################################################
def get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params):
    return int(samples[j,i,2+max_n_wavelet*10+max_n_glitch*6+num_noise_params]!=0.0)

def strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch):
    return np.delete(samples[j,i,2:], list(range(n_wavelet*10,max_n_wavelet*10))+list(range(max_n_wavelet*10+n_glitch*6,max_n_wavelet*10+max_n_glitch*6)) )

def get_n_wavelet(samples, j, i):
    return int(samples[j,i,0])

def get_n_glitch(samples, j, i):
    return int(samples[j,i,1])
