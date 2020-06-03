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
from enterprise.signals import selections
from enterprise.signals.selections import Selection

import enterprise_wavelets as models
import pickle

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
            n_wavelet_prior = np.ones(max_n_wavelet+1)/(max_n_wavelet+1)
        else:
            n_wavelet_prior = np.array(n_wavelet_prior)
            n_wavelet_norm = np.sum(n_wavelet_prior)
            n_wavelet_prior *= 1.0/n_wavelet_norm
        print("Prior on number of wavelets: ", n_wavelet_prior)

    #setting up array for the samples
    num_params = max_n_wavelet*8+1
    if include_gwb:
        num_params += 1

    num_noise_params = 0
    if vary_white_noise:
        num_noise_params += len(pulsars)
    if vary_rn:
        num_noise_params += 2

    num_params += num_noise_params
    print(num_params)
    print(num_noise_params)

    samples = np.zeros((n_chain, N, num_params))

    #filling first sample with random draw
    for j in range(n_chain):
        if n_wavelet_start is 'random':
            n_wavelet = np.random.choice(max_n_wavelet+1)
        else:
            n_wavelet = n_wavelet_start
        samples[j,0,0] = n_wavelet
        print(n_wavelet)
        if n_wavelet!=0:
            #making sure all wavelets get the same sky location and ellipticity
            init_cos_gwtheta = ptas[n_wavelet][0].params[0].sample()
            init_epsilon = ptas[n_wavelet][0].params[1].sample()
            init_gwphi = ptas[n_wavelet][0].params[2].sample()
            for which_wavelet in range(n_wavelet):
                samples[j,0,1+0+which_wavelet*8] = init_cos_gwtheta
                samples[j,0,1+1+which_wavelet*8] = init_epsilon
                samples[j,0,1+2+which_wavelet*8] = init_gwphi
                #randomly pick other wavelet parameters separately fo each wavelet
                samples[j,0,1+3+which_wavelet*8:1+8+which_wavelet*8] = np.hstack(p.sample() for p in ptas[n_wavelet][0].params[3:8])
            #start from injected parameters for testing
            ####samples[j,0,1:n_wavelet*8+1] = np.array([0.0, 1.0, 0.0, -7.522, -5.0, 0.0, 2.738, 0.548])
        if vary_white_noise:
            samples[j,0,max_n_wavelet*8+1:max_n_wavelet*8+1+len(pulsars)] = np.ones(len(pulsars))*efac_start
        if vary_rn:
            samples[j,0,max_n_wavelet*8+1+len(pulsars):max_n_wavelet*8+1+num_noise_params] = np.array([ptas[n_wavelet][0].params[n_wavelet*8+num_noise_params-2].sample(), ptas[n_wavelet][0].params[n_wavelet*8+num_noise_params-1].sample()])
        if include_gwb:
            samples[j,0,max_n_wavelet*8+1+num_noise_params] = ptas[n_wavelet][1].params[n_wavelet*8+num_noise_params].sample()
    print(samples[0,0,:])
    print(np.delete(samples[0,0,1:], range(n_wavelet*8,max_n_wavelet*8)))
    print(ptas[int(samples[0,0,0])][0].get_lnlikelihood(np.delete(samples[0,0,1:], range(int(samples[0,0,0])*8,max_n_wavelet*8))))

    #setting up array for the fisher eigenvalues
    #one for wavelet parameters which we will keep updating
    eig = np.ones((n_chain, max_n_wavelet, 8, 8))*0.1

    #one for GWB and common rn parameters, which we will keep updating
    if include_gwb:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0], [0,0.3,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
    else:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()

    #and one for white noise parameters, which we will not update
    eig_wn = np.broadcast_to(np.eye(len(pulsars))*0.1, (n_chain,len(pulsars), len(pulsars)) ).copy()

    #calculate wn eigenvectors
    for j in range(n_chain):
        if include_gwb:
            wn_eigvec = get_fisher_eigenvectors(np.delete(samples[j,0,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][1], T_chain=Ts[j], n_wavelet=1, dim=len(pulsars), offset=n_wavelet*8)
        else:
            wn_eigvec = get_fisher_eigenvectors(np.delete(samples[j,0,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][0], T_chain=Ts[j], n_wavelet=1, dim=len(pulsars), offset=n_wavelet*8)
        #print(wn_eigvec)
        eig_wn[j,:,:] = wn_eigvec[0,:,:]

    #read in tau_scan data if we will need it
    if tau_scan_proposal_weight+RJ_weight>0:
        if tau_scan_file==None:
            raise Exception("tau-scan data file is needed for tau-scan global propsals")
        with open(tau_scan_file, 'rb') as f:
            tau_scan_data = pickle.load(f)
            print("Tau-scan data read in successfully!")
        
        tau_scan = tau_scan_data['tau_scan']
        
        TAU_list = list(tau_scan_data['tau_edges'])
        F0_list = tau_scan_data['f0_edges']
        T0_list = tau_scan_data['t0_edges']

        #check if same prior range was used
        log_f0_max = float(ptas[-1][-1].params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[-1][-1].params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[-1][-1].params[6]._typename.split('=')[2][:-1])
        t0_min = float(ptas[-1][-1].params[6]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[-1][-1].params[7]._typename.split('=')[2][:-1])
        tau_min = float(ptas[-1][-1].params[7]._typename.split('=')[1].split(',')[0])

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
 

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros(n_chain+2)
    a_no=np.zeros(n_chain+2)
    swap_record = []
    rj_record = []

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + tau_scan_proposal_weight +
                    RJ_weight + gwb_switch_weight + noise_jump_weight)
    swap_probability = PT_swap_weight/total_weight
    tau_scan_proposal_probability = tau_scan_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {3:.2f}%\nGWB-switches: {4:.2f}%\n\
Tau-scan-proposals: {1:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\nNoise jump: {5:.2f}%".format(swap_probability*100,
          tau_scan_proposal_probability*100, regular_probability*100,
          RJ_probability*100, gwb_switch_probability*100, noise_jump_probability*100))

    for i in range(int(N-1)):
        ########################################################
        #
        #print out run state every n_status_update iterations
        #
        ########################################################
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            if jupyter_notebook:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                      'Acceptance fraction (RJ, swap, each chain): ({0:1.2f}, {1:1.2f}, '.format(acc_fraction[0], acc_fraction[1]) +
                      ', '.join(['{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[2:]) +
                      ')' + '\r',end='')
            else:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                      'Acceptance fraction (RJ, swap, each chain): ({0:1.2f}, {1:1.2f}, '.format(acc_fraction[0], acc_fraction[1]) +
                      ', '.join(['{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[2:]) + ')')
        #################################################################################
        #
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        #
        #################################################################################
        if i%n_fish_update==0:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                for j in range(n_chain):
                    n_wavelet = int(np.copy(samples[j,i,0]))
                    if n_wavelet!=0:
                        if include_gwb:
                            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][gwb_on], T_chain=Ts[j], n_wavelet=1, dim=3, offset=n_wavelet*8+len(pulsars))
                        else:
                            gwb_on = 0
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][0], T_chain=Ts[j], n_wavelet=1, dim=2, offset=n_wavelet*8+len(pulsars))
                        eigenvectors = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][gwb_on], T_chain=Ts[j], n_wavelet=n_wavelet)
                        if np.all(eigenvectors):
                            eig[j,:n_wavelet,:,:] = eigenvectors
                        if np.all(eigvec_rn):
                            eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]
                    else:
                        if include_gwb:
                            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][gwb_on], T_chain=Ts[j], n_wavelet=1, dim=3, offset=n_wavelet*8+len(pulsars))
                        else:
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][0], T_chain=Ts[j], n_wavelet=1, dim=2, offset=n_wavelet*8+len(pulsars))
                        #check if eigenvector calculation was succesful
                        #if not, we just keep the initialized eig full of 0.1 values
                        if np.all(eigvec_rn):
                            eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]
            elif samples[0,i,0]!=0:
                n_wavelet = int(np.copy(samples[0,i,0]))
                if include_gwb:
                    gwb_on = int(samples[0,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
                else:
                    gwb_on = 0
                eigenvectors = get_fisher_eigenvectors(np.delete(samples[0,i,1:], range(n_wavelet*8,max_n_wavelet*8)), ptas[n_wavelet][gwb_on], T_chain=Ts[0], n_wavelet=n_wavelet)
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
        #PT swap move
        if jump_decide<swap_probability:
            do_pt_swap(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params)
        #global proposal based on tau_scan
        elif jump_decide<swap_probability+tau_scan_proposal_probability:
            do_tau_scan_global_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data)
        #do RJ move
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability):
            do_rj_move(n_chain, max_n_wavelet, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data)
        #do GWB switch move
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability):
            gwb_switch_move(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, gwb_on_prior, gwb_log_amp_range)
        #do noise jump
        elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability):
            noise_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, eig_wn, include_gwb, num_noise_params, vary_white_noise)
        #regular step
        else:
            regular_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_noise_params, vary_rn)
        #print(samples[0,i,:])


    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record, ptas

################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a wavelet
#
################################################################################
def do_rj_move(n_chain, max_n_wavelet, n_wavelet_prior, ptas, samples, i, Ts, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data):
    tau_scan = tau_scan_data['tau_scan']

    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max
    
    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']
    
    for j in range(n_chain):
        n_wavelet = int(np.copy(samples[j,i,0]))
        #if j==0: print(n_wavelet)

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_wavelet==0 or (direction_decide<add_prob and n_wavelet!=max_n_wavelet): #adding a wavelet------------------------------------------------------
            if j==0: rj_record.append(1)
            #if j==0: print("Propose to add a wavelet")

            log_f0_max = float(ptas[-1][gwb_on].params[3]._typename.split('=')[2][:-1])
            log_f0_min = float(ptas[-1][gwb_on].params[3]._typename.split('=')[1].split(',')[0])
            t0_max = float(ptas[-1][gwb_on].params[6]._typename.split('=')[2][:-1])
            t0_min = float(ptas[-1][gwb_on].params[6]._typename.split('=')[1].split(',')[0])
            tau_max = float(ptas[-1][gwb_on].params[7]._typename.split('=')[2][:-1])
            tau_min = float(ptas[-1][gwb_on].params[7]._typename.split('=')[1].split(',')[0])

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
            log10_h_new = ptas[-1][gwb_on].params[4].sample()
            phase0_new = ptas[-1][gwb_on].params[5].sample()
            #if this is the first wavelet, draw sky location and ellipticity too
            if n_wavelet==0:
                cos_gwtheta_new = ptas[-1][gwb_on].params[0].sample()
                gwphi_new = ptas[-1][gwb_on].params[2].sample()
                epsilon_new = ptas[-1][gwb_on].params[1].sample()
            #if this is not the first wavelet, copy sky location and ellipticity from existing wavelet(s)
            else:
                cos_gwtheta_new = np.copy(samples[j,i,1+0])
                gwphi_new = np.copy(samples[j,i,1+2])
                epsilon_new = np.copy(samples[j,i,1+1])

            prior_ext = (ptas[-1][gwb_on].params[0].get_pdf(cos_gwtheta_new) * ptas[-1][gwb_on].params[1].get_pdf(epsilon_new) *
                         ptas[-1][gwb_on].params[2].get_pdf(gwphi_new) * ptas[-1][gwb_on].params[4].get_pdf(log10_h_new) * 
                         ptas[-1][gwb_on].params[5].get_pdf(phase0_new))

            samples_current = np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8))

            new_point = np.delete(samples[j,i,1:], range((n_wavelet+1)*8,max_n_wavelet*8))
            new_wavelet = np.array([cos_gwtheta_new, epsilon_new, gwphi_new, log_f0_new, log10_h_new, phase0_new, t0_new, tau_new])
            new_point[n_wavelet*8:(n_wavelet+1)*8] = new_wavelet

            #if j==0:
            #    print(samples_current)
            #    print(new_point)

            log_acc_ratio = ptas[(n_wavelet+1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[(n_wavelet+1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnprior(samples_current)
           
            
            #apply normalization
            tau_scan_new_point_normalized = tau_scan_new_point/tau_scan_data['norm']

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==0:
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
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Yeeeh")
                samples[j,i+1,0] = n_wavelet+1
                samples[j,i+1,1:(n_wavelet+1)*8+1] = new_point[:(n_wavelet+1)*8]
                samples[j,i+1,max_n_wavelet*8+1:] = new_point[(n_wavelet+1)*8:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1

        elif n_wavelet==max_n_wavelet or (direction_decide>add_prob and n_wavelet!=0):   #removing a wavelet----------------------------------------------------------
            if j==0: rj_record.append(-1)
            #if j==0: print("Propose to remove a wavelet")
            #choose which wavelet to remove
            remove_index = np.random.randint(n_wavelet)

            samples_current = np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8))
            new_point = np.delete(samples_current, range(remove_index*8,(remove_index+1)*8))

            log_acc_ratio = ptas[(n_wavelet-1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[(n_wavelet-1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnprior(samples_current)

            #getting tau_scan at old point
            tau_old = samples[j,i,1+7+remove_index*8]
            f0_old = 10**samples[j,i,1+3+remove_index*8]
            t0_old = samples[j,i,1+6+remove_index*8]

            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

            tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]

            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/tau_scan_data['norm']

            #getting external parameter priors
            log10_h_old = np.copy(samples[j,i,1+4+remove_index*8])
            phase0_old = np.copy(samples[j,i,1+5+remove_index*8])
            cos_gwtheta_old = np.copy(samples[j,i,1+0+remove_index*8])
            gwphi_old = np.copy(samples[j,i,1+2+remove_index*8])
            epsilon_old = np.copy(samples[j,i,1+1+remove_index*8])

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
                

            prior_ext = (ptas[-1][gwb_on].params[0].get_pdf(cos_gwtheta_old) * ptas[-1][gwb_on].params[1].get_pdf(epsilon_old) *
                         ptas[-1][gwb_on].params[2].get_pdf(gwphi_old) * ptas[-1][gwb_on].params[4].get_pdf(log10_h_old) *
                         ptas[-1][gwb_on].params[5].get_pdf(phase0_old))

            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==1:
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
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Ohhhhhhhhhhhhh")
                samples[j,i+1,0] = n_wavelet-1
                samples[j,i+1,1:(n_wavelet-1)*8+1] = new_point[:(n_wavelet-1)*8]
                samples[j,i+1,max_n_wavelet*8+1:] = new_point[(n_wavelet-1)*8:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1



################################################################################
#
#GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################

def do_tau_scan_global_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, tau_scan_data):
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

    for j in range(n_chain):
        #check if there's any wavelet -- stay at given point of not
        n_wavelet = int(np.copy(samples[j,i,0]))
        #print(n_wavelet)
        if n_wavelet==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1
            #print("No source to vary!")
            continue

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        log_f0_max = float(ptas[n_wavelet][gwb_on].params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(ptas[n_wavelet][gwb_on].params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(ptas[n_wavelet][gwb_on].params[6]._typename.split('=')[2][:-1])
        t0_min = float(ptas[n_wavelet][gwb_on].params[6]._typename.split('=')[1].split(',')[0])
        tau_max = float(ptas[n_wavelet][gwb_on].params[7]._typename.split('=')[2][:-1])
        tau_min = float(ptas[n_wavelet][gwb_on].params[7]._typename.split('=')[1].split(',')[0])

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

        #randomly select other parameters (except sky location and epsilon, which we won't change here)
        #cos_gwtheta_new = ptas[-1][gwb_on].params[0].sample()
        cos_gwtheta_old = np.copy(samples[j,i,1+0])
        #gwphi_new = ptas[-1][gwb_on].params[2].sample()
        gwphi_old = np.copy(samples[j,i,1+2])
        phase0_new = ptas[-1][gwb_on].params[5].sample()
        #epsilon_new = ptas[-1][gwb_on].params[1].sample()
        epsilon_old = np.copy(samples[j,i,1+1])
        log10_h_new = ptas[-1][gwb_on].params[4].sample()

        wavelet_select = np.random.randint(n_wavelet)

        samples_current = np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8))
        new_point = np.copy(samples_current)
        new_point[wavelet_select*8:(wavelet_select+1)*8] = np.array([cos_gwtheta_old, epsilon_old, gwphi_old, log_f0_new,
                                                                     log10_h_new, phase0_new, t0_new, tau_new])

        log_acc_ratio = ptas[n_wavelet][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_wavelet][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnprior(samples_current)

        #getting ratio of proposal densities!
        tau_old = samples[j,i,1+7+wavelet_select*8]
        f0_old = 10**samples[j,i,1+3+wavelet_select*8]
        t0_old = samples[j,i,1+6+wavelet_select*8]

        tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
        f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
        t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

        tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]
        
        log10_h_old = samples[j,i,1+4+wavelet_select*8]
        hastings_extra_factor = ptas[-1][gwb_on].params[4].get_pdf(log10_h_old) / ptas[-1][gwb_on].params[4].get_pdf(log10_h_new)

        acc_ratio = np.exp(log_acc_ratio)*(tau_scan_old_point/tau_scan_new_point) * hastings_extra_factor
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1:n_wavelet*8+1] = new_point[:n_wavelet*8]
            samples[j,i+1,max_n_wavelet*8+1:] = new_point[n_wavelet*8:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1


################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################

def regular_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_noise_params, vary_rn):
    for j in range(n_chain):
        n_wavelet = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        samples_current = np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8))

        #decide if moving in wavelet parameters or GWB/RN parameters
        #case #1: we can vary both
        if n_wavelet!=0 and (gwb_on==1 or vary_rn):
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'WAVE'
            else:
                what_to_vary = 'GWB'
        #case #2: we can only vary wavelet parameters
        elif n_wavelet!=0:
            what_to_vary = 'WAVE'
        #case #3: we can only vary GWB or RN
        elif gwb_on==1 or vary_rn:
            what_to_vary = 'GWB'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1
            #print("Nothing to vary!")
            continue

        if what_to_vary == 'WAVE':
            wavelet_select = np.random.randint(n_wavelet)
            jump_select = np.random.randint(8)
            jump_1wavelet = eig[j,wavelet_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic parameters of selected wavelet
            jump[wavelet_select*8:(wavelet_select+1)*8] = jump_1wavelet
            #and change sky location and ellipticity of all wavelets
            for which_wavelet in range(n_wavelet):
                jump[which_wavelet*8:which_wavelet*8+3] = jump_1wavelet[:3]
            #print('cw')
            #print(jump)
        elif what_to_vary == 'GWB':
            if include_gwb:
                jump_select = np.random.randint(3)
            else:
                jump_select = np.random.randint(2)
            jump_gwb = eig_gwb_rn[j,jump_select,:]
            if gwb_on==0 and include_gwb:
                jump_gwb[-1] = 0
            if include_gwb:
                jump = np.array([jump_gwb[int(i-n_wavelet*8-len(ptas[n_wavelet][gwb_on].pulsars))] if i>=n_wavelet*8+len(ptas[n_wavelet][gwb_on].pulsars) and i<n_wavelet*8+num_noise_params+1 else 0.0 for i in range(samples_current.size)])
            else:
                jump = np.array([jump_gwb[int(i-n_wavelet*8-len(ptas[n_wavelet][gwb_on].pulsars))] if i>=n_wavelet*8+len(ptas[n_wavelet][gwb_on].pulsars) and i<n_wavelet*8+num_noise_params else 0.0 for i in range(samples_current.size)])
            #if j==0: print('gwb+rn')
            #if j==0: print(i)
            #if j==0: print(jump)
        new_point = samples_current + jump*np.random.normal()

        log_acc_ratio = ptas[n_wavelet][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_wavelet][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("ohh jeez")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1:n_wavelet*8+1] = new_point[:n_wavelet*8]
            samples[j,i+1,max_n_wavelet*8+1:] = new_point[n_wavelet*8:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params):
    swap_chain = np.random.randint(n_chain-1)

    n_wavelet1 = int(np.copy(samples[swap_chain,i,0]))
    n_wavelet2 = int(np.copy(samples[swap_chain+1,i,0]))

    if include_gwb:
        gwb_on1 = int(samples[swap_chain,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
        gwb_on2 = int(samples[swap_chain+1,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
    else:
        gwb_on1 = 0
        gwb_on2 = 0

    samples_current1 = np.delete(samples[swap_chain,i,1:], range(n_wavelet1*8,max_n_wavelet*8))
    samples_current2 = np.delete(samples[swap_chain+1,i,1:], range(n_wavelet2*8,max_n_wavelet*8))

    log_acc_ratio = -ptas[n_wavelet1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain]
    log_acc_ratio += -ptas[n_wavelet2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain+1]
    log_acc_ratio += ptas[n_wavelet2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain]
    log_acc_ratio += ptas[n_wavelet1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain+1]

    acc_ratio = np.exp(log_acc_ratio)
    if np.random.uniform()<=acc_ratio:
        for j in range(n_chain):
            if j==swap_chain:
                samples[j,i+1,:] = samples[j+1,i,:]
            elif j==swap_chain+1:
                samples[j,i+1,:] = samples[j-1,i,:]
            else:
                samples[j,i+1,:] = samples[j,i,:]
        a_yes[1]+=1
        swap_record.append(swap_chain)
    else:
        for j in range(n_chain):
            samples[j,i+1,:] = samples[j,i,:]
        a_no[1]+=1

################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################

def noise_jump(n_chain, max_n_wavelet, ptas, samples, i, Ts, a_yes, a_no, eig_wn, include_gwb, num_noise_params, vary_white_noise):
    for j in range(n_chain):
        n_wavelet = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_wavelet*8+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        samples_current = np.delete(samples[j,i,1:], range(n_wavelet*8,max_n_wavelet*8))

        #do the wn jump
        jump_select = np.random.randint(len(ptas[n_wavelet][gwb_on].pulsars))
        #print(jump_select)
        jump_wn = eig_wn[j,jump_select,:]
        jump = np.array([jump_wn[int(i-n_wavelet*8)] if i>=n_wavelet*8 and i<n_wavelet*8+len(ptas[n_wavelet][gwb_on].pulsars) else 0.0 for i in range(samples_current.size)])
        #if j==0: print('noise')
        #if j==0: print(jump)

        new_point = samples_current + jump*np.random.normal()

        log_acc_ratio = ptas[n_wavelet][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_wavelet][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_wavelet][gwb_on].get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Ohhhh")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1:n_wavelet*8+1] = new_point[:n_wavelet*8]
            samples[j,i+1,max_n_wavelet*8+1:] = new_point[n_wavelet*8:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1


################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, T_chain=1, epsilon=1e-4, n_wavelet=1, dim=8, offset=0, use_prior=False):
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
        for i in range(dim):
            for j in range(i+1,dim):
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
                fisher[k,i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[k,j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
        
        #print(fisher)
        #correct for the given temperature of the chain    
        fisher = fisher/T_chain
      
        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            FISHER = np.where(np.isfinite(fisher[k,:,:]), fisher[k,:,:], 1.0)
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


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################

def get_ptas(pulsars, vary_white_noise=True, include_rn=True, vary_rn=True, include_gwb=True, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-13.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], prior_recovery=False, max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11]):
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

        #base_model_gwb = base_model + gwb

    #wavelet models
    wavelets = []
    for i in range(max_n_wavelet):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), -7)(str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)(str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'gwphi')
        phase0 = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'phase0')
        epsilon = parameter.Uniform(0, 1.0)(str(i)+'_'+'epsilon')
        tau = parameter.Uniform(0.2, 5)(str(i)+'_'+'tau')
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

    #glitch models
    glitches = []
    for i in range(max_n_glitch):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), -7)("glitch_"+str(i)+'_'+'log10_f0')
        phase0 = parameter.Uniform(0, 2*np.pi)("glitch_"+str(i)+'_'+'phase0')
        tau = parameter.Uniform(0.2, 5)("glitch_"+str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, 10.0)("glitch_"+str(i)+'_'+'t0')
        psr_idx = parameter.Uniform(-0.5, len(pulsars)-0.5)("glitch_"+str(i)+'_'+'psr_idx')
        if glitch_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(glitch_log_amp_range[0], glitch_log_amp_range[1])("glitch_"+str(i)+'_'+'log10_h')
        elif glitch_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(glitch_log_amp_range[0], glitch_log_amp_range[1])("glitch_"+str(i)+'_'+'log10_h')
        else:
            print("Glitch amplitude prior of {0} not available".format(glitch_amp_prior))
        glitch_wf = models.glitch_delay(log10_h = log10_h, tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, tref=53000*86400,
                                        psr_float_idx = psr_idx, pulsars=pulsars)
        glitches.append(deterministic_signals.Deterministic(glitch_wf, name='glitch'+str(i) ))
    
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
                    gwb_sub_ptas.append(get_prior_recovery_pta(signal_base.PTA(model)))
                else:
                    gwb_sub_ptas.append(signal_base.PTA(model))

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

