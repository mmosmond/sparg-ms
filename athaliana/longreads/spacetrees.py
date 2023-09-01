from scipy.optimize import minimize
import time
import numpy as np
import math

def mle_dispersal_tree(locations, shared_times_inverted):
    """
    Maximum likelihood estimate of dispersal rate given locations and (inverted) shared times between lineages in a tree.
    """

    return np.matmul(np.matmul(np.transpose(locations), shared_times_inverted), locations) / len(locations)

def mle_dispersal_numeric(locations, shared_times_inverted, log_det_shared_times,
                          sigma0=None, bnds=None, method='L-BFGS-B', callbackF=None,
                          important=False, branching_times=None, phi0=None, scale_phi=None, logpcoals=None,
                          quiet=False):
    """
    Numerically estimate maximum likelihood dispersal rate (and possibly branching rate) given sample locations and shared_times.
    """

    L,M,_,_ = shared_times_inverted.shape
    n, d = locations.shape
    if not quiet: print('number of loci:',L,'\nnumber of trees per locus:',M,'\nnumber of samples:',n,'\nnumber of spatial dimensions:',d)

    # initial guess at parameters
    if sigma0 is None:
        if not quiet: print('averaging dispersal rate over all trees to initialize search...')
        sigma0 = np.mean([mle_dispersal_tree(locations, st) for sts in shared_times_inverted for st in sts], axis=0) #average over all trees and loci
    x0 = _sigma_to_sds_rho(sigma0) #convert to standard deviations and correlation

    if important:
        if phi0 is None:
            phi0 = np.log(n)*np.mean([1/bts[-1] for bts in branching_times]) #initial guess at branching rate (exponential growth)
        if scale_phi is None:
            scale_phi = x0[0]/phi0 #we will search for the value of phi*scale_phi that maximizes the likelihood (putting phi on same scale as dispersal accelarates search) 
        if not quiet: print('multiplying branching rate by:',scale_phi)
        x0.append(phi0*scale_phi)
        
    # negative composite log likelihood ratio, as function of x
    f = _sum_mc(locations=locations, shared_times_inverted=shared_times_inverted, log_det_shared_times=log_det_shared_times,
                important=important, branching_times=branching_times, scale_phi=scale_phi, logpcoals=logpcoals)

    # impose bounds on parameters
    if bnds is None:
        bnds = [(1e-6,None),(1e-6,None),(-0.99,0.99)] #FIX: assumes 2d
    if important:
        bnds.append((1e-6,None))
    bnds = tuple(bnds)

    # find mle
    if not quiet: print('\nsearching for maximum likelihood parameters...')
    if callbackF is not None: callbackF(x0)
    t0 = time.time()
    m = minimize(f, x0=x0, bounds=bnds, method=method, callback=callbackF) #find MLE
    if not quiet: print('\nfinding the max took', time.time()-t0, 'seconds')

    if not quiet:
        mle = m.x
        sigma = _sds_rho_to_sigma(mle[0],mle[1],mle[2]) #convert to covariance matrix #FIX: assumes 2d
        print('\nmaximum likelihood dispersal rate:\n',sigma)
        phi = mle[-1]/scale_phi #unscale phi
        print('\nmaximum likelihood branching rate:',phi)

    return m

def _sigma_to_sds_rho(sigma):
    """
    Convert 2x2 covariance matrix to sds and correlation
    """
    
    sdx = sigma[0,0]**0.5
    sdy = sigma[1,1]**0.5
    rho = sigma[0,1]/(sdx * sdy) #note that small sdx and sdy will raise errors
    
    return [sdx, sdy, rho]

def _sum_mc(locations, shared_times_inverted, log_det_shared_times,
            important=False, branching_times=None, scale_phi=None, logpcoals=None):
    """
    Negative log composite likelihood of parameters x given the locations and shared times at all loci and subtrees, as function of x.
    """
    L = len(log_det_shared_times) #number of loci
    n,_ = locations.shape #number of samples
 
    # convert location matrix to vector
    locations_vector = np.transpose(locations).flatten()

    if not important:
        branching_times = [None for _ in log_det_shared_times]
        logpcoals = branching_times

    def sumf(x):

        # reformulate parameters
        sigma = _sds_rho_to_sigma(x[0],x[1],x[2]) #as matrix FIX: assumes 2d
        log_det_sigma = np.linalg.slogdet(sigma)[1] #log of determinant
        sigma_inverted = np.linalg.pinv(sigma) #inverse
        phi = None
        if important: 
            phi = x[-1]/scale_phi

        # calculate negative log composite likelihood 
        g = 0
        # for each locus
        for sti, ldst, bts, lpcs in zip(shared_times_inverted, log_det_shared_times, branching_times, logpcoals):
            g -= _mc(locations=locations_vector, shared_times_inverted=sti, log_det_shared_times=ldst, 
                     sigma_inverted=sigma_inverted, 
                     important=important, branching_times=bts, phi=phi, logpcoals=lpcs) # subtract sum of log likelihood ratios over trees
        g += n*L/2 * log_det_sigma #can factor this out of all dispersal rate likelihoods

        return g

    return sumf

def _sds_rho_to_sigma(sdx,sdy,rho):
    """
    Convert sds and correlation to 2x2 covariance matrix
    """

    cov = sdx*sdy*rho
    sigma = np.array([[sdx**2, cov], [cov, sdy**2]])

    return sigma

def _mc(locations, shared_times_inverted, log_det_shared_times, sigma_inverted,
        important=False, branching_times=None, phi=None, logpcoals=None):
    """
    Monte Carlo estimate of log of likelihood ratio of the locations given parameters (sigma,phi) vs data given standard coalescent, for a given locus
    """

    LLRs = [] #log likelihood ratios at each tree

    if important:
        for sti, ldst, bts, lpc in zip(shared_times_inverted, log_det_shared_times, branching_times, logpcoals):
            LLRs.append(_log_likelihoodratio(locations=locations, shared_times_inverted=sti, log_det_shared_times=ldst, 
                                             sigma_inverted=sigma_inverted, 
                                             important=important, branching_times=bts, phi=phi, logpcoals=lpc))

    else:
        for sti,ldst in zip(shared_times_inverted, log_det_shared_times):
            LLRs.append(_log_likelihoodratio(locations=locations, shared_times_inverted=sti, log_det_shared_times=ldst, 
                                             sigma_inverted=sigma_inverted, 
                                             important=important))

    return _logsumexp(np.array(LLRs)) #sum likelihood ratios over trees then take log

def _logsumexp(a):
    """
    Take the log of a sum of exponentials without losing information.
    """

    a_max = np.max(a) #max element in list a
    tmp = np.exp(a - a_max) #now subtract off the max from each a before taking exponential (ie divide sum of exponentials by exp(a_max))
    s = np.sum(tmp) #and sum those up
    out = np.log(s) #and take log
    out += a_max  #and then add max element back on (ie multiply sum by exp(a_max), ie add log(exp(a_max)) to logged sum)

    return out

def _log_likelihoodratio(locations, shared_times_inverted, log_det_shared_times, sigma_inverted, 
                         important=True, branching_times=None, phi=None, logpcoals=None):
    """ 
    Log of likelihood ratio of parameters under branching brownian motion vs standard coalescent.
    """
  
    # log likelihood of dispersal rate
    LLR = _location_loglikelihood(locations, shared_times_inverted, log_det_shared_times, sigma_inverted)
    
    if important:
        # log probability of branching times given pure birth process with rate phi
        d,_ = sigma_inverted.shape
        LLR += _log_birth_density(branching_times=branching_times, phi=phi, n=int(len(locations)/d+1)) #assumes locations are mean centered (so add 1 to n)
        # log probability of coalescence times given standard coalescent (precalculated as parameter-independent)
        LLR -= logpcoals

    return LLR

def _location_loglikelihood(locations, shared_times_inverted, log_det_shared_times, sigma_inverted):
    """
    Log probability density of x, when x ~ MVN(mu,S). FIX
    """

    d,_ = sigma_inverted.shape
    
    # log of coefficient in front of exponential (times -2)
    logcoeff = d*log_det_shared_times #just the part that depends on data

    # exponent (times -2)
    exponent = np.matmul(np.matmul(np.transpose(locations), np.kron(sigma_inverted, shared_times_inverted)), locations)   

    return -0.5 * (logcoeff + exponent) #add the two terms together and multiply by -1/2

def _log_birth_density(branching_times, phi, n, condition_on_n=True):
    """
    Log probability of branching times given Yule process with branching rate phi.
    """

    T = branching_times[-1] #storing total time as last entry for convenience
    n0 = n - (len(branching_times) - 1) #initial number of lineages (number of samples minus number of coalescence events)
    
    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    k = n0 #initialize number of lineages
    # probability of each branching time
    for t in branching_times[:-1]: #for each branching time t
        logp += np.log(k * phi) - k * phi *  (t - prevt) #log probability of waiting time t-prevt with k lineages
        prevt = t #update time
        k += 1 #update number of lineages

    # probability of no branching from most recent branching to T
    logp += - k * phi * (T - prevt)

    # condition on having n samples from n0 in time T
    if condition_on_n:
        logp -= np.log(math.comb(k - 1, k - n0) * (1 - np.exp(-phi * T))**(k - n0)) - phi * n0 * T # see page 234 of https://www.pitt.edu/~super7/19011-20001/19531.pdf for two different expressions

    return logp

def _log_coal_density(times, Nes, epochs=None, tCutoff=None):

    """
    log probability of coalescent times under standard neutral/panmictic coalescent
    """

    if epochs is None and len(Nes) == 1:
        epochs = [0, max(times)] #one big epoch
        Nes = [Nes[0], Nes[0]] #repeat the effective population size so same length as epochs 

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    prevLambda = 0 #initialize coalescent intensity
    k = len(times) + 1 #number of samples
    if tCutoff is not None:
        times = times[times < tCutoff] #ignore old times
    myIntensityMemos = _coal_intensity_memos(epochs, Nes) #intensities up to end of each epoch

    # probability of each coalescence time
    for t in times: #for each coalescence time t
        kchoose2 = k * (k - 1) / 2 #binomial coefficient
        Lambda = _coal_intensity_using_memos(t, epochs, myIntensityMemos, Nes) #coalescent intensity up to time t
        ie = np.digitize(np.array([t]), epochs) #epoch at the time of coalescence
        logpk = np.log(kchoose2 * 1 / (2 * Nes[ie])) - kchoose2 * (Lambda - prevLambda) #log probability (waiting times are time-inhomogeneous exponentially distributed)
        logp += logpk #add log probability
        prevt = t #update time
        prevLambda = Lambda #update intensity
        k -= 1 #update number of lineages

    # now add the probability of lineages not coalescing by tCutoff
    if k > 1 and tCutoff is not None: #if we have more than one lineage remaining
        kchoose2 = k * (k - 1) / 2 #binomial coefficient
        Lambda = _coal_intensity_using_memos(tCutoff, epochs, myIntensityMemos, Nes) #coalescent intensity up to time tCutoff
        logPk = - kchoose2 * (Lambda - prevLambda) #log probability of no coalescence
        logp += logPk #add log probability

    return logp

def _coal_intensity_using_memos(t, epochs, intensityMemos, Nes):

    """
    add coal intensity up to time t
    """

    iEpoch = int(np.digitize(np.array([t]), epochs)[0] - 1) #epoch 
    t1 = epochs[iEpoch] #time at which the previous epoch ended
    Lambda = intensityMemos[iEpoch] #intensity up to end of previous epoch
    Lambda += 1 / (2 * Nes[iEpoch]) * (t - t1) #add intensity for time in current epoch
    return Lambda

def _coal_intensity_memos(epochs, Nes):

    """
    coalescence intensity up to the end of each epoch
    """

    Lambda = np.zeros(len(epochs))
    for ie in range(1, len(epochs)):
        t0 = epochs[ie - 1] #start time
        t1 = epochs[ie] #end time
        Lambda[ie] = (t1 - t0) #elapsed time
        Lambda[ie] *= 1 / (2 * Nes[ie - 1]) #multiply by coalescence intensity
        Lambda[ie] += Lambda[ie - 1] #add previous intensity

    return Lambda
