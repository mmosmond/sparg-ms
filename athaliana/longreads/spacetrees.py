from scipy.optimize import minimize
import time
import numpy as np
import math
from tqdm import tqdm 

def mle_dispersal_tree(locations, shared_times_inverted):
    """
    Maximum likelihood estimate of dispersal rate given locations and (inverted) shared times between lineages in a tree.
    """

    return np.matmul(np.matmul(np.transpose(locations), shared_times_inverted), locations) / len(locations)

def mle_dispersal_numeric(locations, shared_times_inverted, log_det_shared_times, samples,
                          sigma0=None, bnds=None, method='L-BFGS-B', callbackF=None,
                          important=False, branching_times=None, phi0=None, scale_phi=None, logpcoals=None,
                          quiet=False):
    """
    Numerically estimate maximum likelihood dispersal rate (and possibly branching rate) given sample locations and shared_times.
    """

    L = len(log_det_shared_times)
    M = len(log_det_shared_times[0])
    n, d = locations.shape
    if not quiet: print('number of loci:',L,'\nnumber of trees per locus:',M,'\nnumber of samples:',n,'\nnumber of spatial dimensions:',d)

    # prepare locations
    if not quiet: print('\npreparing locations')
    if sigma0 is not None:
        locsss = []
        for smplss in tqdm(samples): #loci
            locss = []
            for smpls in smplss: #trees
                locs = []
                for smpl in smpls: #subtrees
                    k = len(smpl)
                    loc = np.array([locations[i] for i in smpl]) #locations of that subtree
                    Tmat = np.identity(k) - [[1/k for _ in range(k)] for _ in range(k)]; Tmat = Tmat[0:-1] #mean centering matrix
                    loc_mc = np.matmul(Tmat, loc) #mean centered locations
                    loc_mc_vec = np.transpose(loc_mc).flatten() #make a vector
                    locs.append(loc_mc_vec)
                locss.append(locs)
            locsss.append(locss)
    # and potentially find decent initial dispersal rate
    else:
        if not quiet: print('and initializing dispersal rate')
        locsss = []
        kvsum = np.zeros((d,d))
        ksum = 0
        for stss, smplss in tqdm(zip(shared_times_inverted, samples)): #loci
            locss = []
            kvisum = np.zeros((d,d))
            kisum = 0
            for sts, smpls in zip(stss, smplss): #trees
                locs = []
                kvijsum = np.zeros((d,d))
                kijsum = 0
                for st, smpl in zip(sts, smpls): #subtrees
                    k = len(smpl) #number of samples in subtree before mean centering
                    loc = np.array([locations[i] for i in smpl]) #locations of that subtree
                    Tmat = np.identity(k) - [[1/k for _ in range(k)] for _ in range(k)]; Tmat = Tmat[0:-1] #mean centering matrix
                    loc_mc = np.matmul(Tmat, loc) #mean centered locations
                    loc_mc_vec = np.transpose(loc_mc).flatten() #make a vector
                    locs.append(loc_mc_vec)
                    if k>1: #need more than 1 sample in a subtree to estimate dispersal
                        mle = mle_dispersal_tree(loc_mc, st.astype(float))
                        kvijsum += (k-1)*mle #add weighted mle dispersal rate for subtree 
                        kijsum += k-1
                locss.append(locs)
                if kisum == 0: #just use first tree at each locus to initialize sigma (alternatively could average over tree mles to get BLUP)
                    kvisum += kvijsum #add weighted mle dispersal rate for tree
                    kisum += kijsum
            locsss.append(locss)
            kvsum += kvisum #add weighted mle dispersal rate for locus
            ksum += kisum
        sigma0 = kvsum/ksum
        if not quiet: print('initial dispersal rate:\n',sigma0)
    x0 = _sigma_to_sds_rho(sigma0) #convert to standard deviations and correlation

    # initializing branching rate
    if important:
        if phi0 is None:
            if not quiet: print('\nfinding initial branching rate')
            phi0 = np.mean([np.log(n/len(smpls))/bts[-1] for smplss,btss in zip(samples,branching_times) for smpls,bts in zip(smplss,btss)]) #initial guess at branching rate, from n(t)=n(0)e^(phi*t)
            if not quiet: print('initial branching rate:',phi0) 
        if scale_phi is None:
            scale_phi = x0[0]/phi0 #we will search for the value of phi*scale_phi that maximizes the likelihood (putting phi on same scale as dispersal accelarates search) 
        if not quiet: print('multiplying branching rate by:',scale_phi)
        x0.append(phi0*scale_phi)
        
    # negative composite log likelihood ratio, as function of x
    f = _sum_mc(locations=locsss, shared_times_inverted=shared_times_inverted, log_det_shared_times=log_det_shared_times,
                important=important, branching_times=branching_times, scale_phi=scale_phi, logpcoals=logpcoals)

    # impose bounds on parameters
    if bnds is None:
        bnds = [(1e-6,None),(1e-6,None),(-0.99,0.99)] #FIX: assumes 2d
    if important:
        bnds.append((1e-6,None))

    # find mle
    if not quiet: print('\nsearching for maximum likelihood parameters...')
    if callbackF is not None: callbackF(x0)
    t0 = time.time()
    m = minimize(f, x0=x0, bounds=bnds, method=method, callback=callbackF) #find MLE
    if not quiet: print('finding the max took', time.time()-t0, 'seconds')

    if not quiet:
        mle = m.x
        sigma = _sds_rho_to_sigma(mle[0],mle[1],mle[2]) #convert to covariance matrix #FIX: assumes 2d
        print('\nmaximum likelihood dispersal rate:\n',sigma)
        if important:
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

    if not important:
        L = len(log_det_shared_times) #number of loci
        branching_times = [None for _ in range(L)]
        logpcoals = branching_times

    def sumf(x):

        # reformulate parameters
        sigma = _sds_rho_to_sigma(x[0],x[1],x[2]) #as matrix FIX: assumes 2d
        log_det_sigma = np.linalg.slogdet(sigma)[1] #log of determinant
        sigma_inverted = np.linalg.pinv(sigma) #inverse
        phi = None
        if important: 
            phi = x[-1]/scale_phi

        # calculate negative log composite likelihood ratio
        # by subtracting log likelihood ratio at each locus
        g = 0
        for locs, sti, ldst, bts, lpcs in zip(locations, shared_times_inverted, log_det_shared_times, branching_times, logpcoals): #loop over loci
            g -= _mc(locations=locs, shared_times_inverted=sti, log_det_shared_times=ldst,
                     sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma,
                     important=important, branching_times=bts, phi=phi, logpcoals=lpcs)
        return g
    
    return sumf

def _sds_rho_to_sigma(sdx,sdy,rho):
    """
    Convert sds and correlation to 2x2 covariance matrix
    """

    cov = sdx*sdy*rho
    sigma = np.array([[sdx**2, cov], [cov, sdy**2]])

    return sigma

def _mc(locations, shared_times_inverted, log_det_shared_times, sigma_inverted, log_det_sigma,
        important=False, branching_times=None, phi=None, logpcoals=None):
    """
    Monte Carlo estimate of log of likelihood ratio of the locations given parameters (sigma,phi) vs data given standard coalescent, for a given locus
    """

    LLRs = [] #log likelihood ratios at each tree

    # loop over trees at a locus
    if important:
        for locs, sti, ldst, bts, lpc in zip(locations, shared_times_inverted, log_det_shared_times, branching_times, logpcoals):
            LLRs.append(_log_likelihoodratio(locations=locs, shared_times_inverted=sti, log_det_shared_times=ldst,
                                             sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma, 
                                             important=important, branching_times=bts, phi=phi, logpcoals=lpc))

    else:
        for locs, sti, ldst in zip(locations, shared_times_inverted, log_det_shared_times):
            LLRs.append(_log_likelihoodratio(locations=locs, shared_times_inverted=sti, log_det_shared_times=ldst,
                                             sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma,
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

def _log_likelihoodratio(locations, shared_times_inverted, log_det_shared_times, sigma_inverted, log_det_sigma,
                         important=False, branching_times=None, phi=None, logpcoals=None):
    """ 
    Log of likelihood ratio of parameters under branching brownian motion vs standard coalescent.
    """
 
 
    # log likelihood of dispersal rate
    LLR = 0
    n = 0
    ksum = 0
    for locs, sts, ldst in zip(locations, shared_times_inverted, log_det_shared_times): #loop over subtrees
        k = len(sts); n += k + 1 #the plus 1 assumes times are mean centered
        LLR += _location_loglikelihood(locs, sts.astype(float), ldst, sigma_inverted)
        ksum += k
    d,_ = sigma_inverted.shape
    LLR -= ksum/2 * (d*np.log(2*np.pi) + log_det_sigma)  #can factor this out over subtrees

    if important:
        # log probability of branching times given pure birth process with rate phi
        LLR += _log_birth_density(branching_times=branching_times, phi=phi, n=n) 
        # log probability of coalescence times given standard coalescent (precalculated as parameter-independent)
        LLR -= logpcoals
    
    return LLR

def _location_loglikelihood(locations, shared_times_inverted, log_det_shared_times, sigma_inverted):
    """
    Log probability density of locations when locations ~ MVN(0,sigma_inverted*shared_times_inverted).
    """
    
    # log of coefficient in front of exponential (times -2)
    d,_ = sigma_inverted.shape
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

    return logp[0] #FIX: extra dimn introduced somewhere

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
