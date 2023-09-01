import numpy as np

def get_shared_times(tree, samples):

  T = tree.time(tree.root) #tmrca
  k = len(samples)

  sts = np.zeros((k,k))
  for i in range(k):
    sts[i,i] = T #shared time with self

    for j in range(i):
      st = T - tree.tmrca(samples[i],samples[j]) #shared time of pair
      sts[i,j] = st
      sts[j,i] = st

  return sts

def chop_shared_times(shared_times, tCutoff=None):

  k = len(shared_time) #total number of samples
  samples = np.arange(k) #list of samples 

  # if we don't need to cut then just return a single matrix and associated samples
  if tCutoff is None:
    return [shared_times], [samples]

  T = shared_times[0,0] #tmrca

  if tCutoff > T:
    return [shared_times], [samples]

  # if we do have to cut
  shared_time = tCutoff - (T-sts) #shared time since tCutoff

  # and now the harder part of grouping times and samples for each subtree 
  sts = [] #list of shared times matrices for each subtree
  smpls = [] #list of samples in each subtree
  taken = [False for _ in range(k)] #keep track of which samples already in a subtree
  while sum(taken) < k: #while some samples not yet in a subtree
    i = np.argmax(taken == False) #choose next sample, i, not yet in a subtree
    withi = shared_time[i] >= 0 #samples which share time with i
    taken = np.array([i[0] or i[1] for i in zip(taken, withi)]) #update which samples taken
    stsi = shared_time[withi][:, withi] #shared times of subtree with i
    smplsi = samples[np.where(withi)[0]] #samples in this subtree
    sts.append(stsi)
    smpls.append(smplsi)
      
  return sts, smpls 
