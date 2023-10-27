import numpy as np

def get_shared_times(tree, samples):

  T = tree.time(tree.root) #tmrca of tree
  k = len(samples)

  sts = np.zeros((k,k))
  for i in range(k):
    sts[i,i] = T #shared time with self (same for all samples, so put in any diagonal entry)

    for j in range(i):
      st = T - tree.tmrca(samples[i],samples[j]) #shared time of pair, placed to align with locations
      sts[i,j] = st
      sts[j,i] = st

  return sts

def chop_shared_times(shared_times, tCutoff=None):

  k = len(shared_times) #total number of samples
  samples = np.arange(k) #list of samples (assumes the shared times are ordered already)
  
  # if we don't need to cut then just return a single matrix and associated samples
  T = shared_times[0,0] #tmrca
 
  if tCutoff is None or tCutoff>T:
    return [shared_times], [samples]

  # if we do have to cut
  shared_times_sinceT = tCutoff - (T-shared_times) #calculate shared time since tCutoff

  # and now the harder part of grouping times and samples for each subtree 
  sts = [] #list of shared times matrices for each subtree
  smpls = [] #list of samples in each subtree
  taken = [False for _ in range(k)] #keep track of which samples already in a subtree
  while sum(taken) < k: #while some samples not yet in a subtree
    i = np.argmax(taken == False) #choose next sample, i, not yet in a subtree
    withi = shared_times_sinceT[i] >= 0 #samples which share time with i
    taken = np.array([i[0] or i[1] for i in zip(taken, withi)]) #update which samples taken
    stsi = shared_times_sinceT[withi][:, withi] #shared times of subtree with i
    smplsi = samples[np.where(withi)[0]] #samples in this subtree
    sts.append(stsi)
    smpls.append(smplsi)
      
  return sts, smpls 

def center_shared_times(shared_times):
 
  n = len(shared_times) #number of samples in subtree
  Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[0:-1]; #matrix for mean centering
  stc = np.matmul(Tmat, np.matmul(shared_times, np.transpose(Tmat))) #center shared times in subtree
 
  return stc
