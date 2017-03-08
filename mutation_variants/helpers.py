import pandas as pd
import numpy as np
import pylab as pp
import scipy as sp
import torch
import os, sys
import pdb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
HOME_DIR =  os.environ["HOME"]
DATA_DIR = os.path.join( HOME_DIR, "data/mutation_variants/" )
RESULTS_DIR = os.path.join( HOME_DIR, "results/mutation_variants/" )

def bootstraps( x, m ):
  # samples from arange(n) with replacement, m times.
  #x = np.arange(n, dtype=int)
  n = len(x)
  N = np.zeros( (m,n), dtype=int)
  for i in range(m):
    N[i,:] = sklearn.utils.resample( x, replace = True )
    
  return N
  
def xval_folds( n, K, randomize = False, seed = None ):
  if randomize:
    if seed is not None:
      np.random.seed(seed)
      
    x = np.random.permutation(n)
  else:
    x = np.arange(n,dtype=int)
    
  kf = KFold( K )
  train = []
  test = []
  for train_ids, test_ids in kf.split( x ):
    #train_ids = np.setdiff1d( x, test_ids )
    
    train.append( train_ids )
    test.append( test_ids )
  
  return train, test