import pandas as pd
import numpy as np
import pylab as pp
import scipy as sp
import torch
import os, sys
import pdb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
import seaborn as sns
sns.set(style="whitegrid")

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
  if randomize is True:
    print("XVAL RANDOMLY PERMUTING")
    if seed is not None:
      print( "XVAL SETTING SEED = %d"%(seed) )
      np.random.seed(seed)
      
    x = np.random.permutation(n)
  else:
    print( "XVAL JUST IN ARANGE ORDER")
    x = np.arange(n,dtype=int)
    
  kf = KFold( K )
  train = []
  test = []
  for train_ids, test_ids in kf.split( x ):
    #train_ids = np.setdiff1d( x, test_ids )
    
    train.append( x[train_ids] )
    test.append( x[test_ids] )
  #pdb.set_trace()
  return train, test
  
def check_and_mkdir( path_name, verbose = False ):
  ok = False
  if os.path.exists( path_name ) == True:
    ok = True
  else:
    if verbose:
      print "Making directory: ", path_name
    os.makedirs( path_name )
    ok = True
      
  return ok