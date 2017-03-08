from mutation_variants.helpers import *

def fair_rank( x ):
  #ranks = []
  ix = np.argsort(x)
  sx = x[ix]
  rnk = 0
  old_x = sx[0]
  ranks = [rnk]
  cnter = 1
  for xi in sx[1:]:
    if xi > old_x:
      rnk += 1 #cnter
      cnter = 1
    else:
      cnter += 1
    old_x = xi
    ranks.append(rnk)
  ranks = np.array(ranks, dtype=float)/float(rnk)
  return ranks[np.argsort(ix)]
#
def fair_rank_order_normalization( X ):
  #ORDER = np.argsort( -X, 1 )
  Y = X.copy()
  #vals = np.linspace(eps,1.0-eps,X.shape[1])
  for idx in range(Y.shape[0]):
    Y[idx,:] = fair_rank( Y[idx] )
  return Y

def global_order_normalization( X ):
  max_val = np.max(X)
  return X.astype(float)/max_val
    
def load_from_csv( location ):
  return pd.read_csv( location )