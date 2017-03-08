from mutation_variants.helpers import *


def viz_weights_vertical( w, names  ):
  order = np.argsort( -w )
  
  d = len(w)
  
  #centers = 1+np.arange( n )
  x_values = np.arange( d )
  
  f = pp.figure( figsize=(6,10))
  
  ax1 = f.add_subplot(111)
  w_ordered = w[order]
  neg = w_ordered<0
  pos = w_ordered >=0
  ax1.plot( w_ordered[pos], x_values[pos] , 'bo' )
  ax1.plot( w_ordered[neg], x_values[neg] , 'ro' )
  pp.yticks(x_values, names[order], rotation='horizontal', fontsize=6)
  pp.margins(0.05)
  pp.subplots_adjust(left=0.15)
  ax1.grid(color='k', linestyle='--', linewidth=0.5,axis='x',alpha=0.5)
  return f, ax1

def viz_weights_horizontal( w, names ):
  
  #max_w = np.max(np.abs(W))
  #normed_W = W / max_w 
  
  order = np.argsort( -w )
  
  d = len(w)
  
  #centers = 1+np.arange( n )
  x_values = np.arange( d )
  
  f = pp.figure( figsize=(18,8))
  
  ax1 = f.add_subplot(111)
  w_ordered = w[order]
  neg = w_ordered<0
  pos = w_ordered >=0
  ax1.plot( x_values[pos], w_ordered[pos], 'bo' )
  ax1.plot( x_values[neg], w_ordered[neg], 'ro' )
  pp.xticks(x_values, names[order], rotation='vertical', fontsize=8)
  pp.margins(0.05)
  pp.subplots_adjust(bottom=0.15)
  ax1.grid(color='k', linestyle='--', linewidth=0.5,axis='y',alpha=0.5)
  return f, ax1