from mutation_variants.helpers import *
from mutation_variants.data.data import *
from mutation_variants.models.logistic_regression import *

def main( location ):
  X = load_from_csv( location )
  
  return X
  
if __name__ == "__main__":
  n_epochs = 10000
  extra_folder = "bas_apc"
  file_name = "COUNT_DATA.csv"
  #file_name = "RANK_DATA.csv"
  
  normalization = fair_rank_order_normalization
  #normalization = global_order_normalization
  
  location = os.path.join( DATA_DIR, extra_folder, file_name )
  results_location = os.path.join( RESULTS_DIR, extra_folder )
  print location
  print results_location
  #pdb.set_trace
  X = main(location)
  
  
  genes = X.columns[2:].values
  barcodes = X["Unnamed: 0"].values
  mutations = X["APC_MUTATED"].values
  R = X[genes].values
  
  normed_X = pd.DataFrame( normalization( R ), columns = genes, index = barcodes )
  normed_y = pd.DataFrame( mutations, columns = ["mutation"], index = barcodes )
  
  n,dim = normed_X.values.shape

  model = LogisticRegression( dim, l2=0.001 )
  
  X_all = normed_X.values
  y_all = normed_y.values.astype(int)
  
  l1s = [0.005,0.001,0.0005]
  K   = 4
  
  tr_ids, te_ids = xval_folds( n, K, randomize = True, seed = 0 )
  
  aucs = np.zeros( len(l1s) )
  y_true = np.squeeze( normed_y.values ).astype(int)
  #predictions = np.zeros( (n,len(l1s)), dtype=float )
  predictions = np.zeros( (n,len(l1s)), dtype=float )
  fig = pp.figure()
  for idx in range(len(l1s)):
    l1 = l1s[idx]
    print "running xval for l1 = ",l1
    k = 0
    for train_ids, test_ids in zip( tr_ids, te_ids ):
      print "   fold %d of %d"%(k,K)
  
      train_X = torch.from_numpy(normed_X.values[train_ids,:]).float() 
      train_y = torch.from_numpy(normed_y.values[train_ids].astype(int)).float() 
      test_X = torch.from_numpy(normed_X.values[test_ids,:]).float() 
      
      model = LogisticRegression( dim, l1=l1 )
      model.fit( train_X, train_y, lr=5*1e-1, n_epochs = n_epochs, logging_frequency=10000 )
  
      predict_y = np.squeeze( model.predict(test_X).data.numpy() )

      w = model.w.data.numpy()
      
      predictions[test_ids,idx] = predict_y
      k+=1
      
    y_est = np.squeeze( predictions[:,idx] )
    
    auc = roc_auc_score( y_true, y_est )
    fpr,tpr,thresholds = roc_curve( y_true, y_est )
    pp.plot( fpr, tpr, 'o-', label = "%s = %0.2f"%(str(l1),auc) )

    aucs[idx] = auc 
  pp.legend(loc="lower right")  
  pp.show()
  #gene_order = np.argsort( - np.abs(w) )
  # print genes[ gene_order[:10] ]
  #
  # y_est = np.squeeze( predict_y )
  # y_true = np.squeeze( normed_y.values ).astype(int)
  # auc = roc_auc_score( y_true, y_est )
  # fpr,tpr,thresholds = roc_curve( y_true, y_est )
  
  print model