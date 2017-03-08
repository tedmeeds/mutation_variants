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
  
  normed_X = pd.DataFrame( fair_rank_order_normalization( R ), columns = genes, index = barcodes )
  normed_y = pd.DataFrame( mutations, columns = ["mutation"], index = barcodes )
  
  n,dim = normed_X.values.shape

  model = LogisticRegression( dim, l2=0.001 )
  
  train_X = torch.from_numpy(normed_X.values).float() 
  train_y = torch.from_numpy(normed_y.values.astype(int)).float() 
  model.fit( train_X, train_y, lr=5*1e-1, n_epochs = n_epochs, logging_frequency=1000 )
  
  predict_y = model.predict(train_X).data.numpy()
  
  P = []
  for p in model.model.parameters():
    P.append( np.squeeze( p.data.numpy() ) )
   
  w = P[0]
  
  gene_order = np.argsort( - np.abs(w) )
  print genes[ gene_order[:10] ]
   
  y_est = np.squeeze( predict_y )
  y_true = np.squeeze( normed_y.values ).astype(int)
  auc = roc_auc_score( y_true, y_est )
  fpr,tpr,thresholds = roc_curve( y_true, y_est )
  
  print model