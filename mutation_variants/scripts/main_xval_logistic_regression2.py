from mutation_variants.helpers import *
from mutation_variants.data.data import *
from mutation_variants.models.logistic_regression import *
from mutation_variants.viz.viz_weights import *

def load_x( data_location ):
  X =  pd.read_csv( data_location, sep="\t", index_col=0 )
  
  # pdb.set_trace()
  # genes = X.index.values
  # barcodes = X.columns.values
  # mutations = X["APC_MUTATED"].values
  # R = X[genes].values
  
  return X.T
  
def load_y( data_location ):
  y =  pd.read_csv( data_location, sep="\t", index_col=0 )
  
  return y
  
def main( X, normed_y, results_location, normalization, K, l1s, l2s ):
  #pd.read_csv( location, sep = sep )
  #X =  pd.read_csv( data_location, sep="\t", index_col=0 )
  
  #pdb.set_trace()
  genes = X.columns.values
  barcodes = X.index.values
  #mutations = y["mutation_status"].values
  
  R = X.values #X[genes].values
  
  normed_X = pd.DataFrame( normalization( R ), columns = genes, index = barcodes )
  #normed_y = pd.DataFrame( mutations, columns = ["mutation"], index = barcodes )
  
  n,dim = normed_X.values.shape
  
  tr_ids, te_ids = xval_folds( n, K, randomize = True, seed = 1 )
  
  aucs = np.zeros( len(l2s) )
  y_true = np.squeeze( normed_y.values ).astype(int)
  #predictions = np.zeros( (n,len(l1s)), dtype=float )
  predictions = np.zeros( (n,len(l2s)), dtype=float )
  fig = pp.figure()
  Ws = []
  Ys = []
  for idx in range(len(l2s)):
    l2 = l2s[idx]
    l1 = l1s[idx]
    print "running xval for l2 = ",l2
    print "running xval for l1 = ",l1
    k = 0
    w_l2 = 0.0
    for train_ids, test_ids in zip( tr_ids, te_ids ):
      print "   fold %d of %d"%(k,K)
  
      train_X = torch.from_numpy(normed_X.values[train_ids,:]).float() 
      train_y = torch.from_numpy(normed_y.values[train_ids].astype(int)).float() 
      test_X = torch.from_numpy(normed_X.values[test_ids,:]).float() 
      
      print "     number of 1's " + str(train_y.sum())
      print "     number of 0's " + str((1-train_y).sum())
      # model = LogisticRegression( dim, l2=l2 )
      model = LogisticRegression( dim, l1=l1 )
      model.fit( train_X, train_y, lr=1e-3, n_epochs = n_epochs, logging_frequency=500 )
  
      predict_y = np.squeeze( model.predict(test_X).data.numpy() )

      w = np.squeeze( model.w.data.numpy() )
      w_l2+=w
      
      predictions[test_ids,idx] = predict_y
      k+=1
     
    w_l2 /= K 
    Ws.append(w_l2)
    y_est = np.squeeze( predictions[:,idx] )
    
    Ys.append(y_est)
    auc = roc_auc_score( y_true, y_est )
    print "    => AUC = %f"%(auc)
    fpr,tpr,thresholds = roc_curve( y_true, y_est )
    pp.plot( fpr, tpr, 'o-', label = "%s = %0.2f"%(str(l2),auc) )

    aucs[idx] = auc 
  pp.legend(loc="lower right")  
  
  Ws = np.array(Ws)
  Ys = np.array(Ys)
  
  
  fig.savefig( results_location + "/xval_aucs.png", fmt="png", bbox_inches='tight')
  
  return normed_X, normed_y, Ys, aucs, Ws
  

  #gene_order = np.argsort( - np.abs(w) )
  # print genes[ gene_order[:10] ]
  #
  # y_est = np.squeeze( predict_y )
  # y_true = np.squeeze( normed_y.values ).astype(int)
  # auc = roc_auc_score( y_true, y_est )
  # fpr,tpr,thresholds = roc_curve( y_true, y_est )
  
  
  return X
  
if __name__ == "__main__":
  extra_folder = "bas_apc"
  file_name = "COUNT_DATA.csv"
  if len( sys.argv ) > 1:
    extra_folder = sys.argv[1]
    
  if len( sys.argv ) > 2:
    file_name_x  = sys.argv[2]
    
  if len( sys.argv ) > 3:
    file_name_y  = sys.argv[3]
    
  results_modifier = ""
  if len( sys.argv ) > 4:
    results_modifier  = sys.argv[4]
  
  # nbr of epochs for training
  n_epochs = 4000
  
  # regularization parameters for L2 penalty
  l2s = [10.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]# ,0.001,0.0005]
  l1s = np.array(l2s)/100
  # K-folds of X-validation
  K   = 5
  
  # we assume a folder: ~/data/mutation_variants
  # then an extra folder for this data: ~/data/mutation_variants/extra_folder
  # then one of these file names: ~/data/mutation_variants/extra_folder/COUNT_DATA.csv
  
  # if using COUNT, using fair rank normalization
  
  normalization = fair_rank_order_normalization
  
  # if using RANK use global order (just divides by the max)
  #file_name = "RANK_DATA.csv"
  #normalization = global_order_normalization
  
  # this is the full path of file:  ~/data/mutation_variants/extra_folder/COUNT_DATA.csv
  x_location = os.path.join( DATA_DIR, extra_folder, file_name_x )
  y_location = os.path.join( DATA_DIR, extra_folder, file_name_y )
  
  x_data = load_x( x_location )
  y_data = load_y( y_location )
  
  # filter mss and impact
  stable      = y_data["mss"] == 1
  high_impact_stable = stable&(y_data["expected_impact"]==3)
  no_impact_stable   = stable&(y_data["expected_impact"]==0)
  
  query = high_impact_stable | no_impact_stable
  
  query = query[ query.values ]
  x_data_select = x_data.loc[ query.index ]

  y_data_select = y_data.loc[ query.index ]
  y_data_select["mutation"] = (y_data_select["expected_impact"]==3).astype(int)
  y_data_select = y_data_select[ "mutation" ]
  #
  # # assumes results folder: ~/data/mutation_variants/extra_folder/
  #
  results_location = os.path.join( RESULTS_DIR, extra_folder+results_modifier )
  check_and_mkdir(results_location)
  #
  #print "Data location: ", data_location
  print "Results location: ", results_location
  #
  #
  normed_X, normed_y, Ys, aucs, Ws = main( x_data_select, \
                                           y_data_select, \
                                           results_location, \
                                           normalization, K, l1s, l2s )
  genes = normed_X.columns

  best_auc_id = np.argmax(aucs)
  best_w      = Ws[best_auc_id]
  best_y_est  = Ys[best_auc_id]
  best_auc    = aucs[best_auc_id]
  best_l2     = l2s[best_auc_id]
  order_weights = np.argsort( -np.abs(best_w) )

  nbr_2_select = 100
  
  weight_ids = order_weights[ :nbr_2_select]
  best_genes = genes[ weight_ids ]
  best_weights = best_w[ weight_ids ]
  f_vert,ax_vert = viz_weights_vertical( best_weights, best_genes )
  ax_vert.set_title( "AUC = %0.3f L2 = %f K = %d"%(best_auc,best_l2,K ))
  f_horz,ax_horz = viz_weights_horizontal( best_weights, best_genes )
  ax_horz.set_title( "AUC = %0.3f L2 = %f K = %d"%(best_auc,best_l2,K ))
  #
  #
  f_vert.savefig( results_location + "/xval_best_w_vert.png", fmt="png", bbox_inches='tight')
  f_horz.savefig( results_location + "/xval_best_w_horz.png", fmt="png", bbox_inches='tight')


  # 
  
  
  
