from mutation_variants.helpers import *
from mutation_variants.data.data import *

def main( location ):
  X = load_from_csv( location )
  
  return X
  
if __name__ == "__main__":
  extra_folder = "bas_apc"
  file_name = "COUNT_DATA.csv"
  location = os.path.join( DATA_DIR, extra_folder, file_name )
  results_location = os.path.join( RESULTS_DIR, extra_folder )
  print location
  print results_location
  #pdb.set_trace
  X = main(location)
  
  
  genes = X.columns[2:]
  barcodes = X["Unnamed: 0"].values
  mutations = X["APC_MUTATED"].values
  R = X[genes].values
  
  normed_X = pd.DataFrame( fair_rank_order_normalization( R ), columns = genes, index = barcodes )
  normed_y = pd.DataFrame( mutations, columns = ["mutation"], index = barcodes )
  print normed_X.columns
  
  print normed_X, normed_y