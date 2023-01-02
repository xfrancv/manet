import numpy as np
from manet.hmc import HMC
from tqdm import tqdm
import pickle
import os
import sys


if __name__ == "__main__":

    tvt_split = [1000,5000,5000]  # trn/val/tst examples
    n_splits  = 5

    for p_missing in [0.0, 0.1, 0.2]:

        np.random.seed(42)
    
        data_file = f"./data/hmc_30x30x100_mis{(p_missing*100):.0f}.pkl"

        ## setup HMC model
        n  = 30      # number of symbols = number of hidden states
        length = 100 # sequence length
        alpha = 0.7
        beta = 0.7

        # edges
        E = np.concatenate(( np.arange(0,length-1).reshape((1,length-1)),
                            np.arange(1,length).reshape((1,length-1)) ),axis=0)

        # transition matrix
        trans = (1-beta)*np.ones((n,n)) / (n-1)
        np.fill_diagonal( trans, beta ) 

        # emission matrix
        emis = (1-alpha)*np.ones( (n,n )) / (n - 1)
        np.fill_diagonal( emis, alpha )

        # initial hidden state
        p0 = np.ones( n ) / n

        ##
        if os.path.exists(data_file):
            print(f"Output file {data_file} already exist. Erase it first.")
        else:

            Hmc = HMC( p0, trans, emis, alphabet=np.int16)

            ## generate all examples
            m = sum( tvt_split )
            examples = []
            for i in tqdm( range( m*n_splits )):

                X, Y = Hmc.generate( length )
                    
                examples.append({
                    'X': X,
                    'Y': Y,
                    'E': E,
                    'n_x': n,
                    'n_y': n,
                    'graph': "chain",
                    'n_objects': length} )


            ## generate splits of the data
            splits = []
            offset = 0
            for s in range( n_splits ):    
                perm = np.random.permutation( m )
                splits.append( {'trn': offset+perm[0:tvt_split[0]], 
                            'val': offset+perm[tvt_split[0]:tvt_split[0]+tvt_split[1]],
                            'tst': offset+perm[tvt_split[0]+tvt_split[1]:]} )
                offset += m
                
            for s in range( n_splits ):    
                for i in splits[s]["trn"]:
                    mis = np.random.choice( 2, length, p=[1-p_missing,p_missing] )
                    examples[i]["Y"][mis==1] = -1
                        
            ## save to file
            pickle.dump( [examples, splits], open( data_file, "wb" ) )
