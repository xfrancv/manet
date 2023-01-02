from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle

def load_results( results_folder, lambda_range=(0,np.Inf) ): 

    files = [f for f in listdir(results_folder) if isfile(join(results_folder, f)) and f[-3:]=="pkl"]

    split = []
    lam = []
    n_examples = []
    for f in files:
        split.append( int(f.split("_")[1][5:]) )
        lam.append( float(f.split("_")[2][6:]) )
        n_examples.append( int( f.split("_")[3].split('.')[0][1:]) )

    lam = np.array( lam )
    n_examples = np.array( n_examples )
    split = np.array( split )

    set_lambda = np.unique(lam)
    set_lambda = set_lambda[(set_lambda >= lambda_range[0]) & (set_lambda <= lambda_range[1])]
    set_n_examples = np.unique(n_examples)
    set_split = np.unique(split)


    best_lambda = np.zeros( (len(set_split), len( set_n_examples )) )
    best_obj_val = np.zeros( (len(set_split), len( set_n_examples )) )
    best_tst_err = np.zeros( (len(set_split), len( set_n_examples )) )
    best_trn_err = np.zeros( (len(set_split), len( set_n_examples )) )
    best_val_err = np.zeros( (len(set_split), len( set_n_examples )) )
    best_tst_err_01 = np.zeros( (len(set_split), len( set_n_examples )) )
    best_trn_err_01 = np.zeros( (len(set_split), len( set_n_examples )) )
    best_val_err_01 = np.zeros( (len(set_split), len( set_n_examples )) )
    for si,s in enumerate(set_split):
        for ni,n in enumerate(set_n_examples):        
            idx = np.where( np.logical_and(s == split, n == n_examples ) )

            min_val_err = np.Inf
            for i in idx[0]:
                if lam[i] >= lambda_range[0] and lam[i] <= lambda_range[1]:
                    data = pickle.load( open( results_folder + files[i], "rb" ) )
                    results = data[1]
                    val_err = results["val_err"]
                    if val_err < min_val_err:
                        min_val_err = val_err
                        best_lambda[si,ni] = lam[i]
                        best_obj_val[si,ni] = results["obj_val"][-1]
                        best_tst_err[si,ni] = results["tst_err"]
                        best_trn_err[si,ni] = results["trn_err"]
                        best_val_err[si,ni] = results["val_err"]
                        best_tst_err_01[si,ni] = results["tst_err_01"]
                        best_trn_err_01[si,ni] = results["trn_err_01"]
                        best_val_err_01[si,ni] = results["val_err_01"]

    return {
        "lambdas": set_lambda,
        "splits": set_split,
        "n_examples": set_n_examples, 
        "best_lambda": best_lambda,
        "obj_val": best_obj_val,
        "tst_err": best_tst_err,
        "trn_err": best_trn_err,
        "val_err": best_val_err,
        "tst_err_01": best_tst_err_01,
        "trn_err_01": best_trn_err_01,
        "val_err_01": best_val_err_01,
    }

def get_results( results_folder, lambda_range=(0,np.Inf) ): 

    files = [f for f in listdir(results_folder) if isfile(join(results_folder, f)) and f[-3:]=="pkl"]

    split = []
    lam = []
    n_examples = []
    for f in files:
        split.append( int(f.split("_")[1][5:]) )
        lam.append( float(f.split("_")[2][6:]) )
        n_examples.append( int( f.split("_")[3].split('.')[0][1:]) )

    lam = np.array( lam )
    n_examples = np.array( n_examples )
    split = np.array( split )

    set_lambda = np.unique(lam)
    set_lambda = set_lambda[(set_lambda >= lambda_range[0]) & (set_lambda <= lambda_range[1])]
    set_n_examples = np.unique(n_examples)
    set_split = np.unique(split)

    best_lambda = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_obj_val = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_tst_err = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_trn_err = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_val_err = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_tst_err_01 = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_trn_err_01 = pd.DataFrame(columns=set_split,index=set_n_examples)
    best_val_err_01 = pd.DataFrame(columns=set_split,index=set_n_examples)

    for s in set_split:
        for n in set_n_examples:   
            idx = np.where( np.logical_and(s == split, n == n_examples ) )

            min_val_err = np.Inf
            for i in idx[0]:
                if lam[i] >= lambda_range[0] and lam[i] <= lambda_range[1]:
                    #obj_val = pickle.load( open( results_folder + files[i], "rb" ) )[1]
                    obj_val = [None]
                    results = pickle.load( open( results_folder + files[i][0:-4] + "/results.pkl", "rb" ) )
                    val_err = results["val_err"]
                    if val_err < min_val_err:
                        min_val_err = val_err
                        best_lambda.loc[n,s] = lam[i]
                        best_obj_val.loc[n,s] = obj_val[-1]
                        best_tst_err.loc[n,s] = results["tst_err"]
                        best_trn_err.loc[n,s] = results["trn_err"]
                        best_val_err.loc[n,s] = results["val_err"]
                        best_tst_err_01.loc[n,s] = results["tst_err_01"]
                        best_trn_err_01.loc[n,s] = results["trn_err_01"]
                        best_val_err_01.loc[n,s] = results["val_err_01"]

    best_lambda['mean'] = best_lambda.loc[:,set_split].mean(axis=1)
    best_lambda['std'] = best_lambda.loc[:,set_split].std(axis=1)
    best_obj_val['mean'] = best_obj_val.loc[:,set_split].mean(axis=1)
    best_obj_val['std'] = best_obj_val.loc[:,set_split].std(axis=1)
    best_tst_err['mean'] = best_tst_err.loc[:,set_split].mean(axis=1)
    best_tst_err['std'] = best_tst_err.loc[:,set_split].std(axis=1)
    best_val_err['mean'] = best_val_err.loc[:,set_split].mean(axis=1)
    best_val_err['std'] = best_val_err.loc[:,set_split].std(axis=1)
    best_trn_err['mean'] = best_trn_err.loc[:,set_split].mean(axis=1)
    best_trn_err['std'] = best_trn_err.loc[:,set_split].std(axis=1)
    best_tst_err_01['mean'] = best_tst_err_01.loc[:,set_split].mean(axis=1)
    best_tst_err_01['std'] = best_tst_err_01.loc[:,set_split].std(axis=1)
    best_val_err_01['mean'] = best_val_err_01.loc[:,set_split].mean(axis=1)
    best_val_err_01['std'] = best_val_err_01.loc[:,set_split].std(axis=1)
    best_trn_err_01['mean'] = best_trn_err_01.loc[:,set_split].mean(axis=1)
    best_trn_err_01['std'] = best_trn_err_01.loc[:,set_split].std(axis=1)

    return {
        "lambdas": set_lambda,
        "splits": set_split,
        "n_examples": set_n_examples, 
        "best_lambda": best_lambda,
        "obj_val": best_obj_val,
        "tst_err": best_tst_err,
        "trn_err": best_trn_err,
        "val_err": best_val_err,
        "tst_err_01": best_tst_err_01,
        "trn_err_01": best_trn_err_01,
        "val_err_01": best_val_err_01,
    }

if __name__ == "__main__":

    root_folder = "/home/xfrancv/Work/ConsistentSurrogate/code/PyMaxSum/results/"    

    results = get_results( root_folder + "hmc_30x30x100_A/adam_mrhomo/", lambda_range=(0,np.Inf) )

    print( results["tst_err"] )    
    print( results["val_err"] )    
    print( results["trn_err"] )