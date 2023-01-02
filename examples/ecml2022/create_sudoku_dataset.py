import numpy as np
from tqdm import tqdm
import pickle
import os
import csv
from math import ceil


if __name__ == "__main__":


    file_with_sudoku_assignments = './data/sudoku10000.csv';
    tvt_split = [1000,100,100]   # number of training/validation/test examples
    n_splits = 5                 # number of splits


    for p_missing in [0.0, 0.1, 0.2]:

        np.random.seed(42)

        ##
        output_file = f"data/sudoku_{tvt_split[0]}_{tvt_split[1]}_{tvt_split[2]}_mis{p_missing*100:.0f}.pkl"

        ##
        if os.path.exists( output_file):
            print(f"Output file {output_file} already exist. Erase it first.")
        else:

            ## load Sudoku assignments/solutions from CSV file
            m = sum( tvt_split )
            X = []
            Y = []
            with open( file_with_sudoku_assignments, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                row =next( reader)
                cnt = 0
                for row in tqdm(reader):
                    cnt += 1
                    if cnt > m*n_splits:
                        break

                    x = row[0].split(',')[0]
                    y = row[0].split(',')[1]
                    x = np.array([char for char in x]).astype(int).reshape(9,9)
                    X.append( x )
                    Y.append(np.array([char for char in y]).astype(int).reshape(9,9)-1)


            ## create edges according to Sudoku rules
            E = []
            for i1 in range(1,10):
                for j1 in range(1,10):
                    for i2 in range(1,10):
                        for j2 in range(1,10):
                            if i1==i2 or j1==j2 or (ceil(i1/3)==ceil(i2/3) and ceil(j1/3)==ceil(j2/3)):
                                v0 = i1 + (j1-1)*9 - 1
                                v1 = i2 + (j2-1)*9 - 1
                                if v0 != v1 and ([v0,v1] not in E) and ([v1,v0] not in E):
                                    E.append([v0,v1])

            E = np.array(E).transpose()

            ##
            examples = []
            for i in tqdm( range( m*n_splits )):    
                examples.append({
                    'X': X[i].reshape(-1),
                    'Y': Y[i].reshape(-1),
                    'E': E,
                    'n_x': 10,
                    'n_y': 9,
                    'graph': "general",
                    'n_objects': 9*9} )

            
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
                    mis = np.random.choice( 2, examples[i]["n_objects"], p=[1-p_missing,p_missing] )
                    examples[i]["Y"][np.logical_and(mis==1,examples[i]["X"]==0)] = -1

            
            ## save to file
            pickle.dump( [examples, splits], open( output_file, "wb" ) )


