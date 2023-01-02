import numpy as np
from sklearn.datasets import fetch_openml
import pickle
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm 

if __name__ == "__main__":

    num_basis = 2000
    kernel_width = 0.5

    for symbolic_sudoku_file in ["./data/sudoku_1000_100_100_mis0.pkl", 
                                 "./data/sudoku_1000_100_100_mis10.pkl",
                                 "./data/sudoku_1000_100_100_mis20.pkl"]:

        output_file = symbolic_sudoku_file[:-4] + f"_{num_basis}.pkl"
        print(f"[{output_file}]")

        # load symbolic sudoku assignments
        exam, splits = pickle.load( open( symbolic_sudoku_file, "rb" ) )


        # load mnist
        mnist = fetch_openml('mnist_784')
        X = mnist.data.transpose()
        X = X/np.linalg.norm(X,axis=0) # normalization
        Y = mnist.target.astype(int)

        hist = np.histogram( Y, bins=[0,1,2,3,4,5,6,7,8,9,10] )
        num_mnist_digits_per_class = hist[0]
        print(num_mnist_digits_per_class)

        for s in range( len(splits)):

            print(f"[split {s}]")

            split = splits[s]

            # count digit frequency in Sudokus of val+tst split 
            num_valtst_digits_per_class = np.zeros(10).astype(int)
            for part in ["val","tst"]:
                for i in split[part]:
                    for x in exam[i]["X"]:
                        num_valtst_digits_per_class[x] += 1

            # decide which mnist digits be used for training and validation+testing
            digit_indices = [np.argwhere(Y==y) for y in range(10)]

            trn_indices = []
            valtst_indices = []
            for x in range( 10 ):
                # n = the number of examples that remain for training after removing the val+tst examples
                n = num_mnist_digits_per_class[x] - num_valtst_digits_per_class[x] 
                if n <= 0: 
                    print(f"not enough digits in class {x}")
                    trn_indices.append( [] )
                    valtst_indices.append( [] )
                else:
                    trn_indices.append( digit_indices[x][0:n] )
                    valtst_indices.append( digit_indices[x][n:] )
                print(f"digit={x}: all_in_mnist={num_mnist_digits_per_class[x]} trn={n} val+tst={num_valtst_digits_per_class[x]} all: {len(digit_indices[x])}")
        
            # select basis vectors randomly from all classes except digit 0 which is unused
            basis_indices = np.empty(0,dtype=int)
            for y in range(1,10):
                basis_indices = np.concatenate( (basis_indices.reshape(-1), trn_indices[y].reshape(-1)))

            basis_indices = np.random.permutation(basis_indices)
            basis_indices = basis_indices[0:num_basis]
            basis_X = X[:,basis_indices]

            # create whitening matrix for computing explicit kernel features
            kernel = RBF(kernel_width)
            K = kernel( basis_X.transpose() )
            U, S, _ = np.linalg.svd( K, full_matrices=True)
            A = np.diag(1/np.sqrt(S)) @ U.transpose()

            used_trn_digits = np.zeros(10,dtype=int)
            used_valtst_digits = np.zeros(10,dtype=int)

            for part in ["trn","val","tst"]:
                print(f"[processing {part}]")
                for i in tqdm(split[part]):
                    new_X = np.zeros((X.shape[0],9*9))
                    for j,x in enumerate( exam[i]["X"] ):
                        if x > 0: # x == 0 is unfilled cell which we consider to be a black image
                            if part == "trn":
                                idx = trn_indices[x][used_trn_digits[x]]
                                used_trn_digits[x] += 1
                                # training digits can be used multiple times in different sudokus
                                if used_trn_digits[x] >= len(trn_indices[x]):
                                    used_trn_digits[x] = 0

                            else:
                                idx = valtst_indices[x][used_valtst_digits[x]]
                                used_valtst_digits[x] += 1
                            
                            new_X[:,j] = X[:,idx].reshape(-1)
                                
                    # kernel features                        
                    exam[i]["X"] = A @ kernel( basis_X.transpose(), new_X.transpose() )

        pickle.dump([exam, splits], open( output_file, 'wb') )
        print(f"data saved to {output_file}")