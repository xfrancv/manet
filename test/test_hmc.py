import numpy as np
from manet.hmc import HMC
from tqdm import tqdm


if __name__ == "__main__":

    n  = 30      # number of symbols = number of hidden states
    length = 100 # sequence length
    alpha = 0.7
    beta = 0.7

    trans = (1-beta)*np.ones((n,n)) / (n-1)
    np.fill_diagonal( trans, beta ) 
    emis = (1-alpha)*np.ones( (n,n )) / (n - 1)
    np.fill_diagonal( emis, alpha )
    p0 = np.ones( n ) / n

    # 
    Hmc = HMC( p0, trans, emis)

    #
    m = 10000

    risk_map = 0
    risk_decode = 0
    for i in tqdm( range( m )):
        X, Y = Hmc.generate( length )

        #
        Ymap, log_map = Hmc.map(X)
        risk_map += np.count_nonzero( Y-Ymap ) 

        # 
        P = Hmc.decode( X )
        Yham = np.argmax( P, axis = 0)
        risk_decode += np.count_nonzero( Y-Yham )
        
    risk_map /= m*length
    risk_decode /= m*length
        
    print(f"Risk(MAP) = {risk_map}")
    print(f"Risk(decode) = {risk_decode}")