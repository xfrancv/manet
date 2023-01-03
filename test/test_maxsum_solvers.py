#!/usr/bin/env python3
# IDE might complain with "no module found" here, even when it exists

import numpy as np
from manet.maxsum import adag
from manet.maxsum import viterbi
from math import ceil


if __name__ == "__main__":

    ### Simple chain
#    Q = np.array([[3, 2, 2], [-1,2,0]])
#    G = np.array([[[-1,0],[0,-2]], [[0,-1],[0,-1]]]  )
#    E = np.array( [[0,1],[1,2],[0,1]] )
    Q = np.array([[3, 2, 2], [-1,2,0]])
    G = np.array( [[-1,4],[2,-2]] )
    E = np.array( [[0,1],[1,2],[0,0]] )

    labels, energy = adag( Q, G, E,0 )
    print(f"energy={energy}")
    print( labels )

    labels, energy = viterbi( Q, G)
    print(f"energy={energy}")
    print( labels )

    ### Sudoku
    K = 1
    sudoku = "004300209005009001070060043006002087190007400050083000600000105003508690042910300"
    sudoku = np.array([char for char in sudoku]).astype(int)  #.reshape(9,9)
    Q = -K*np.ones([9,9*9] )
    for i,x in enumerate( sudoku ):
        if x == 0:
            Q[:,i] = 0
        else:
            Q[x-1,i] = 0
    G = -K*np.identity( 9 )
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
    E = np.concatenate( (np.array(E).transpose(), np.zeros((1,len(E)),dtype=int)) ,axis=0 )

    labels, energy = adag( Q, G, E, 0 )
    print(f"energy={energy}")
    print( sudoku.reshape(9,9))
    print( labels.reshape(9,9)+1 )

