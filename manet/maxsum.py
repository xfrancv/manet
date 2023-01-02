#!/usr/bin/env python3
#####################################################################
#
# Modul implementing max-sum solvers:
#   - Viterbi algorithm on chain
#   - Schlesinger's ADAG solver for generic graphs
#
#####################################################################

from .adag_solver.adag_solver import lib
import numpy as np
import cffi

##########################################################
def viterbi( Q, G ):
    """
    Solve maxsum on a chain.

    Input:
        Q [nK x nT] unary functions
        G [(nT-1) x nK x nK] pair functions
    Output:
        labels [nT] maximal labelling 
        energy [float] 
    """

    n_y, length = Q.shape
    Y = np.zeros( Q.shape, dtype = int )
    F = np.zeros( Q.shape )

    if len( G.shape) == 2:
        G = np.repeat( np.expand_dims(G, axis=0), length-1, axis=0)

    F[:,0] = Q[:,0]
    for t in range( 1, length ):
        for y in range( n_y ):
            yy = np.argmax( G[t-1,:,y] + F[:,t-1] )
            F[y,t] = G[t-1,yy,y] + F[yy,t-1] + Q[y,t]
            Y[y,t] = yy
        
    Y_best = np.zeros( length , dtype = int )

    Y_best[length-1] = np.argmax( F[:,length-1] )
    energy = F[Y_best[length-1],length-1]

    for t in range( length-1,0,-1):
        Y_best[t-1] = Y[ Y_best[t], t ]

    return Y_best, energy

        

#####################################################
def adag( Q, G, E, theta =0 ):
    """
    Solve max-sum problem on a general graph using ADAG.

    Input:
        Q [nK x nT] unary functions
        G [nK x nK x nG ] pair functions
        E [3 x nE] edges between objects
    Output:
        labels [nT] maximal labelling 
        energy [float] 
    """

    #    
    ffi = cffi.FFI()

    #
    nK, nT = Q.shape
    if len( G.shape) == 2:
        G = np.expand_dims(G, axis=0)    
    nG = G.shape[0]
    nE = E.shape[1]

    # ADAG algorithm wroks in fixed point 32bit arithmentic
    # Hence, values of G a Q are rescaled to the interval <-10^8,0>
    # resulting functions are stored in form suitable for CFFI API
    min_value = np.min( [np.amin(Q), np.amin(G)])  
    max_value = np.max( [np.amax(Q), np.amax(G)])
    
    mult_const = 10**8/( max_value-min_value) 
    add_const = -10**8-mult_const*min_value
    #mult_const = 1
    #add_const = 0

    nnz_in_G = np.count_nonzero(G != np.NINF )
    _G = ffi.new("int[]", nnz_in_G*4)
    cnt = 0
    for g in range(nG):
        for k in range(nK):
            for kk in range(nK):
                if G[g,k,kk] != np.NINF:
                    _G[cnt] = g
                    _G[cnt+1] = k
                    _G[cnt+2] = kk
                    _G[cnt+3] = int( mult_const*G[g,k,kk]+add_const)
                    cnt = cnt + 4 
    
    _Q = ffi.new("int[]", nT*nK )
    cnt = 0
    for t in range( nT ):
        for k in range( nK ):
            _Q[cnt] = int( mult_const*Q[k,t]+add_const)
            cnt = cnt + 1

    #
    _E = ffi.new("unsigned int[]", 3*nE )
    cnt = 0
    for e in range( nE):
        _E[cnt] = E[0,e]
        _E[cnt+1] = E[1,e]
        _E[cnt+2] = E[2,e]
        cnt = cnt + 3

    #
    f = ffi.new("int[]", nK*nE*2)
    
    #
    _labels = ffi.new("unsigned char[]", nK*nT )
    _energy = ffi.new("double[]", 1)

    exitflag = lib.adag_maxsum(_labels, _energy, nT, nK, nE, _E, nnz_in_G, _G, _Q, f, theta )

    energy = (np.ceil( _energy[0]) - (nT+nE)*add_const)/mult_const

    labels = np.zeros( nT, dtype=np.uintc )
    cnt = 0
    one_hot = np.zeros( nK)
    for t in range(nT):
        for l in range(nK):
            one_hot[l] = int( _labels[t*nK+l] )
        labels[t] = np.argmax( one_hot )

    #
    return labels, energy

