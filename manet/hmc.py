import numpy as np

class HMC:
    def __init__( self, p0, transition, emission, alphabet=np.uint16 ):
        self.n_y = emission.shape[0]
        self.n_x = emission.shape[1]
        self.transition = transition
        self.emission = emission
        self.p0 = p0
        self.alphabet = alphabet
        
        
    def generate( self, length ):
        
        X = np.zeros(length,dtype = self.alphabet)
        Y = np.zeros(length,dtype = self.alphabet)
        
        Y[0] = np.random.choice( self.n_y,1, p=self.p0)
        X[0] = np.random.choice( self.n_x,1, p=self.emission[Y[0],:])
        for t in range(1, length ):
            Y[t] = np.random.choice( self.n_y,1, p=self.transition[Y[t-1],:] )
            X[t] = np.random.choice( self.n_x,1, p=self.emission[Y[t],:])
        
        return X, Y
    
    
    def map( self, X ):
        
        length = len( X )
        Y = np.zeros( (self.n_y,length), dtype = self.alphabet)
        F = np.zeros( (self.n_y, length) )
        
        G = np.log( self.transition )
        Q = np.zeros( (self.n_y, length) )
        Q[:,0] = np.log( self.p0 )
        
        for t in range( length):
            Q[:,t] = Q[:,t] + np.log( self.emission[:,X[t]] )
        
        F[:,0] = Q[:,0]
        for t in range(1,length):
            for y in range(self.n_y):
                yy = np.argmax( G[:,y] + F[:,t-1] )
                F[y,t] = G[yy,y] + F[yy,t-1] + Q[y,t]
                Y[y,t] = yy
        
        Ymap = np.zeros( length , dtype = self.alphabet)
        
        Ymap[length-1] = np.argmax( F[:,length-1] )
        log_map = F[Ymap[length-1], length-1 ]

        for t in range( length-1,0,-1):
            Ymap[t-1] = Y[ Ymap[t],t]
        
        return Ymap, log_map
    
    def decode( self, X ):
        n = len( X )
        
        F = np.zeros( (self.n_y,n ))        
        F[:,0] = self.emission[:,X[0]]*self.p0
        for v in range(1,n):
            for y in range( self.n_y):
                F[y,v] = self.emission[y,X[v]]*np.dot(F[:,v-1],self.transition[:,y])
        for v in range(0,n):
            F[:,v] = F[:,v] / np.sum( F[:,v])
            
        B = np.zeros( (self.n_y, n))
        B[:,n-1] = np.ones(self.n_y)        
        for v in range(n-2,-1,-1):
            for y in range( self.n_y):
                B[:,v] += self.emission[y,X[v+1]]*self.transition[:,y]*B[y,v+1]

        P = F*B
        for v in range(0,n):
            P[:,v] = P[:,v] / np.sum( P[:,v])
                
                            
        return P

