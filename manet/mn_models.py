import numpy as np
from scipy.sparse import coo_matrix

# Margin-rescaling loss + Markov Networks + Homogeneous unary and pair-wise pontetionals
# X [n_objects x n_y]
# Y [n_objects]
# E [2 x n_edges ]
class MrMaNetHomo:
    def __init__(self, n_y, X, Y, E, graph='chain'):
        self.alphabet = np.int16
        self.X = X
        self.Y = Y.astype(self.alphabet)
        self.n_objects = np.amax( E ) + 1
        self.n_edges = E.shape[1]
        self.E = np.concatenate( (E.astype(int),np.zeros((1,self.n_edges),dtype=int)) ,axis=0 )
        self.n_feat = X.shape[0]
        self.n_y = n_y
        self.n_params_unary = self.n_feat*self.n_y
        self.n_params_pair = self.n_y**2
        self.n_params = self.n_params_unary + self.n_params_pair
        self.alpha12 = np.zeros((self.n_y,self.n_edges))        
        self.alpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.mAlpha12 = np.zeros((self.n_y,self.n_edges))        
        self.mAlpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.vAlpha12 = np.zeros((self.n_y,self.n_edges))        
        self.vAlpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.graph = graph

        self.grad_alpha12 = np.zeros((self.n_y,self.n_edges))    
        self.grad_alpha21 = np.zeros((self.n_y,self.n_edges))
        self.grad_weights = np.zeros(self.n_params)

        # Hamming loss
        self.Loss = np.ones( (self.n_y,self.n_objects)) / self.n_objects
        for v in range( self.n_objects):
            self.Loss[self.Y[v],v] = 0
                                            
        # features for true labeling
        self.phi_true = np.zeros( self.n_feat+self.n_y**2 )
                        
        # R12, R21 - helper matrices for fast reparametrization of unary scores
        row12 = []
        col12 = []
        row21 = []
        col21 = []
        self.N12 = []
        self.N21 = []
        for v in range(self.n_objects):
            N12 = np.where( self.E[0,:]==v)[0]
            self.N12.append( N12 )
            for e in N12:
                row12.append(e)
                col12.append(v)
                
            N21 = np.where(self.E[1,:]==v)[0]
            self.N21.append( N21 )
            for e in N21:
                row21.append(e)
                col21.append(v)
        
        self.R12 = coo_matrix((np.ones(len(row12)),(row12,col12)),shape=(self.n_edges,self.n_objects)).tocsr()
        self.R21 = coo_matrix((np.ones(len(row21)),(row21,col21)),shape=(self.n_edges,self.n_objects)).tocsr()

    # set ground truth features
    def set_ground_truth_features(self, pz):

        phi_unary = np.zeros( (self.n_feat, self.n_y))
        for v in range(self.n_objects):
            if self.Y[v] >= 0:
                phi_unary[:,self.Y[v]] += self.X[:,v]/pz[v]

        phi_pair = np.zeros((self.n_y,self.n_y))
        for e in range(self.n_edges):
            v0 = self.E[0,e]
            v1 = self.E[1,e]
            if self.Y[v0] >=0 and self.Y[v1] >=0:
                phi_pair[self.Y[v0],self.Y[v1]] += 1/( pz[v0]*pz[v1] )

        self.phi_true = np.concatenate( (phi_unary.reshape(-1),phi_pair.reshape(-1) ))

    # Q [ n_y, n_objects]  
    def get_unary_scores( self, weights ):
        W = np.reshape( weights[0:self.n_params_unary], (self.n_feat, self.n_y) )
        Q = np.matmul( W.T, self.X)
        return Q
    
    # G [ n_y x n_y]
    def get_pair_scores( self, weights):
        G = weights[self.n_params_unary:].reshape( self.n_y, self.n_y )
        return G
    
    def eval_loss( self, weights, compute_grad=True ):                
        # 
        loss = -np.dot( weights, self.phi_true )

        self.grad_alpha12.fill( 0 )
        self.grad_alpha21.fill( 0 )
        
        grad_weights_unary = np.zeros( (self.n_feat, self.n_y))
        grad_weights_pair = np.zeros( (self.n_y,self.n_y))

        Q = self.Loss + self.get_unary_scores( weights ) - self.alpha12*self.R12 - self.alpha21*self.R21
        G = self.get_pair_scores( weights )

        Y = np.argmax( Q, axis=0 )
        if compute_grad:
            # compute loss and its gradient
            for v in range( self.n_objects ):
                loss += Q[Y[v],v]
                grad_weights_unary[:,Y[v]] += self.X[:,v]
                self.grad_alpha12[Y[v],self.N12[v]] -= 1
                self.grad_alpha21[Y[v],self.N21[v]] -= 1

            # reparametrize pair functions, compute loss and its gradient
            for e in range( self.n_edges ):
                R1 = self.alpha12[:,e].reshape((self.n_y,1))*np.ones((1,self.n_y)) 
                R2 = np.ones((self.n_y,1))*self.alpha21[:,e].reshape((1,self.n_y))
                G_rep = G + R1 + R2

                y,yy = np.unravel_index(np.argmax( G_rep ),shape=(self.n_y,self.n_y))
                loss += G_rep[y,yy]
                
                grad_weights_pair[y,yy] += 1
                self.grad_alpha12[y,e] += 1
                self.grad_alpha21[yy,e] += 1

            self.grad_weights[0:self.n_params_unary] = grad_weights_unary.reshape(-1)
            self.grad_weights[self.n_params_unary:] = grad_weights_pair.reshape(-1)
            self.grad_weights -= self.phi_true
        else:
            for v in range( self.n_objects ):
                loss += Q[Y[v],v]

            for e in range( self.n_edges ):
                R1 = self.alpha12[:,e].reshape((self.n_y,1))*np.ones((1,self.n_y)) 
                R2 = np.ones((self.n_y,1))*self.alpha21[:,e].reshape((1,self.n_y))
                G_rep = G + R1 + R2

                y,yy = np.unravel_index(np.argmax( G_rep ),shape=(self.n_y,self.n_y))
                loss += G_rep[y,yy]

                        
        return loss


# Adversarial + Markov Networks + Homogeneous unary and pair-wise pontetionals
# X [n_objects x n_y]
# Y [n_objects]
# E [2 x n_edges ]
class AdvMaNetHomo:
    def __init__(self, n_y, X, Y, E, graph='chain'):
        self.alphabet = np.int16
        self.X = X
        self.Y = Y.astype(self.alphabet)
        self.n_objects = np.amax( E ) + 1
        self.n_edges = E.shape[1]
        self.E = np.concatenate( (E.astype(int),np.zeros((1,self.n_edges),dtype=int)) ,axis=0 )
        self.n_feat = X.shape[0]
        self.n_y = n_y
        self.n_params_unary = self.n_feat*self.n_y
        self.n_params_pair = self.n_y**2
        self.n_params = self.n_params_unary + self.n_params_pair
        self.alpha12 = np.zeros((self.n_y,self.n_edges))        
        self.alpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.mAlpha12 = np.zeros((self.n_y,self.n_edges))        
        self.mAlpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.vAlpha12 = np.zeros((self.n_y,self.n_edges))        
        self.vAlpha21 = np.zeros((self.n_y,self.n_edges)) 
        self.graph = graph

        self.grad_alpha12 = np.zeros((self.n_y,self.n_edges))    
        self.grad_alpha21 = np.zeros((self.n_y,self.n_edges))
        self.grad_weights = np.zeros(self.n_params)

        # Hamming loss
        self.Loss = np.ones( (self.n_y,self.n_objects)) / self.n_objects
        for v in range( self.n_objects):
            self.Loss[self.Y[v],v] = 0
                                            
        # features for true labeling
        self.phi_true = np.zeros( self.n_feat+self.n_y**2 )
                        
        # R12, R21 - helper matrices for fast reparametrization of unary scores
        row12 = []
        col12 = []
        row21 = []
        col21 = []
        self.N12 = []
        self.N21 = []
        for v in range(self.n_objects):
            N12 = np.where( self.E[0,:]==v)[0]
            self.N12.append( N12 )
            for e in N12:
                row12.append(e)
                col12.append(v)
                
            N21 = np.where(self.E[1,:]==v)[0]
            self.N21.append( N21 )
            for e in N21:
                row21.append(e)
                col21.append(v)
        
        self.R12 = coo_matrix((np.ones(len(row12)),(row12,col12)),shape=(self.n_edges,self.n_objects)).tocsr()
        self.R21 = coo_matrix((np.ones(len(row21)),(row21,col21)),shape=(self.n_edges,self.n_objects)).tocsr()

    # set ground truth features
    def set_ground_truth_features(self, pz):

        phi_unary = np.zeros( (self.n_feat, self.n_y))
        for v in range(self.n_objects):
            if self.Y[v] >= 0:
                phi_unary[:,self.Y[v]] += self.X[:,v]/pz[v]

        phi_pair = np.zeros((self.n_y,self.n_y))
        for e in range(self.n_edges):
            v0 = self.E[0,e]
            v1 = self.E[1,e]
            if self.Y[v0] >=0 and self.Y[v1] >=0:
                phi_pair[self.Y[v0],self.Y[v1]] += 1/( pz[v0]*pz[v1] )

        self.phi_true = np.concatenate( (phi_unary.reshape(-1),phi_pair.reshape(-1) ))


    # Q [ n_y, n_objects]  
    def get_unary_scores( self, weights ):
        W = np.reshape( weights[0:self.n_params_unary], (self.n_feat, self.n_y) )
        Q = np.matmul( W.T, self.X)
        return Q
    
    # G [ n_y x n_y]
    def get_pair_scores( self, weights):
        G = weights[self.n_params_unary:].reshape( self.n_y, self.n_y )
        return G
    
    def eval_loss( self, weights, compute_grad=True ):                
        # 
        loss = -np.dot( weights, self.phi_true )

        self.grad_alpha12.fill( 0 )
        self.grad_alpha21.fill( 0 )
        
        grad_weights_unary = np.zeros( (self.n_feat, self.n_y))
        grad_weights_pair = np.zeros( (self.n_y,self.n_y))

#        Q = self.Loss + self.get_unary_scores( weights ) - self.alpha12*self.R12 - self.alpha21*self.R21
        Q = self.get_unary_scores( weights ) - self.alpha12*self.R12 - self.alpha21*self.R21
        G = self.get_pair_scores( weights )

#       Y = np.argmax( Q, axis=0 )
        K = 1/self.n_objects
        if compute_grad:
            # compute loss and its gradient
            for v in range( self.n_objects ):
                order = np.argsort( -Q[:,v] )
                max_val = np.NINF
                acc = 0
                for i,y in enumerate( order ):
                    acc += Q[y,v]
                    val = (acc+(i+1)*K-K )/(i+1)
                    if val > max_val:
                        max_val = val
                        n = i+1
                    else:
                        break

                loss += max_val
                for y in order[0:n]:
                    grad_weights_unary[:,y] += self.X[:,v]/n
                    self.grad_alpha12[y,self.N12[v]] -= 1/n
                    self.grad_alpha21[y,self.N21[v]] -= 1/n


            # reparametrize pair functions, compute loss and its gradient
            for e in range( self.n_edges ):
                R1 = self.alpha12[:,e].reshape((self.n_y,1))*np.ones((1,self.n_y)) 
                R2 = np.ones((self.n_y,1))*self.alpha21[:,e].reshape((1,self.n_y))
                G_rep = G + R1 + R2

                y,yy = np.unravel_index(np.argmax( G_rep ),shape=(self.n_y,self.n_y))
                loss += G_rep[y,yy]
                
                grad_weights_pair[y,yy] += 1
                self.grad_alpha12[y,e] += 1
                self.grad_alpha21[yy,e] += 1

            self.grad_weights[0:self.n_params_unary] = grad_weights_unary.reshape(-1)
            self.grad_weights[self.n_params_unary:] = grad_weights_pair.reshape(-1)
            self.grad_weights -= self.phi_true
        else:
            for v in range( self.n_objects ):
                order = np.argsort( -Q[:,v] )
                max_val = np.NINF
                acc = 0
                for i,y in enumerate( order ):
                    acc += Q[y,v]
                    val = (acc+(i+1)*K-K )/(i+1)
                    if val > max_val:
                        max_val = val
                        n = i+1
                    else:
                        break

                loss += max_val

            for e in range( self.n_edges ):
                R1 = self.alpha12[:,e].reshape((self.n_y,1))*np.ones((1,self.n_y)) 
                R2 = np.ones((self.n_y,1))*self.alpha21[:,e].reshape((1,self.n_y))
                G_rep = G + R1 + R2

                y,yy = np.unravel_index(np.argmax( G_rep ),shape=(self.n_y,self.n_y))
                loss += G_rep[y,yy]

                        
        return loss

