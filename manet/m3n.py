import numpy as np
from tqdm import tqdm
from manet.maxsum import viterbi
from manet.maxsum import adag
import pickle
from datetime import datetime

############################################################################
class M3N:
    def __init__( self, config ):
        self.config = config

        if "verb" not in config.keys():
            self.config["verb"] = False

        if "annotation" not in config.keys():
            self.config["annotation"] = "homo"

        if config["solver"] == "adam":
            if "beta1" not in config.keys():
                self.config["beta1"] = 0.9
                self.config["beta2"] = 0.999
            if "epsilon" not in config.keys():
                self.config["epsilon"] = 0.00000001

    def train( self, examples, lam):
        """
        Learn M3N classifier on examples.

        Input:
            examples [list] list of MaxSum problems
            lam [float] non-negative regularization constant
        Output:
            W [np array] weight vector
            obj_val [np array] history of objective function
        """

        ## setup ground truth features
        # get maximal number of objects
        max_n_objects = 0
        for i in range( len( examples )):
            max_n_objects = max( max_n_objects, examples[i].n_objects )

        # get annotation completenes
        n_examples_per_object = np.zeros( max_n_objects ) 
        completeness = np.zeros( max_n_objects )
        for i in range( len( examples )):
            n_objects = examples[i].n_objects
            n_examples_per_object[0:n_objects] += np.ones( n_objects )
            completeness[0:n_objects] += (examples[i].Y >= 0).astype(np.int16)

        # estimate annotation schema
        if self.config["annotation"] == "homo":
            pz = np.ones( max_n_objects ) * completeness.sum() / n_examples_per_object.sum()
        elif self.config["annotation"] == "unhomo":
            nnz = np.nonzero( n_examples_per_object )
            pz = np.zeros( max_n_objects )
            pz = completeness[nnz] / n_examples_per_object[nnz]

        # set ground truth features
        for i in range( len( examples )):
            examples[i].set_ground_truth_features( pz[0:examples[i].n_objects] )

        ## call solver
        if self.config["solver"] == "sgd":
            W, obj_val = m3n_sgd( examples, lam, num_epochs=self.config["num_epochs"], \
                lr=(self.config["lr_const"],self.config["lr_exp"]), \
                eval_obj = self.config["eval_obj"],\
                verb=self.config["verb"])

        elif self.config["solver"] == "adam":
            W, obj_val = m3n_adam( examples, lam, num_epochs=self.config["num_epochs"], \
                lr = (self.config["lr_const"],self.config["lr_exp"]), \
                beta1=self.config["beta1"], beta2=self.config["beta2"], \
                epsilon=self.config["epsilon"], \
                eval_obj = self.config["eval_obj"],\
                verb=self.config["verb"])

        return W, obj_val 
        

    def eval( self, W, examples ):
        """
        Evaluate M3N classifier on examples.

        Input:
            W [np array] wights
            examples [list] list of MaxSum problems
        Output:
            
        """

        #
        loss_hamming = np.zeros( len( examples ) )
        loss_01 = np.zeros( len( examples ) )

        for i in range( len( examples)):
            Q = examples[i].get_unary_scores( W )
            G = examples[i].get_pair_scores( W )
        
            if examples[i].graph == 'chain':
                Y_pred, energy = viterbi( Q, G )
            else:
                Y_pred, energy = adag( Q, G, examples[i].E )

            loss_hamming[i] = np.count_nonzero( np.logical_and(examples[i].Y != Y_pred, examples[i].Y >= 0)) / Y_pred.size
            loss_01[i] = float( loss_hamming[i] != 0 )

        return loss_hamming, loss_01

    def predict( self, W, examples ):

        if type(examples) != list:
            examples = [examples]
            single_input = True
        else:
            single_input = False

        predictions = []
        scores = []
        for i in range( len( examples)):
            Q = examples[i].get_unary_scores( W )
            G = examples[i].get_pair_scores( W )
        
            if examples[i].graph == 'chain':
                Y_pred, energy = viterbi( Q, G )
            else:
                Y_pred, energy = adag( Q, G, examples[i].E )

            predictions.append( Y_pred )
            scores.append( energy )

        if single_input:
            return predictions[0], scores[0]
        else:
            return predictions, scores





def one_hot(n_x, X):
    X_one_hot = np.zeros((n_x,len(X)))
    for t in range( len(X)):
        X_one_hot[X[t],t] = 1.0
    return X_one_hot 

###############################################################################
def m3n_sgd( examples, lam, num_epochs, lr, eval_obj = 1 , verb=False ):
    """
    SGD solver for Markov Network training with LP relaxed loss.
    """

    n_params = examples[0].n_params
    n_examples = len( examples )
    W = np.zeros( n_params )

    if eval_obj > 0:
        obj_val = eval_objective( examples, W )
        obj_hist = [ obj_val ]
    else:
        obj_hist = []

    for epoch in range( num_epochs ):
        _lr = lr[0]*(epoch+1)**lr[1]

        for i in np.random.permutation( n_examples ):

            examples[i].eval_loss( W, compute_grad=True )

            W = W - _lr * (examples[i].grad_weights + lam*W)

            examples[i].alpha12 = examples[i].alpha12 - _lr*examples[i].grad_alpha12
            examples[i].alpha21 = examples[i].alpha21 - _lr*examples[i].grad_alpha21
            
        if epoch == num_epochs-1 or (eval_obj > 0 and (epoch % eval_obj) == 0 ):
            obj_val = eval_objective( examples, W )
            obj_hist.append( obj_val )
            if verb:
                print(f"epoch={epoch} obj_val={obj_val}")

    return W, obj_hist

###############################################################################
def m3n_adam( examples, lam, num_epochs, lr, beta1, beta2, epsilon, eval_obj = 1 , verb=False ):
    """
    ADAM solver for Markov Network training with LP relaxed loss.
    """
    n_params = examples[0].n_params
    n_examples = len( examples )
    W = np.zeros( n_params )
    vW = np.zeros( n_params)
    mW = np.zeros( n_params)

    if eval_obj > 0:
        obj_val = eval_objective( examples, W )
        obj_hist = [ obj_val ]
    else:
        obj_hist = []

    pbeta1 = beta1
    pbeta2 = beta2

    for epoch in range( num_epochs ):
        _lr = lr[0]*(epoch+1)**lr[1]

        for i in np.random.permutation( n_examples ):

            pbeta1 = pbeta1*beta1
            pbeta2 = pbeta2*beta2

            examples[i].eval_loss( W, compute_grad=True )

            gradW = examples[i].grad_weights + lam*W

            mW = mW*beta1+(1-beta1)*gradW
            vW = vW*beta2+(1-beta2)*np.power( gradW, 2)            
            W -= _lr * (mW/(1-pbeta1))/( np.sqrt(vW/(1-pbeta2)) + epsilon )

            examples[i].mAlpha12 = examples[i].mAlpha12*beta1 + (1-beta1)*examples[i].grad_alpha12
            examples[i].vAlpha12 = examples[i].vAlpha12*beta2 + (1-beta2)*np.power(examples[i].grad_alpha12, 2)
            examples[i].alpha12 -= _lr * (examples[i].mAlpha12/(1-pbeta1))/( np.sqrt(examples[i].vAlpha12/(1-pbeta2)) + epsilon ) 

            examples[i].mAlpha21 = examples[i].mAlpha21*beta1 + (1-beta1)*examples[i].grad_alpha21
            examples[i].vAlpha21 = examples[i].vAlpha21*beta2 + (1-beta2)*np.power(examples[i].grad_alpha21, 2)
            examples[i].alpha21 -= _lr * (examples[i].mAlpha21/(1-pbeta1))/( np.sqrt(examples[i].vAlpha21/(1-pbeta2)) + epsilon ) 
            
        # objective is evaluated at the end or according to eval_obj period
        if epoch == num_epochs-1 or (eval_obj > 0 and (epoch % eval_obj) == 0 ):
            obj_val = eval_objective( examples, W )
            obj_hist.append( obj_val )
            if verb:
                print(f"epoch={epoch} obj_val={obj_val}")

    return W, obj_hist


def eval_objective( examples, W ):
    n_examples = len( examples )
    obj = 0.0

    for i in range( n_examples ):
        obj += examples[i].eval_loss( W, compute_grad=False )

    obj = obj / n_examples
    return obj
