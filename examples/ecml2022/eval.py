############################################################
# Training M3N classifiers with various algorithms
#

from pathlib import Path
import os
import numpy as np
import manet
from manet.m3n import M3N
from manet.m3n import one_hot
from manet.mn_models import MrMaNetHomo, AdvMaNetHomo
from tqdm import tqdm
import pickle
from sys import getsizeof
import sys
import yaml
from datetime import datetime
import argparse
from sys import getsizeof

##
def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    values = []
    for i, item in enumerate(iterable, 1):
        values.append(item)
        if i % n == 0:
            yield values
            values = []
    if values:
        yield values


##
def run(  data_file, output_folder, config, chunk_size, task_id, dry_run=False ):

    # load data
    print("loading data...")
    data, splits = pickle.load( open( data_file, "rb" ) )

    # all combination of parameters and chunks
    settings = []
    for s in range( len( splits ) ):
        for l in config["lambda"]:
            for n in config["n_examples"]:
                split = dict( splits[s] )

                if isinstance(n,int) and n > 0 and n <= len (split["trn"]):
                    nn = n
                else:
                    nn = len (split["trn"])

                split['trn'] = split['trn'][0:nn]
                indices = np.concatenate( (split['trn'], split['val'], split['tst'] ) )

                for chunk in chunks( range(len(indices)), chunk_size ):
                    output_sub_folder = f"{output_folder}/model_split{s}_lambda{l}_n{nn}"
                    output_file = f"{output_sub_folder}/chunk_{chunk[0]}_{chunk[-1]}.pkl"
                    model_file = output_sub_folder + ".pkl"

                    settings.append( {'split':s, 'lambda': l, 'n_examples': nn, \
                        'chunk': (chunk[0],chunk[-1]), 'indices': indices[chunk], \
                        'model_file': model_file, 'output_folder': output_sub_folder, \
                        'output_file': output_file } )

    print(f"total number of tasks: {len(settings)}")

    if dry_run:
        missing = 0
        done = 0
        for i, setting in enumerate( settings ):
            if not os.path.isfile( setting['output_file'] ):
                missing += 1
                print(f"MISSING: {setting['output_file']}")
            else:
                done += 1
                print(f"FINISHED: {setting['output_file']}")
        print(f"#tasks: {len(settings)}, #missing tasks: {missing}, #finished tasks: {done}")
        return

    if task_id > len( settings)-1 or task_id < 0:
        print(f"task {task_id} is out of range")
    else:
        setting = settings[task_id]

        if os.path.isfile( setting['output_file'] ):
            print(f"task {task_id} is already processed")
        else:
            print(f"processing task {task_id}")
            print(f"output file: {setting['output_file']}")
            print(f"split: {setting['split']}")
            print(f"lambda: {setting['lambda']}")
            print(f"n_examples: {setting['n_examples']}")
            print(f"chunk: {setting['chunk'][0]} - {setting['chunk'][1]}")

            # create examples
            examples = []
            model = getattr( manet.mn_models, config["model"] )

            for i in tqdm( setting["indices"] ):
                if data[i]['X'].ndim == 1:
                    # vector is assumed to be a sequence of symbols which is 
                    # represented by a matrix when i-th column is 
                    # # one-hot-encoding of i-th symbol
                    examples.append( model( data[i]['n_y'],one_hot(data[i]['n_x'],\
                        data[i]['X']),data[i]['Y'],data[i]['E'], graph=data[i]['graph'] ))
                else:
                    examples.append( model( data[i]['n_y'],data[i]['X'],data[i]['Y'],\
                        data[i]['E'], graph=data[i]['graph'] ))

            # load weights 
            W = pickle.load( open( setting["model_file"], "rb" ) )[0]

            # init M3N learning algorithm 
            algo = M3N( config )

            # create output folder if needed
            Path( setting["output_folder"] ).mkdir(parents=True, exist_ok=True)

            #
            loss_hamming, loss_01 = algo.eval( W, examples )        

            #
            pickle.dump( [loss_hamming, loss_01, setting["indices"]], open( setting["output_file"], "wb" ) )
            print(f"partial results saved to {setting['output_file']}")

    # check if all chunks have been processes; if yes, merge chunks for each setting to a single file
    for s in range( len( splits ) ):
        for l in config["lambda"]:
            for n in config["n_examples"]:
                split = dict( splits[s] )
                if isinstance(n,int) and n > 0 and n <= len (split["trn"]):
                    nn = n
                else:
                    nn = len (split["trn"])

                split['trn'] = split['trn'][0:nn]

                indices = np.concatenate( (split['trn'], split['val'], split['tst'] ) )

                output_sub_folder = f"{output_folder}/model_split{s}_lambda{l}_n{nn}"
                result_file = f"{output_sub_folder}/results.pkl"

                if not os.path.isfile( result_file ):
                    loss_hamming = np.zeros( np.max(indices)+1 )
                    loss_01 = np.zeros( np.max(indices)+1 )
                    missing_chunk = False
                    for chunk in chunks( range(len(indices)), chunk_size ):
                        chunk_file = f"{output_sub_folder}/chunk_{chunk[0]}_{chunk[-1]}.pkl"
                        if not os.path.isfile( chunk_file ):
                            missing_chunk = True
                            break
                        else:
                            partial_results = pickle.load( open( chunk_file, "rb" ) )

                            loss_hamming_ = partial_results[0]
                            loss_01_ = partial_results[1]
                            indices_ = partial_results[2]

                            loss_hamming[indices_] = loss_hamming_
                            loss_01[indices_] = loss_01_

                    if not missing_chunk:
                        results = {
                            'trn_err': np.mean( loss_hamming[split['trn']] ),
                            'val_err': np.mean( loss_hamming[split['val']] ),
                            'tst_err': np.mean( loss_hamming[split['tst']] ),
                            'trn_err_01': np.mean( loss_01[split['trn']] ),
                            'val_err_01': np.mean( loss_01[split['val']] ),
                            'tst_err_01': np.mean( loss_01[split['tst']] )
                            }
                        pickle.dump( results, open( result_file, "wb" ) )
                        print( f"results saved to {result_file}")

                        

#######################
def get_duration( duration_in_sec ):
    
    d = divmod(duration_in_sec,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    return {'days': d[0], 'hours':h[0], 'minutes': m[0], 'seconds': s}


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Evaluate Maximal Margin Markov Network classifier.",\
        usage="eval config dataset chunk_size task_id")

    parser.add_argument("config")
    parser.add_argument("dataset")
    parser.add_argument("chunk_size",type=int)
    parser.add_argument("task_id",type=int)
    parser.add_argument("--dry",action="store_true")
    
    args = parser.parse_args()
    chunk_size = args.chunk_size

    data_file = f"./data/{args.dataset}.pkl"
    solver_config = f"./config/{args.config}.yaml"
    output_folder = f"./results/{args.dataset}/{args.config}"

    # load config
    with open( solver_config,'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # 
    run( data_file, output_folder, config, chunk_size, task_id=args.task_id, dry_run=args.dry )
