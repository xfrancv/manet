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


def run(  data_file, output_folder, config, task_id ):

    # load data
    print("loading data...")
    data, splits = pickle.load( open( data_file, "rb" ) )

    # all combination of parameters
    settings = []
    for s in range( len( splits ) ):
        for l in config["lambda"]:
            for n in config["n_examples"]:

                if isinstance(n,int) and n > 0 and n <= len (splits[s]["trn"]):
                    nn = n
                else:
                    nn = len (splits[s]["trn"])

                settings.append( {'split':s, 'lambda': l, 'n_examples': nn,
                   'output_file': f"{output_folder}/model_split{s}_lambda{l}_n{nn}.pkl",
                   'log_file': f"{output_folder}/model_split{s}_lambda{l}_n{nn}.pkl.log",
                   })

    setting = settings[task_id]

    print(f"total number of tasks: {len(settings)}")
    if task_id > len( settings)-1 or task_id < 0:
        print(f"task {task_id} is out of range")
    elif os.path.isfile( setting["log_file"]) or os.path.isfile(setting["output_file"]):
        print(f"task {task_id} is being already processed") 
    else:
        print(f"processing task {task_id}")
        print(f"split: {setting['split']}")
        print(f"lambda: {setting['lambda']}")
        print(f"n_examples: {setting['n_examples']}")

        # create examples
        model = getattr( manet.mn_models, config["model"] )

        split = dict( splits[ setting['split']] )
        split['trn'] = split['trn'][0:setting['n_examples']]

        examples = []
        for i in tqdm( split['trn'] ):
            if data[i]['X'].ndim == 1:
                # vector is assumed to be a sequence of symbols which is 
                # represented by a matrix when i-th column is 
                # # one-hot-encoding of i-th symbol
                examples.append( model( data[i]['n_y'],one_hot(data[i]['n_x'],\
                    data[i]['X']),data[i]['Y'],data[i]['E'], graph=data[i]['graph'] ))
            else:
                examples.append( model( data[i]['n_y'],data[i]['X'],data[i]['Y'],\
                    data[i]['E'], graph=data[i]['graph'] ))

        # init M3N learning algorithm 
        algo = M3N( config )

        # create output folder if needed
        Path( output_folder ).mkdir(parents=True, exist_ok=True)

        # open log file
        log_file = open( setting['output_file'] + ".log", "w")
        start_time = datetime.now()
        time = start_time.strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f"start_time: {time}\n")
        log_file.write(f"#trn_examples: {len(examples)}\n")
        for key, val in zip( config.keys(), config.values() ):
            log_file.write(f"{key}: {val}\n" )
        log_file.flush()

        # call the solver
        W, obj_val = algo.train( examples, setting['lambda'] )

        # report timing 
        end_time = datetime.now()
        duration_in_sec = (end_time - start_time).total_seconds()
        epoch_time_in_sec = duration_in_sec/config['num_epochs']
        duration = get_duration( duration_in_sec )
        time = end_time.strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f"end time: {time}\n")
        log_file.write(f"duration: {duration['days']} days, {duration['hours']} hours," +
            f" {duration['minutes']} minutes, {duration['seconds']} sec\n")
        log_file.write(f"epoch time: {epoch_time_in_sec}[s]\n")

        if len(obj_val)>0:
            log_file.write(f"obj_val: {obj_val[-1]}\n")
        
        log_file.write(f"saving weights to: {setting['output_file']}\n")

        pickle.dump( [W, obj_val, config, duration_in_sec, epoch_time_in_sec], \
            open( setting["output_file"], "wb" ) )

        log_file.close()


#######################
def get_duration( duration_in_sec ):
    
    d = divmod(duration_in_sec,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    return {'days': d[0], 'hours':h[0], 'minutes': m[0], 'seconds': s}


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train Maximal Margin Markov Network classifier.",\
        usage="train config dataset task_id")

    parser.add_argument("config")
    parser.add_argument("dataset")
    parser.add_argument("task_id",type=int)
    
    args = parser.parse_args()

    data_file = f"./data/{args.dataset}.pkl"
    config_file = f"./config/{args.config}.yaml"
    output_folder = f"./results/{args.dataset}/{args.config}"

    # load config
    with open( config_file,'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # 
    run( data_file, output_folder, config, task_id=args.task_id )
