from pathlib import Path
import string
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
import math


from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from scipy.stats import shapiro
output_folder = '../../data/output/'
output_folder = '/home/jamal/Documents/Research/sourcecode/covid-gan/lipizzaner-covidgan/src/output/'


# ESTOS SON BUENOS
output_folder = '/media/jamal/44F89DCFF89DC01A/euro-gp2021-results/ensemble-size-three/'
output_folder = '/media/jamal/44F89DCFF89DC01A/euro-gp2021-results/euro-gp2020-final/'

# output_folder = '/home/jamaltoutouh/semi-supervised/lipizzaner-gan/src/output/'
data_folder = '../../data/'
dataset = 'mnist'  #'circular' #'mnist'

def get_all_master_log_files():
    return [filepath for filepath in glob.iglob(output_folder + 'log/*.log')]


def split_best_result(line):
    splitted_data = re.split(' |\(|,|\)', line)
    return float(splitted_data[-4]), float(splitted_data[-2]), splitted_data[-7]

def split_distributed_result(line):
    splitted_data = re.split(' |\(|,|\)', line)
    return float(splitted_data[-4]), float(splitted_data[-2]), splitted_data[-10]


def get_independent_run_params(file_name):
    parameters = None
    for line in open(file_name, 'r'):
        if 'Parameters: ' in line:
            splitted_data = re.split("Parameters: ", line)
            parameters = json.loads(str(splitted_data[1]).replace("\'", "\"").replace("True", "true").replace("False", "false").replace("None", "null"))
    return parameters

def get_fid_tvd_time_bestclient_from_master_log(master_log_path):
    fid, init_time = None, None
    fids, tvds, clients = list(), list(), list()
    for line in open(master_log_path, 'r'):
        splitted_data = re.split("- |,", line)
        if init_time is None:
            init_time = datetime.strptime(splitted_data[0], '%Y-%m-%d %H:%M:%S')
        if 'Stopping heartbeat...' in line:
            stop_time = datetime.strptime(splitted_data[0], '%Y-%m-%d %H:%M:%S')
        if 'Best result:' in line:
            fid, tvd, best_client = split_best_result(line)
        if 'yielded a score of' in line:
            fid, tvd, client = split_distributed_result(line)
            fids.append(fid)
            tvds.append(tvd)
            clients.append(client)
            stop_time = datetime.strptime(splitted_data[0], '%Y-%m-%d %H:%M:%S')

    # print(master_log_path)
    # print(fids)
    # print(fid)

    if not fid is None:
        results = {'client': clients, 'fid': fids, 'tvd': tvds}
        results = pd.DataFrame(results)
        best_results = results.loc[results['fid'] == min(fids)]
        execution_time = stop_time - init_time
        execution_time_minutes = execution_time.total_seconds() / 60
        # return None, None, None, None, None
        return float(best_results.fid), float(best_results.tvd), execution_time_minutes, list(best_results.client)[0], str(init_time) #datetime.strptime(init_time, '%Y-%m-%d %H:%M:%S')
    else:
        return None, None, None, None, None



def get_fid_weight_evolution(master_log_file):
    f = open(master_log_file, 'r')
    line = f.readline()
    scores = list()
    improvement = -1

    improvement, grid_size, score_after, tvd = None, None, None, None
    grid_size = 0


    while line:
        if 'Init score:' in line:
            splitted_data = re.split(" |:|\t", line)
            scores.append(float(splitted_data[11]))
        elif 'Score of new weights:' in line:
            splitted_data = re.split(" |:|\t", line)
            scores.append(float(splitted_data[21]))
        elif 'Successfully started experiment on http' in line:
            grid_size += 1
        elif 'Score after mixture weight optimzation:' in line:
            splitted_data = re.split(" |:|\t|\n", line)
            if 'TVD' in line:
                score_after = float(splitted_data[-6])
                tvd = float(splitted_data[-2])
                score_before = float(splitted_data[-13])
            else:
                score_after = float(splitted_data[-2][:-2])
                tvd = None
                score_before = float(splitted_data[-9])
            improvement = score_before - score_after
            scores.append(float(score_after))
        line = f.readline()

    master_log_filename = master_log_file.split('/')[-1]
    if scores is not None and len(scores)>0:
        pd.DataFrame(scores).to_csv(data_folder + '/evolution/' + dataset + '-fid_ensemble_evolution-evolution-' +
                                      master_log_filename[:-4] + '-{}_grid-'.format(grid_size) + '.csv',
                                      index=False)
        print(data_folder + '/final/' + dataset + '-fid_ensemble_evolution-evolution-evolution-' +
              master_log_filename[:-4] + '.csv')
    return improvement, grid_size, score_after, tvd


def get_stats(values):
    num = np.array(values)
    minn = num.min()
    maxx = num.max()
    mean = num.mean()
    std = num.std()
    return minn, maxx, mean, std

fids, tvds = list(), list()
for logfile in get_all_master_log_files():
    fid, tvd, exec_time, client, init_time = get_fid_tvd_time_bestclient_from_master_log(logfile)

    improvement, grid_size, score_after_weight, tvd_after_weight = get_fid_weight_evolution(
        logfile)  # QUedarnos con el nuevo score si es mejor

    if fid is not None:
        print('------------------------')
        print(logfile)
        print(fid)
        print(score_after_weight)
        print('------------------------')
    if fid is not None and score_after_weight is not None:
        if score_after_weight < fid:
            fid = score_after_weight
            tvd = tvd_after_weight
        fids.append(fid)
        tvds.append(tvd)




print(get_stats(list(filter(None, fids))))
# print(tvds)
print(get_stats(list(filter(None, tvds)) ))



#(23.58998184966481, 36.1939566724343, 29.47170931530286, 5.179761937853182)
#(0.07732371794871795, 0.11077724358974358, 0.09464476495726497, 0.013683179833701722)