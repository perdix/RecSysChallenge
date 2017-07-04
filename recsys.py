# Recommender system for spotify user based on listened tracks
# RecSys Challenge Information Retrieval 2017
# Team: rcmmndrs

#usage
#python <recsys.py> -i <infile> -n <nrrecs> -o <outfile>
#python recsys.py -i ir2017.csv -n 10 -o results.csv

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

from recommender_system import RecommenderSystem
from recommenders.recommender_content_based import RecommenderContentBased
from recommenders.recommender_user_based import RecommenderUserBased
from recommenders.recommender_album_based import RecommenderAlbumBased
import logging
import os


def get_precision(result_test_set, result_recommender):
    if len(result_recommender) == 0:
        return 0
    else:
        return len(set(result_recommender).intersection(result_test_set))/len(result_recommender)

def get_recall(result_test_set, result_recommender):
    if len(result_test_set) == 0:
        return 0
    else:
        return len(set(result_recommender).intersection(result_test_set))/len(result_test_set)


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()

parser.add_argument("-i", help="Name of the infile")
parser.add_argument("-n", help="Number of recommended songs per user")
parser.add_argument("-o", help="Name of the outfile")

# Read the arguments and
args = parser.parse_args()
if args.i == None:
    infile = 'ir2017.csv'
else:
    infile = args.i
if args.n == None:
    n = 100
else:
    n = int(args.n)
if args.o == None:
    outfile = 'tmp/results.csv'
else:
    outfile = args.o

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('tmp'):
    os.makedirs('tmp')


rec_sys = RecommenderSystem(
    infile,
    [
        RecommenderUserBased(),
        RecommenderContentBased(),
        RecommenderAlbumBased()
    ]
)

# Read training and test data
# plays_test = pd.read_csv(infile.replace('training', 'test'))
# # add test to dict
# test_dict = {}
# groups = plays_test.groupby('user')
# for i, group in groups:
#     user_id = group['user'].iloc[0]
#     track_ids = set(group['track'])
#     test_dict[user_id] = track_ids


# Recommendation
user_dict_result = {}
#precisions = {}
#recalls = {}

logging.info('Rec-Ctrlr.: Starting recommending.')
plays = pd.read_csv(infile)
groups = plays.groupby('user')

count = 0
for i, group in groups:
    count += 1

    logging.info('Rec-Ctrlr.: ### Recommending for user ' + str(count) + ' (' + str(len(group)) + ' tracks) ###')
    user = group['user'].iloc[0]
    tracks = set(group['track'])

    user_dict_result[user] = rec_sys.recommend_by_tracks(list(tracks), n)[:n]

    #if not (user in test_dict and user in user_dict_result):
    #    logging.warning('Rec-Ctrlr.: User not not available in the test and/or training data set.')
    #    continue

    #precisions[user] = get_precision(test_dict[user], user_dict_result[user])
    #recalls[user] = get_recall(test_dict[user], user_dict_result[user])
    #true_positives = len(set(user_dict_result[user]).intersection(test_dict[user]))
    #logging.info('Rec-Ctrlr.: Results for user ' + str(count) + ': True positives=' + str(true_positives) + ' Precision=' + str(precisions[user]) + ';  Recall=' + str(recalls[user]))

logging.info('Rec-Ctrlr.: Finished recommending.')


# Write recommendations to File
df = pd.DataFrame(columns=['track', 'user'])
for user, tracks in user_dict_result.items():
    size = len(tracks)
    new = pd.DataFrame({'user':[user]*size,'track':tracks})
    df = df.append(new)
df.to_csv(outfile, index=False)
logging.info('Rec-Ctrlr.: Recommendations saved to file.')
