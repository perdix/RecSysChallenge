# Recommender system for spotify user based on listened tracks
# RecSys Challenge Information Retrieval 2017
# Team: rcmmndrs

#usage
#python <recsys.py> -i <infile> -n <nrrecs> -o <outfile>
#python recsys.py -i ir2017.csv -n 10 -o results.csv

import argparse
import recommender_cf_user_based
#import recommender_content_based
import recommender_album_based
import pandas as pd
from pathlib import Path
from collections import Counter
import sys


# Helper method
def predict(tracks):

	result_counter = Counter()
	
	ub_result = recommender_cf_user_based.recommend(tracks, 10*n)
	for track, count in ub_result:
		result_counter.update({track: count})

	# cb_result = recommender_content_based.recommend(tracks, 10*n)
	# for track, count in cb_result:
	# 	result_counter.update({track: count})

	album_result = recommender_album_based.recommend(tracks, 10*n)
	for track, count in album_result:
		result_counter.update({track: count})

	result = [i[0] for i in result_counter.most_common(n)] 
	return result





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
    n = 10
else:
    n = int(args.n)
if args.o == None:
    outfile = 'results.csv'
else:
    outfile = args.o


#Training
users_reduced = Path('cache/users_reduced.npy')
if not users_reduced.exists():
	print("---- Training Start ----")
	recommender_cf_user_based.fit(infile)
	print("---- Training End ----")
else:
	print("---- Already Trained ----")


# Predicting
user_dict_predict = {}
user_dict_result = {}

print("---- Prediction Start ----")
plays = pd.read_csv(infile)
groups = plays.groupby('user')
for i, group in groups:
    user = group['user'].iloc[0]
    tracks = set(group['track'])
    user_dict_predict[user] = tracks

count = 0
for user, tracks in user_dict_predict.items():
	count += 1
	sys.stdout.write("User: "+str(count)+"#"+str(len(tracks))+"\r")
	sys.stdout.flush()
	user_dict_result[user] = predict(tracks)
print("---- Prediction End ----")


# Write Prediction to File
df = pd.DataFrame(columns=['track', 'user'])
for user, tracks in user_dict_result.items():
	size = len(tracks)
	new = pd.DataFrame({'user':[user]*size,'track':tracks})
	df = df.append(new)
df.to_csv(outfile, index=False)
print("---- Predicton Saved To File ----")








