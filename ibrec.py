# Recommender system module
# Implementation of Item based recommedation filter
# Team: rcmmndrs
# Paul Opitz


import pandas as pd
import numpy as np
import sys
from scipy.sparse import *
from scipy import *
import pickle


def fit(csvfile):
	training = pd.read_csv(csvfile)[['user', 'track']]

	tracks = training['track'].unique()
	size = len(tracks)

	# generate track id dictionaries
	track_id_dict = {}
	id_track_dict = {}

	for i, track in enumerate(tracks):
		track_id_dict[track] = i
		id_track_dict[i] = track

	with open('cache/track_to_id.pickle', 'wb') as handle:
		pickle.dump(track_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
	with open('cache/id_to_track.pickle', 'wb') as handle:
		pickle.dump(id_track_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  

	# generate user track colletions
	user_dict = {}
	groups = training.groupby('user')
	for i, group in groups:
		user_id = group['user'].iloc[0]
		track_ids = set(group['track'])
		user_dict[user_id] = track_ids    	

	# generate sparse item2item matrix
	items_sparse = csr_matrix( (size,size), dtype=int16 )
	count = 0
	for tracks in user_dict.values():
		track_ids = [track_id_dict[track] for track in tracks]
		sys.stdout.write("User: "+str(count)+" --- "+str(len(track_ids))+" tracks \r")
		sys.stdout.flush()
		count += 1
		if len(track_ids) < 500 and len(track_ids) > 30:
			for track_id in track_ids:
				#print(track_id_dict[track_id])
				row = array([track_id for i in track_ids])
				col = array(track_ids)
				data = array([1 for i in track_ids])
				acc = csr_matrix( (data,(row,col)), shape=(size,size), dtype=int16 )
				items_sparse += acc
	save_npz('cache/items_sparse.npz', items_sparse)
		   
				
			 



def recommend_to_file(infile, n, outfile):

	# read input file
	user_dict = {}
	result_dict = {}
	training = pd.read_csv(infile)[['user', 'track']]
	groups = training.groupby('user')
	for i, group in groups:
		user_id = group['user'].iloc[0]
		tracks = set(group['track'])
		user_dict[user_id] = tracks

	# get dictionaries and items
	with open('cache/track_to_id.pickle', 'rb') as handle:
		track_id_dict = pickle.load(handle)
	with open('cache/id_to_track.pickle', 'rb') as handle:
		id_track_dict = pickle.load(handle)

	items_sparse = load_npz('cache/items_sparse.npz')


	for user, tracks in user_dict.items():

		id_list = [track_id_dict[track] for track in tracks if track in track_id_dict]
		size = items_sparse.shape[0]

		result_sparse = csr_matrix( (1,size), dtype=int8 )
		for track_id in id_list:
			row_sparse = items_sparse.getrow(track_id)
			result_sparse += row_sparse

		indices = result_sparse.nonzero()[1].tolist()
		values =  result_sparse.data.tolist()

		zipped = list(zip(indices, values))
		filtered = list(filter(lambda item: item[0] not in id_list, zipped))
		filtered.sort(key=lambda item: item[1], reverse=True)
		shortened = filtered[:n]
		result = [id_track_dict[i[0]] for i in shortened]
		result_dict[user] = result

	#write to outfile
	df = pd.DataFrame(columns=['track', 'user'])
	for user, tracks in result_dict.items():
		size = len(tracks)
		new = pd.DataFrame({'user':[user]*size,'track':tracks})
		df = df.append(new)
	df.to_csv(outfile, index=False)




def recommend(tracks, n):

	with open('cache/track_to_id.pickle', 'rb') as handle:
		track_id_dict = pickle.load(handle)
	with open('cache/id_to_track.pickle', 'rb') as handle:
		id_track_dict = pickle.load(handle)

	track_ids = [track_id_dict[track] for track in tracks if track in track_id_dict]
	
	items_sparse = load_npz('cache/items_sparse.npz')
	size = items_sparse.shape[0]


	result_sparse = csr_matrix((1,size), dtype=int8)
	for track_id in track_ids:
		row_sparse = items_sparse.getrow(track_id)
		result_sparse += row_sparse

	indices = result_sparse.nonzero()[1].tolist()
	values =  result_sparse.data.tolist()
	zipped = list(zip(indices, values))
	filtered = list(filter(lambda item: item[0] not in track_ids, zipped))
	filtered.sort(key=lambda item: item[1], reverse=True)
	shortened = filtered[:n]
	result = [id_track_dict[i[0]] for i in shortened]

	return result

