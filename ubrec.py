# Recommender system module
# Implementation of an user based recommedation filter
# Team: rcmmndrs
# Paul Opitz


import pandas as pd
import numpy as np
import sys
from scipy.sparse import *
from scipy import *
import pickle
from sklearn.metrics import euclidean_distances



def fit(csvfile):
 	# read training set
	training = pd.read_csv("csvfile")[['user', 'track']]
	training.head()


	tracks = training['track'].unique()
	tracks_size = len(tracks)

	users = training['user'].unique()
	users_size = len(users)


	#items = np.zeros((size,size))
	users_sparse = csr_matrix( (users_size,tracks_size), dtype=int16 )


	track_id_dict = {}
	id_track_dict = {}
	for i,track in enumerate(tracks):
	    track_id_dict[track] = i
	    id_track_dict[i] = track
	    

	user_id_dict = {}
	id_user_dict = {}
	for i,user in enumerate(users):
	    user_id_dict[user] = i
	    id_user_dict[i] = user
	    
	    
	with open('track_to_id.pickle', 'wb') as handle:
	    pickle.dump(track_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
	with open('id_to_track.pickle', 'wb') as handle:
	    pickle.dump(id_track_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)     
	    
	with open('user_to_id.pickle', 'wb') as handle:
	    pickle.dump(user_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
	with open('id_to_user.pickle', 'wb') as handle:
	    pickle.dump(id_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  

	# Create user dict
	user_dict = {}
	groups = training.groupby('user')
	for i, group in groups:
	    user_id = group['user'].iloc[0]
	    track_ids = set(group['track'])
	    user_dict[user_id] = track_ids    


	 # Generate user item matrix
	count = 0
	for user, tracks in user_dict.items():
	    track_ids = [track_id_dict[track] for track in tracks]
	    user_id = user_id_dict[user]
	    sys.stdout.write("User: "+str(count)+"\r")
	    sys.stdout.flush()
	    count += 1
	    if len(track_ids) < 500 and len(track_ids) > 25:
	        for track_id in track_ids:
	            #print(track_id_dict[track_id])
	            row = array([user_id for i in track_ids])
	            col = array(track_ids)
	            data = array([1 for i in track_ids])
	            acc = csr_matrix( (data,(row,col)), shape=(users_size,tracks_size), dtype=int16 )
	            users_sparse += acc   

	save_npz('users_sparse.npz', users_sparse)


def recommend_to_file(infile, n, outfile):
	print("not yet implemented")




def recommend(tracks, n):
	#result = set()
	users_sparse = load_npz('items_sparse.npz')
	
	with open('track_to_id.pickle', 'rb') as handle:
		track_id_dict = pickle.load(handle)
	with open('id_to_track.pickle', 'rb') as handle:
		id_track_dict = pickle.load(handle)

	tracks_size = len(track_id_dict.keys())

	track_ids = [track_id_dict[track] for track in tracks]

    row = array([0]*len(track_ids))
	col = array(track_ids)
	data = array([1 for i in track_ids])
	input_sparse = csr_matrix( (data,(row,col)), shape=(1,tracks_size), dtype=int16 )

	result_matrix = pairwise_distances(input_sparse, users_sparse)
	#get best 3, add up and get ids
	# remove given tracks_ids












