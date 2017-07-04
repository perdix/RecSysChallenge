# User-based collaborative filtering recommender class
# Implementation of an user based recommedation filter
# Team: rcmmndrs
# Paul Opitz


from scipy.sparse import *
from scipy import *
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from recommenders.recommender_base import RecommenderBase
from pathlib import Path
import logging
import sys


class RecommenderUserBased(RecommenderBase):

    def __init__(self):
        super().__init__()
        self.name = 'User-based'

    def setup(self, input_dataframe):
        users_reduced = Path('cache/users_reduced.npy')
        if not users_reduced.exists():
            logging.info(self.name + ': ' + 'Starting training.')
            self.fit(input_dataframe)
            logging.info(self.name + ': ' + 'Ended training.')
        else:
            logging.info(self.name + ': ' + 'Already trained.')

        self.is_setup = True


    def fit(self, input_dataframe):
        # read training set
        training = input_dataframe[['user', 'track']]

        tracks = training['track'].unique()
        tracks_size = len(tracks)

        users = training['user'].unique()
        users_size = len(users)

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

        with open('cache/track_to_id.pickle', 'wb') as handle:
            pickle.dump(track_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/id_to_track.pickle', 'wb') as handle:
            pickle.dump(id_track_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('cache/user_to_id.pickle', 'wb') as handle:
            pickle.dump(user_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/id_to_user.pickle', 'wb') as handle:
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
            sys.stdout.write("User: " + str(count) + "\r")
            sys.stdout.flush()
            count += 1
            track_ids = track_ids[:1000]
            for track_id in track_ids:
                #print(track_id_dict[track_id])
                row = array([user_id for i in track_ids])
                col = array(track_ids)
                data = array([1 for i in track_ids])
                acc = csr_matrix( (data,(row,col)), shape=(users_size,tracks_size), dtype=int16 )
                users_sparse += acc

        # Apply SVD (dimensionality reduction to 300)
        svd = TruncatedSVD(n_components=50, algorithm='arpack')
        svd.fit(users_sparse)
        users_reduced = svd.transform(users_sparse)

        with open('cache/users_svd.pickle', 'wb') as handle:
            pickle.dump(svd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save('cache/users_reduced.npy', users_reduced)
        save_npz('cache/users_sparse.npz', users_sparse)


    def recommend_by_tracks(self, tracks, n):
        tracks = list(tracks)[:1000]

        users_sparse = load_npz('cache/users_sparse.npz')
        users_reduced = np.load('cache/users_reduced.npy')

        with open('cache/track_to_id.pickle', 'rb') as handle:
            track_id_dict = pickle.load(handle)
        with open('cache/id_to_track.pickle', 'rb') as handle:
            id_track_dict = pickle.load(handle)
        with open('cache/users_svd.pickle', 'rb') as handle:
            svd = pickle.load(handle)

        tracks_size = users_sparse.shape[1]

        track_ids = [track_id_dict[track] for track in tracks if track in track_id_dict.keys()]

        row = array([0] * len(track_ids))
        col = array(track_ids)
        data = array([1 for i in track_ids])

        user_sparse = csr_matrix((data, (row, col)), shape=(1, tracks_size), dtype=int16)
        user_reduced = svd.transform(user_sparse)

        user_sim = cosine_similarity(user_reduced, users_reduced)


        indices = user_sim[0].nonzero()[0].tolist()
        values = user_sim[0].data.tolist()
        zipped = list(zip(indices, values))
        zipped.sort(key=lambda item: item[1], reverse=True)
        # take best 25 users
        user_ids = [i[0] for i in zipped[:25]]
        # add up
        result_sparse = csr_matrix((1, tracks_size), dtype=int8)
        for user_id in user_ids:
            row_sparse = users_sparse.getrow(user_id)
            result_sparse += row_sparse

        indices = result_sparse.nonzero()[1].tolist()
        values = result_sparse.data.tolist()

        tracks = [id_track_dict[i] for i in indices]
        zipped = list(zip(tracks, values))
        filtered = list(filter(lambda item: item[0] not in track_ids, zipped))
        filtered.sort(key=lambda item: item[1], reverse=True)
        result = filtered[:n]

        return result
