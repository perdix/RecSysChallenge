# Content-based recommender class
# Team: rcmmndrs


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from collections import Counter
from sklearn.preprocessing import normalize
from recommenders.recommender_base import RecommenderBase
from sklearn.preprocessing import StandardScaler

import logging

audio_feature_names = [
    'tempo', 'energy', 'echo_key', 'mode', 'liveness', 'speechiness',
    'acousticness', 'danceability', 'duration', 'loudness', 'valence',
    'instrumentalness'
]


class RecommenderContentBased(RecommenderBase):

    def __init__(self):
        super().__init__()
        self.name = 'Content-b.'

    def setup(self, plays_dataframe):
        self.plays_dataframe = plays_dataframe
        self.tracks_feature_data = self.plays_dataframe.drop('user', axis=1).drop_duplicates('track').set_index('track')
        self.scaler = StandardScaler().fit(self.tracks_feature_data)
        self.tracks_feature_data_normalized = self.scaler.transform(self.tracks_feature_data)

        self.is_setup = True


    def recommend_by_tracks(self, tracks, n):
        tracks = tracks[:1000]

        user_track_features = self.tracks_feature_data.loc[tracks]
        user_track_features_normalized = self.scaler.transform(user_track_features)
        similarity_matrix = euclidean_distances(
            user_track_features_normalized,
            self.tracks_feature_data_normalized
        )
        sorted_similarity_matrix_indices = np.argsort(similarity_matrix, axis=1)

        most_similar_tracks = sorted_similarity_matrix_indices.flatten(order='F')[len(tracks):n+len(tracks)]
        track_ids = self.tracks_feature_data.index[most_similar_tracks]

        return list(zip(track_ids, [1] * len(track_ids)))

