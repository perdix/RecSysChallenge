import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances

tracks_with_names = pd.read_csv('tracks_with_names.csv', index_col=0)

track_feature_data = tracks_with_names.drop(['name', 'artist'], axis=1)

track_similarity_matrix = euclidean_distances(track_feature_data.iloc[0:500])

def get_row_index_for_track_id(track_id):
    return track_feature_data.index.get_loc(track_id)

def get_track_id_for_row(row):
    return track_feature_data.index[row]

def recommend(track_id, n):
    similar_tracks = track_similarity_matrix[get_row_index_for_track_id(track_id)]
    recommended_indices = np.argsort(similar_tracks)
    recommended_tracks = [get_track_id_for_row(index) for index in recommended_indices]
    return recommended_tracks[1:n+1]

recommended_tracks = recommend(track_feature_data.index[5], 100)

for index, track_id in enumerate(recommended_tracks):
    print(str(index + 1) + '. track: ' +
          tracks_with_names.ix[track_id, 'artist'] + ' - ' + tracks_with_names.ix[track_id, 'name']
    )
