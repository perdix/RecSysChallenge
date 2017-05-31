import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

tracks_with_names = pd.read_csv('../data/tracks_with_metadata.csv')
tracks_with_names = tracks_with_names.drop(['play_count', 'album_track_number','popularity','explicit_content'], axis=1)

data = pd.DataFrame(columns=['artist', 'cosine_similarity', 'euclidean_distance'])

artist = '50 Cent'

artists_to_compare = ['Lupe Fiasco', 'Europe', 'G-Unit', 'Michael Jackson', 'JAY Z', 'DMX', 'Enya', 'Bonobo', 'Stimming']

# testing different sets of columns (last line one is used)
columns = ['speechiness', 'danceability', 'instrumentalness']
columns = ['energy', 'echo_key', 'liveness', 'speechiness', 'acousticness', 'danceability', 'loudness', 'valence', 'instrumentalness']
columns = ['energy', 'liveness', 'speechiness', 'acousticness', 'danceability', 'valence', 'instrumentalness']


print ("Comparing with '" + artist + "'\n")
# calculate the mean of all tracks of the given artists
artist1_data = (tracks_with_names[tracks_with_names['artist'] == artist].groupby('artist')[columns].mean())
print(artist1_data)

print('Considered features: ' + str(len(columns)) + '\n  - ' + '\n  - '.join(columns) + '\n\n')

for artist_to_compare in artists_to_compare:
    # calculate the mean of all tracks of the given artists
    artist2_data = (tracks_with_names[tracks_with_names['artist'] == artist_to_compare].groupby('artist')[columns].mean())
    print(artist2_data)
    
    data = data.append({
        'artist': artist_to_compare,
        'cosine_similarity': str(cosine_similarity(artist1_data, artist2_data)[0,0]),
        'euclidean_distance': str(euclidean_distances(artist1_data, artist2_data)[0,0])
    }, ignore_index=True)

data = data.append({
    'artist': 'testvector',
    'cosine_similarity': str(cosine_similarity(artist1_data, np.zeros(len(columns)).reshape(1,-1))[0,0]),
    'euclidean_distance': str(euclidean_distances(artist1_data, np.zeros(len(columns)).reshape(1,-1))[0,0])
}, ignore_index=True)


print(data)
print('theend')
