import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN

from sklearn.manifold import MDS

tracks_with_names = pd.read_csv('../tracks_with_names.csv', encoding='latin_1')

sample_data = tracks_with_names.sample(1000)
sample_data['genre'] = ''
# add X number of tracks from known artists (eg. Eminem) to visually test clustering
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Eminem'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == '2Pac'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Jay-Z'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Dr. Dre'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == '50 Cent'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'D12'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'DMX'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Xzibit'].iloc[0:25].assign(genre='Hip-Hop'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Snoop Dogg'].iloc[0:25].assign(genre='Hip-Hop'))


sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Stimming'].iloc[0:25].assign(genre='Electronic Music'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Kollektiv Turmstrasse'].iloc[0:25].assign(genre='Electronic Music'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Rodriguez Jr.'].iloc[0:25].assign(genre='Electronic Music'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'David August'].iloc[0:25].assign(genre='Electronic Music'))
sample_data = sample_data.append(tracks_with_names[tracks_with_names['artist'] == 'Solomun'].iloc[0:25].assign(genre='Electronic Music'))




df_features = sample_data[['tempo','energy','echo_key','mode','liveness','speechiness','acousticness','danceability','duration','loudness','valence','instrumentalness']]

mds = MDS(n_components=2, dissimilarity='precomputed')
kmeans = KMeans(n_clusters=8, random_state=0)
kmeans.fit(df_features)

sample_data['cluster'] = pd.Series(kmeans.labels_, index=sample_data.index)

similarities_eculidean = euclidean_distances(df_features)
reduced_data_eculidean = mds.fit(similarities_eculidean, df_features).embedding_

sample_data['reduced_x'] = reduced_data_eculidean[:, 0]
sample_data['reduced_y'] = reduced_data_eculidean[:, 1]

sample_data.to_csv('for_tableau.csv')

print('theend')

