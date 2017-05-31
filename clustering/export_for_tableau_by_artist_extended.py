import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

tracks_with_names = pd.read_csv('../data/tracks_with_metadata.csv')
tracks_with_names = tracks_with_names[(tracks_with_names['duration'] > 120)]

# group by artist, take groups with > x (eg. 10) tracks, then take a sample of y (eg. 100) artists
artists = tracks_with_names.groupby('artist').filter(lambda x: len(x) > 20 and len(x) < 50).groupby('artist').mean().sample(50)
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Eminem'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == '2Pac'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Jay-Z'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Dr. Dre'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == '50 Cent'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'D12'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'DMX'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Xzibit'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Snoop Dogg'].iloc[0:25].groupby('artist').mean().assign(genre='Hip-Hop'))


artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Stimming'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Kollektiv Turmstrasse'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Rodriguez Jr.'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'David August'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Solomun'].iloc[0:25].groupby('artist').mean().assign(genre='Electronic Music'))


artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Britney Spears'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Pink'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))
artists = artists.append(tracks_with_names[tracks_with_names['artist'] == 'Christina Aguilera'].iloc[0:25].groupby('artist').mean().assign(genre='Pop'))

tracks_with_names[tracks_with_names['artist'] == 'Snoop Dogg'].iloc[0:25].groupby('artist').mean()
tracks_with_names[tracks_with_names['artist'] == 'Enya'].iloc[0:25].groupby('artist').mean()


df_features = artists[['tempo','energy','echo_key','mode','liveness','speechiness','acousticness','danceability','duration','loudness','valence','instrumentalness']]


# clustering with kmeans
kmeans = KMeans(n_clusters=8, random_state=0)
kmeans.fit(df_features)

artists['cluster_kmeans'] = pd.Series(kmeans.labels_, index=artists.index)


artists_copy = artists.copy()

# dimensionality reduction with MDS
mds = MDS(n_components=2, dissimilarity='precomputed')

similarities_eculidean = euclidean_distances(df_features)
reduced_data_eculidean = mds.fit(similarities_eculidean, df_features).embedding_

artists['reduced_x'] = reduced_data_eculidean[:, 0]
artists['reduced_y'] = reduced_data_eculidean[:, 1]
artists['data_type'] = 'KMeans (MDS)'


# dimensionality reduction with PCA
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(df_features)

artists_copy['reduced_x'] = reduced_data_pca[:, 0]
artists_copy['reduced_y'] = reduced_data_pca[:, 1]
artists_copy['data_type'] = 'KMeans (PCA)'

artists = artists.reset_index().append(artists_copy.reset_index())


artists.to_csv('for_tableau_by_artist_extended.csv', encoding='utf-8')

print('theend')

