# Script to generate a tracks dataset that contains not only the audio features, but also artist and track name
# Team: rcmmndrs
# Martin Plattner

import pandas as pd

# load original dataset
tracks_original = pd.read_csv(
    '../tracks_unique.csv',
    index_col=0
)

# load additional dataset with track names and artist names that were fetched using the Spotify API
tracks_metadata = pd.read_csv(
    '../spotify_api_fetcher/tracks_metadata.csv',
    names=['track','name', 'artist'],
    header = 0,
    encoding='utf-8'
)

# prepend the string 'spotify:track:' to track IDs
tracks_metadata['track'] = 'spotify:track:' + tracks_metadata['track'].astype(str)

# drop duplicates, as some tracks were fetched twice (for unknown reasons)
tracks_metadata = tracks_metadata.drop_duplicates(subset='track')

# set the index to allow the subsequent join
tracks_metadata = tracks_metadata.set_index('track')

# join both datasets and write to csv
tracks_original_and_metadata = tracks_original.join(tracks_metadata)
tracks_original_and_metadata.to_csv('../tracks_with_names_utf8.csv', encoding='utf-8')
