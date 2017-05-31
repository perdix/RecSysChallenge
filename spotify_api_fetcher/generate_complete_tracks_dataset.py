# Script to generate a tracks dataset that contains not only the audio features, but also artist and track name
# Team: rcmmndrs
# Martin Plattner

import pandas as pd

# load original dataset
plays = pd.read_csv('../data/ir2017.csv')

# add play_count column to dataframe
plays_per_track = plays.groupby('track').size().to_frame('play_count')
plays_with_play_count = plays.set_index('track').join(plays_per_track).reset_index()

# unique tracks
tracks = plays_with_play_count.drop('user', axis=1).drop_duplicates(subset='track').set_index('track')



# load additional dataset with track metadata like track and artist names (fetched using the Spotify API)
tracks_metadata = pd.read_csv(
    '../spotify_api_fetcher/tracks_metadata.csv',
    names=['track', 'name', 'artist', 'album', 'album_type', 'album_track_number', 'explicit_content', 'popularity'],
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
tracks_with_metadata = tracks.join(tracks_metadata)
tracks_with_metadata.to_csv('../data/tracks_with_metadata.csv', encoding='utf-8')
