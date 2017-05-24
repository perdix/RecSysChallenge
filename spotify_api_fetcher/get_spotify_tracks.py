# Script to retrieve track metadata for every track in the original dataset using the Spotify API
# Team: rcmmndrs
# Martin Plattner

import pandas as pd
import requests
from pymongo import MongoClient

url = 'https://api.spotify.com/v1/tracks?ids='
headers = {
    'Accept': 'application/json',
    'Authorization': 'Bearer <API-KEY-GOES-HERE>'
}

# connect to mongodb to store fetched data
client = MongoClient()
db = client.spotify_data

s = requests.Session()

# import track data, drop duplicates, replace "spotify:track:", and return a list of all remaining ids
original_track_data = pd.read_csv('../ir2017.csv')
unique_series = pd.Series(original_track_data.track.unique())
track_ids = unique_series.str.replace('spotify:track:', '').tolist()

# generate a list containing lists of 50 track ids (50 tracks can be fetched at once using the Spotify API)
tracks_in_chunks_of_50 = [track_ids[i:i + 50] for i in range(0, len(track_ids), 50)]

# look the chunks list and fetch 50 tracks at once for every chunk
for chunk_number, chunk in enumerate(tracks_in_chunks_of_50):
    response = s.get(url + ','.join(chunk), headers=headers)
    # check if the request was successful
    if response.status_code == 200:
        json = response.json()

        # save every individual track
        for track in json['tracks']:
            db.track_data1.insert_one(track)

    # status message every 1000 tracks
    if chunk_number + 1 % 20 == 0:
        print('Fetched ' + str((chunk_number+1) * 50) + ' tracks.')
