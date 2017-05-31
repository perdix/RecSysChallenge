# Script to retrieve track metadata for every track in the original dataset using the Spotify API
# Team: rcmmndrs
# Martin Plattner

import pandas as pd
import requests
from pymongo import MongoClient

url = 'https://api.spotify.com/v1/artists?ids='
headers = {
    'Accept': 'application/json',
    #'Authorization': 'Bearer <API-KEY-GOES-HERE>'
}

# connect to mongodb to store fetched data
client = MongoClient()
db = client.spotify_data

tracks = db.track_data.find({},{'artists': 1})
artist_ids = set()
for track in tracks:
    artist_ids.add(track['artists'][0]['id'])

artist_ids = list(artist_ids)


s = requests.Session()

# generate a list containing lists of 50 track ids (50 tracks can be fetched at once using the Spotify API)
albums_in_chunks_of_50 = [artist_ids[i:i + 50] for i in range(0, len(artist_ids), 50)]

# look the chunks list and fetch 50 albums at once for every chunk
for chunk_number, chunk in enumerate(albums_in_chunks_of_50):
    response = s.get(url + ','.join(chunk), headers=headers)
    # check if the request was successful
    if response.status_code == 200:
        json = response.json()

        # save every individual track
        for artist in json['artists']:
            db.artist_data.insert_one(artist)

    # status message every 1000 albums
    if chunk_number + 1 % 20 == 0:
        print('Fetched ' + str((chunk_number+1) * 50) + ' artists.')
