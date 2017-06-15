# Recommender system module
# Implementation of an recommender based on tracks by the same album
# Team: rcmmndrs
# Paul Opitz, Martin Plattner


from collections import Counter
import requests
import base64
import time
import sys


def recommend(tracks, n):

    s = requests.Session()

    # Get spotify token with client id und client secret
    encoded = base64.b64encode(b'8f61bd6e36ef4b4e9607a0d20288be1c:3fefca6753a440daaf5395da6554af7e').decode("utf-8") 
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Basic ' + encoded
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = s.post('https://accounts.spotify.com/api/token', headers=headers, data=data)
    access_token = response.json()['access_token']
    #print(access_token)

    # Wait to get sure response is there (not nice but quick solution)
    time.sleep(1)

    # Fetch Spotify API
    url_track = 'https://api.spotify.com/v1/tracks/?ids='
    url_album = 'https://api.spotify.com/v1/albums/?ids='
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + access_token
    }
    


    tracks = [track.replace('spotify:track:', '') for track in tracks]

    #tracks.str.replace('spotify:track:', '').tolist()

    tracks_in_chunks_of_50 = [tracks[i:i + 50] for i in range(0, len(tracks), 50)]

    albums = []
    result_counter = Counter()

    # look the chunks list and fetch 50 tracks at once for every chunk
    for chunk_number, chunk in enumerate(tracks_in_chunks_of_50):
        #print( url_track + ','.join(chunk) )
        response = s.get(url_track + ','.join(chunk), headers=headers)
        #if chunk_number == 1:
        #    print(url_track + ','.join(chunk))
        #    print(headers)
        # check if the request was successful
        if response.status_code == 200:
            json = response.json()
            for track in json['tracks']:
                albums.append(track['album']['id'])
        else:
            sys.stdout.write(response.json())
            sys.stdout.flush()



    albums_in_chunks_of_20 = [albums[i:i + 20] for i in range(0, len(albums), 20)]
    for chunk_number, chunk in enumerate(albums_in_chunks_of_20):
        response = s.get(url_album + ','.join(chunk), headers=headers)
        # check if the request was successful
        if response.status_code == 200:
            json = response.json()
            for album in json['albums']:
                album_tracks = [ 'spotify:track:'+i['id'] for i in album['tracks']['items'] ]
                result_counter.update(album_tracks)
        else:
            sys.stdout.write(response.json())
            sys.stdout.flush()

    time.sleep(5)

    result = result_counter.most_common(n)
    return result 


