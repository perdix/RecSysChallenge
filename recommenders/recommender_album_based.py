# Album-based recommender class
# Implementation of an recommender based on tracks by the same album
# Team: rcmmndrs
# Paul Opitz, Martin Plattner


from collections import Counter

import requests
import base64
import logging
import time
import random

from simplejson import JSONDecodeError

from recommenders.recommender_base import RecommenderBase


class RecommenderAlbumBased(RecommenderBase):

    def __init__(self):
        super().__init__()
        self.name = 'Album-bas.'
        self.sleep_between_requests = 1
        self.spotify_access_token_retries = 0
        self.spotify_access_token = ''
        self.spotify_track_details_url = 'https://api.spotify.com/v1/tracks/?ids='
        self.spotify_album_details_url = 'https://api.spotify.com/v1/albums/?ids='


    def setup(self, input_dataframe):
        self.request_session = requests.Session()
        self.get_spotify_token()
        self.is_setup = True


    def get_spotify_token(self):
        logging.info(self.name + ': Calling get_spotify_token() to renew token.')

        # Get spotify token
        encoded = base64.b64encode(b'8f61bd6e36ef4b4e9607a0d20288be1c:3fefca6753a440daaf5395da6554af7e').decode("utf-8")
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Basic ' + encoded
        }
        data = {
            'grant_type': 'client_credentials'
        }
        response = self.request_session.post('https://accounts.spotify.com/api/token', headers=headers, data=data)
        time.sleep(self.sleep_between_requests)

        if response.status_code == 200:
            try:
                access_token = response.json()['access_token']
                self.spotify_access_token = access_token
            except JSONDecodeError as e:
                logging.warning(self.name + ': Got JSONDecodeError exception. Will try again soon.')
        else:
            self.get_spotify_token()


    def request_spotify_api(self, url):
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + self.spotify_access_token
        }

        request_tries = 0
        while (request_tries <= 2):
            response = self.request_session.get(url, headers=headers)
            request_tries += 1

            time.sleep(self.sleep_between_requests)

            # check if the request was successful
            if response.status_code == 200:
                try:
                    json = response.json()
                    return json
                except JSONDecodeError as e:
                    logging.warning(self.name + ': Got JSONDecodeError exception. Will possibly try again.')

            # request was not successful, retry once
            if request_tries < 3:
                logging.warning(self.name + ': Request failed. Trying again with fresh spotify access token.')
                self.get_spotify_token()
            elif request_tries == 3:
                logging.warning(self.name + ': Request failed 3 times. No more retries are performed.')

        # request failed
        return None

    def recommend_by_tracks(self, tracks, n):
        if len(tracks) > 100:
            # get a random sample of 100 tracks
            tracks_set = set()
            while len(tracks_set) < 100:
                tracks_set.add(random.choice(tracks))

            # convert set to list
            tracks = list(tracks_set)

        # Fetch Spotify API
        tracks_ids = [track.replace('spotify:track:', '') for track in tracks]

        tracks_in_chunks_of_50 = [tracks_ids[i:i + 50] for i in range(0, len(tracks_ids), 50)]

        album_ids = []
        result_counter = Counter()

        # look the chunks list and fetch 50 tracks at once for every chunk
        for chunk_number, chunk in enumerate(tracks_in_chunks_of_50):
            response = self.request_spotify_api(self.spotify_track_details_url + ','.join(chunk))
            if response is not None:
                for track in response['tracks']:
                    album_ids.append(track['album']['id'])


        # get related tracks
        albums_in_chunks_of_20 = [album_ids[i:i + 20] for i in range(0, len(album_ids), 20)]
        for chunk_number, chunk in enumerate(albums_in_chunks_of_20):
            response = self.request_spotify_api(self.spotify_album_details_url + ','.join(chunk))
            if response is not None:
                for album in response['albums']:
                    album_tracks = ['spotify:track:' + item['id'] for item in album['tracks']['items']]
                    result_counter.update(album_tracks)

        result = result_counter.most_common()
        filtered = list(filter(lambda item: item[0] not in tracks_ids, result))
        return filtered[:n]
