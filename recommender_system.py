# Recommender system class
# This class controls all recommender modules and processes given tracks according to the defined pipeline.
# Team: rcmmndrs


from collections import Counter
import pandas as pd
import logging


class RecommenderSystem:
    def __init__(self, input_file, pipeline):
        self.input_file = input_file
        self.pipeline = pipeline
        self.name = 'RECSys'

        self.plays = pd.read_csv(self.input_file)


    def recommend_by_tracks(self, tracks, n=10):
        result_counter = Counter()

        for recommender in self.pipeline:
            if not recommender.is_setup:
                logging.info(recommender.name + ': Calling setup().')
                recommender.setup(self.plays)

            logging.info(recommender.name + ': Calling recommend_by_tracks() with ' + str(len(tracks)) + ' tracks.')
            recommender_result = recommender.recommend_by_tracks(tracks, 2 * n)
            logging.info(recommender.name + ': Found ' + str(len(recommender_result)) + ' tracks.')

            for track, count in recommender_result:
                result_counter.update({track: count})

        result = [i[0] for i in result_counter.most_common(n)]
        # logging.info(self.name  + ': Rec and MostCommon are the same: ' + str(sorted([item[0] for item in recommender_result]) == sorted(result)))
        return result
