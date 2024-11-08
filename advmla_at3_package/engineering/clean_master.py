import pandas as pd
import numpy as np
import json


def preprocessing_data(df_train, df_test):

    avg_distances = (
    df_train.dropna(subset=['totalTravelDistance'])
    .groupby(['startingAirport', 'destinationAirport'])['totalTravelDistance']
    .mean()
    .round(3)
    .astype(float)
    .reset_index()
    .rename(columns={'totalTravelDistance': 'averageDistance'})
    )

    df_train = pd.merge(df_train, avg_distances, on=['startingAirport', 'destinationAirport'], how='left')
    df_train['totalTravelDistance'].fillna(df_train['averageDistance'], inplace=True)
    df_train.drop(columns='averageDistance', inplace=True)

    df_test = pd.merge(df_test, avg_distances, on=['startingAirport', 'destinationAirport'], how='left')
    df_test['totalTravelDistance'].fillna(df_test['averageDistance'], inplace=True)
    df_test.drop(columns='averageDistance', inplace=True)

    avg_distances_dict = avg_distances.to_dict(orient='records')

    with open('../data/average_distances.json', 'w') as json_file:
        json.dump(avg_distances_dict, json_file, indent=4)

    return df_train, df_test