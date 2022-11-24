# try:
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import DistanceMetric
# except ModuleNotFoundError:
#     !pip install numpy
#     !pip install pandas
#     !pip install -U scikit-learn
# 
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from sklearn.metrics import DistanceMetric
from tqdm import tqdm


def df_from_file(file):
    '''Чтение JSON файла'''
    with open(file) as f:
        json_data = json.load(f)
    return pd.DataFrame(json_data['response']['postOfficeLocations'])


def common_data(path):
    '''Чтение из каталога всех файлов и создание общей таблицы'''
    data = {'latitude': [], 'longitude': [], 'region': [], 'fullAddress': [], 'addressGuid':[]}       
    for file in os.listdir(path):
        with open(f'{path}/{file}', encoding='utf-8', mode='r') as f:
            json_data = json.load(f)        
        for post in json_data['response']['postOffices']:
            data['latitude'].append(float(post['latitude']))
            data['longitude'].append(float(post['longitude']))
            data['region'].append(str(post['address']['region']))
            data['fullAddress'].append(str(post['address']['fullAddress']))
            data['addressGuid'].append(str(post['address']['addressGuid']))
    data = pd.DataFrame(data)
    idx = data.index
    data.insert(loc=0, column='id', value=idx)
    return data


def get_metre_dist(df):
    '''Матрица расстояний в метрах'''
    dist = DistanceMetric.get_metric('haversine')
    return dist.pairwise(
        np.radians(df[['latitude', 'longitude']]), 
        np.radians(df[['latitude', 'longitude']])
    ) * 6371000


def row_to_df(dist_matrix):
    '''Вычисление пути и расстояния для кадого кластера'''
    if dist_matrix.shape == (1, 1):
        return pd.DataFrame(
            {'n': dist_matrix.keys()[0], 'd': [0]},
            index=dist_matrix.index,
        )
    np.fill_diagonal(dist_matrix.values, np.nan)
    exist = []    # глобальный индекс
    distance = [] # расстояние
    indexes = list(dist_matrix.columns)
    index = indexes.pop(0)
    first_index = index
    while len(indexes):
        arg_min = dist_matrix.loc[index, indexes].idxmin()
        exist.append(arg_min)
        distance.append(round(dist_matrix.loc[index, arg_min]))
        index = indexes.pop(indexes.index(arg_min))       
    exist.append(first_index)
    distance.append(round(dist_matrix.loc[arg_min, first_index]))
    return pd.DataFrame(
            {'n': exist, 'd': distance},
            index=dist_matrix.index,
        )


def geo_data(df, k):
    '''Создание кластеров и формирование итоговой таблицы'''
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df[['latitude', 'longitude']])
    df['k'] = pd.Series(kmeans.labels_)
    frames = []
    for i in tqdm(range(k)):
        data = df[df['k'] == i]
        dist_matrix = pd.DataFrame(
            get_metre_dist(data),
            index=data.index,
            columns=data.index
        )
        frames.append(row_to_df(dist_matrix))
    return df.join(pd.concat(frames))


def save_to_json(data, name):
    '''Сохрание в виде списка скловарей'''
    data.to_json(name, orient="records")   
    

def main(k, path):
    data = common_data(path)
    result = geo_data(data, k)
    save_to_json(result, f"klustered_result/klustered.json")

if __name__ == '__main__':
    main(k=1000, path='pochta_offices_info_42209')
