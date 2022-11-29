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
from geopy.distance import geodesic, great_circle


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

def get_metre_dist(df, approx):
    '''Матрица расстояний в метрах'''
    if approx:
        dist = DistanceMetric.get_metric('haversine')
        return dist.pairwise(
            np.radians(df[['latitude', 'longitude']]),
            np.radians(df[['latitude', 'longitude']])
        ) * 6371000
    df = df.copy()
    df['coords'] = df[['latitude', 'longitude']].apply(tuple, axis=1)
    square = pd.DataFrame(
        np.zeros(len(df) ** 2).reshape(len(df), len(df)),
        index=df.index, columns=df.index
    )
    distances = square.apply(lambda x: df['coords'].apply(geodesic, args=(df.loc[x.name, 'coords'],), ellipsoid='WGS-84'), axis=1).T
    return distances.applymap(lambda x: x.meters)


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
    index = indexes.pop(indexes.index(dist_matrix.iloc[0].idxmin()))
    while len(indexes):
        arg_min = dist_matrix.loc[index, indexes].idxmin()
        distance.append(round(dist_matrix.loc[index, arg_min]))
        exist.append(index)
        index = indexes.pop(indexes.index(arg_min))       
    distance.append(round(dist_matrix.loc[arg_min, exist[0]]))
    exist.append(index)
    return pd.DataFrame(
            {'n': exist[1:] + [exist[0]], 'd': distance},
            index=exist, 
        )


def geo_data(df, k, min_clusters, approx):
    '''Создание кластеров и формирование итоговой таблицы'''
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df[['latitude', 'longitude']])
    df['k'] = pd.Series(kmeans.labels_)
    cluster_count = {}
    frames = []
    for i in tqdm(range(k)):
        data = df[df['k'] == i]
        if len(data) >= min_clusters:
            cluster_count[i] = len(data)
            dist_matrix = pd.DataFrame(
                get_metre_dist(data, approx),
                index=data.index,
                columns=data.index
            )
            distance = row_to_df(dist_matrix)
            distance['count'] = len(data)
            frames.append(distance)
    result = df.join(pd.concat(frames))
    result = result.dropna()
    result['n'] = result['n'].astype(int)
    result['d'] = result['d'].astype(int)
    # print(cluster_count) # сумма элементов внутри каждого кластера
    return result 


def save_to_json(data, name):
    '''Сохрание в виде списка скловарей'''
    data.pivot_table(index='k', values='count', aggfunc='first').sort_index().to_excel(name.replace('json', 'xlsx'))
    with open(name, 'w', encoding='utf-8') as file:
        data.to_json(file, orient="records", force_ascii=False)


def main(k, min_clusters, path, approx=False):
    data = common_data(path)
    result = geo_data(data, k, min_clusters, approx)
    eq_type = 'approx' if approx else 'geopy'
    save_to_json(result, f"clustered_result/cluster_{k}_filter_{min_clusters}_{eq_type}.json")

if __name__ == '__main__':
    main(k=500, min_clusters=1, path='pochta_offices_info_42209', approx=True)

