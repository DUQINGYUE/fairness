import numpy as np
import pandas as pd
import ipdb
import data_utils
import scipy.sparse as sp
from random import choice
def make_dataset(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    train_ratings = pd.read_csv('./ml-100k/u1.base', sep='\t', names=r_cols,
			  encoding='latin-1')
    test_ratings = pd.read_csv('./ml-100k/u1.test', sep='\t', names=r_cols,
			  encoding='latin-1')
    if load_sidechannel:
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        users = pd.read_csv('./ml-100k/u.user', sep='|', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        bins = np.linspace(5, 75, num=15, endpoint=True)
        inds = np.digitize(users['age'].values, bins)
        m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        movies = pd.read_csv('./ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                             encoding='latin-1')
        movie_ratings = pd.merge(movies, train_ratings)
        df = pd.merge(movie_ratings, users)
        df.drop(df.columns[[3,4,7]], axis=1, inplace=True)
        movies.drop(movies.columns[[3,4]], inplace = True, axis = 1 )

    train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    train_ratings_matrix.fillna( 0, inplace = True )
    test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    test_ratings_matrix = test_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    test_ratings_matrix.fillna( 0, inplace = True )
    train_ratings_matrix.head()
    columnsTitles=["user_id","rating","movie_id"]
    train_ratings=train_ratings.reindex(columns=columnsTitles)-1
    test_ratings=test_ratings.reindex(columns=columnsTitles)-1
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1

    if load_sidechannel:
        return train_ratings,test_ratings,users,movies
    else:
        return train_ratings,test_ratings
def make_dataset_1M_new(load_sidechannel=False,train_np=0,test_np=0):
    #print("make dataset")
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=r_cols,
                          encoding='latin-1')
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings) * 0.9))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    if load_sidechannel:
        u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
        m_cols = ['movie_id', 'title', 'genre']
        users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        movies = pd.read_csv('./ml-1m/movies.dat', sep='::', names=m_cols,
                             encoding='latin-1', parse_dates=True)

    train_ratings.drop("unix_timestamp", inplace=True, axis=1)
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'], \
                                                     columns=['user_id'], values='rating').reset_index(drop=True)
    test_ratings.drop("unix_timestamp", inplace=True, axis=1)
    columnsTitles = ["user_id", "rating", "movie_id"]
    train_ratings = train_ratings.reindex(columns=columnsTitles) - 1
    test_ratings = test_ratings.reindex(columns=columnsTitles) - 1
    users.user_id = users.user_id.astype(np.int64)
    movies.movie_id = movies.movie_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1
    # print(type(train_ratings))
    train_data = train_ratings[['user_id', 'movie_id']]
    test_data = test_ratings[['user_id', 'movie_id']]
    ra = train_data.append(test_data)

    res = ra.sort_values(by='movie_id')
    res6 = res.groupby('movie_id').count()

    res6 = res6.sort_values(by='user_id')
    longi = res6.loc[res6["user_id"] > 1000].index
    short0 = res6.loc[res6["user_id"] <= 1000].index
    longlist = []
    for i in longi:
        longlist.append(i)
    shortlist = []
    for i in short0:
        shortlist.append(i)

    # 电影是从0开始的
    # print(train_data.sort_values(by='user_id'))
    user_num = train_data['user_id'].max() + 1
    item_num = train_data['movie_id'].max() + 1
    train_data = train_data.values.tolist()
    # print(user_num,item_num)6040 3952
    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    # print("train——mat")
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    print("采样开始")
    train_d = []
    for x in train_data:
        u, i = x[0], x[1]
        for t in range(0, train_np):
            if t >1:
                j = choice(longlist)
                while (u, j) in train_mat:
                    j = choice(longlist)
                train_d.append([u, i, j])
            else:
                j = choice(shortlist)
                while (u, j) in train_mat:
                    j = choice(shortlist)
                train_d.append([u, i, j])
    train_d = pd.DataFrame(data=train_d, columns=["u_id", "i", "j"])
    # train_d.to_csv("./data/train.csv",header=None, index=None)

    test_data = test_data.values.tolist()
    '''
    test_r = []
    for x in test_data:
        u, i = x[0], x[1]
        for t in range(4):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            test_r.append([u, i, j])
    test_r = pd.DataFrame(data=test_r, columns=["u_id", "i", "j"])
    '''

    # 测试数据集获取
    test_r = []
    test_d = []
    for x in test_data:
        u, i = x[0], x[1]
        arr = []
        arr.append(x[0])
        arr.append(x[1])
        for t in range(test_np):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            arr.append(j)
            test_r.append([u, i, j])
        test_d.append(arr)
    test_r = pd.DataFrame(data=test_r, columns=["u_id", "i", "j"])
    if load_sidechannel:
        return train_d, test_d, users, movies, user_num, item_num, train_mat,longlist,shortlist
    else:
        return train_ratings, test_ratings

def make_dataset_1M(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('BPR-pytorch-master/ml-1m/ratings.dat', sep='::', names=r_cols,
                          encoding='latin-1')
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings)*0.9))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    if load_sidechannel:
        u_cols = ['user_id','sex','age','occupation','zip_code']
        m_cols = ['movie_id','title','genre']
        users = pd.read_csv('BPR-pytorch-master/ml-1m/users.dat', sep='::', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        movies = pd.read_csv('BPR-pytorch-master/ml-1m/movies.dat', sep='::', names=m_cols,
                             encoding='latin-1', parse_dates=True)

    train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    columnsTitles=["user_id","rating","movie_id"]
    train_ratings=train_ratings.reindex(columns=columnsTitles)-1
    test_ratings=test_ratings.reindex(columns=columnsTitles)-1
    users.user_id = users.user_id.astype(np.int64)
    movies.movie_id = movies.movie_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1
    if load_sidechannel:
        return train_ratings,test_ratings,users,movies
    else:
        return train_ratings,test_ratings

# if __name__ == '__main__':
    make_dataset_1M(True)
