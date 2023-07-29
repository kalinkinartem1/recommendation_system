"""
Файл с FastApi
"""

import os
from typing import List
from fastapi import FastAPI, HTTPException, Depends

from datetime import datetime
from loguru import logger
from schema import PostGet
from psycopg2.extras import RealDictCursor
import psycopg2

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import datetime as dt



# чтобы посмотреть доку
# http://127.0.0.1:8899/docs

# запуск сервера
# запускать нужно из директории где находится app
# uvicorn lesson_22_6_new_features:app --reload --port 8000
# uvicorn app:app --reload --port 8000

#
URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

app = FastAPI()

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models(model_name):
    if model_name=='catboost':
        model_path = get_model_path("/Users/artemkalinkin/Jupyter Notebook Project/start_ml_jupyter/ML/lesson_22/catboost_new_feature_6")

        from catboost import CatBoostClassifier
        # Загружаем модель
        print(f'Модель начала загружаться')
        catboost = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
        print(f'Модель загрузилась')
        return catboost.load_model(model_path)
    elif model_name=='log_reg':
        # менять имя модели или путь целиком
        model_path = get_model_path("/Users/artemkalinkin/Jupiter Notebook Project/start_ml_jupyter/ML/lesson_22/log_reg")
        # Загружаем модель
        print(f'Модель начала загружаться')
        loaded_model = pickle.load(open(model_path, 'rb'))
    # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
        print(f'Модель загрузилась')
    elif model_name=='knn':
        model_path = get_model_path(
            "/Users/artemkalinkin/Jupiter Notebook Project/start_ml_jupyter/ML/lesson_22/knn")
        # Загружаем модель
        loaded_model = pickle.load(open(model_path, 'rb'))
    elif model_name == 'xgboost':
        model_path = get_model_path(
            "/Users/artemkalinkin/Jupiter Notebook Project/start_ml_jupyter/ML/lesson_22/knn")
        loaded_model = pickle.load(open(model_path, 'rb'))

    return loaded_model



# Загрузка признаков по чанкам
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



def get_db():
    conn = psycopg2.connect(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
                            cursor_factory = RealDictCursor)
    return conn

# Загрузка признаков из базы данных
def load_table() -> pd.DataFrame:
    print(f'Таблица юзеров начала загружаться')
    user_data_df = batch_load_sql(f'SELECT * FROM "a_kalinkin_features_lesson_22_user_new_features_change" ')
    print(f'Таблица юзеров загрузилась')

    print(f'Таблица всех постов начала загружаться')
    post_text_df = batch_load_sql('SELECT * FROM "a_kalinkin_features_lesson_22_post_new_features_change"')
    print(f'Таблица всех постов загрузилась')

    return user_data_df, post_text_df


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int,time: datetime,limit: int = 5) -> List[PostGet]:

    hour= time.hour
    month = time.month
    day_of_week = time.weekday()

    # создам датафрэйм для того чтобы предсказать
    # сначала объеденю информацию по user_id с post_id
    # будет 7023 строки потому что столько строк в таблице post, причём user_id будет во всех одинаковый,
    # напишу пример с user_id = 200
    # user_id | post_id
    #  200     |   1
    #  200     |   2
    #  ...
    #  200     |  7319
    X = pd.DataFrame({'user_id': [id] * len(post_id_list), 'post_id': post_id_list})
    # далее объединю датафрэйм Х с таблицей user и post по user_id и post_id
    X = X. merge(post, on='post_id', how='left').merge(user, on='user_id', how='left')
    X = X.set_index(['user_id', 'post_id'])
    X['hour'] = hour
    X['month'] = month
    X['day_of_week'] = day_of_week
    X = X.drop(['text', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
    print(X.head())
    print(X.columns)


    # далее делаю предсказания и вывожу вероятности принадлежности 1 классу
    pred_prod = model.predict_proba(X)[:, 1]

    # Сортировка рекомендаций по вероятности и ограничение количества
    top_indices = pred_prod.argsort()[::-1][:limit]

    # теперь отбираю эти посты
    recommended_posts = []
    for index in top_indices:
        post_id = post.iloc[index]['post_id']
        text = str(post[post['post_id']==post_id]['text'].values[0])
        topic = str(post[post['post_id'] == post_id]['topic'].values[0])

        recommended_posts.append({
            'id': post_id,
            'text': text,
            'topic': topic
        })

    if recommended_posts:
        return recommended_posts
    else:
        raise HTTPException(404, "user not found")

#
#
# код из предыдущего шага
# # загрузка таблицы user_new_features в БД
# user_new_features = pd.read_csv('/Users/artemkalinkin/Jupyter Notebook Project/start_ml_jupyter/ML/lesson_22/user_new_features_change')
# table_1 = save_features(user_new_features, 'a_kalinkin_features_lesson_22_user_new_features_change')
#
# # загрузка таблицы post_new_features в БД
# post_new_features = pd.read_csv('/Users/artemkalinkin/Jupyter Notebook Project/start_ml_jupyter/ML/lesson_22/post_new_features_change')
# table_2 = save_features(post_new_features, 'a_kalinkin_features_lesson_22_post_new_features_change')

# # # проверка того что хранится в БД
# sql_1 = "SELECT * FROM a_kalinkin_features_lesson_22_user_new_features_change LIMIT 1000;"
# sql_2 = "SELECT * FROM a_kalinkin_features_lesson_22_post_new_features_change LIMIT 1000;"
# #
# engine = create_engine(URL)
#
# with engine.connect() as conn:
#     test_1 = pd.read_sql(sql_1, conn)
#     test_2 = pd.read_sql(sql_2, conn)


user, post = load_table()
# user = user.drop('Unnamed: 0',axis=1)
# post = post.drop('Unnamed: 0',axis=1)

post_id_list = post['post_id'].tolist()
model = load_models(model_name='catboost')

# print(post)
# print(user)