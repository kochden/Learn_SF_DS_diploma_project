import requests
import pandas as pd
import numpy as np
import os

# отправка запроса и получение результата
def sending_request(end_point, query_string):
    # выполняем POST-запрос на сервер по эндпоинту predict с параметром json
    r = requests.post(end_point, json=query_string)
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        # возвращаем результат
        return r.status_code, r.json()['prediction']
    else:
        # если запрос завершён с кодом, отличным от 200,
        # возвращаем содержимое ответа
        return r.status_code, r.text

#-----------------------------------------------------------

if __name__ == '__main__':
    
    # ендпоинт
    end_point = 'http://localhost:5000/predict'
    
    # отберем для теста два небольших фрагмента датасета
    file_name = 'data.zip'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', file_name)
    df = pd.read_csv(path_to_file)
    df = df.drop(columns=['target'])
    # в количестве 1 строка
    df_1 = df.sample(n=1, random_state=42)
    # и в количестве 5 строк
    df_5 = df.sample(n=5, random_state=42)
    # соберем их в список
    ls_df = [df_1, df_5]
    # сделаем запросы к серверу на предсказание
    for df in ls_df:
        query_string = df.to_json()
        result = sending_request(end_point, query_string)
        if result[0] == 200:
            print(result[1])
        else:
            print(f'Ошибка. Код ошибки: {result[0]}')
