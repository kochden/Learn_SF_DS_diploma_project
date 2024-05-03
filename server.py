
# импорт необходимых библиотек
from flask import Flask, request, jsonify
import pickle
import os

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

import re
import difflib
import ast

#-----------------------------------------------------------

app = Flask(__name__)

#-----------------------------------------------------------

# функция по очистке входящих данных
def clearing_data(df):
    
    #бассейн
    # преобразовываем
    df['private_pool'] = df[['private pool', 'PrivatePool']].apply(lambda x: 1 if type(x[0]) == str or type(x[1]) == str else 0, axis=1)
    # удаляем
    df = df.drop(columns=['private pool', 'PrivatePool'])
    
    # MlsId
    # немного почистим
    def clearing_mls(feature):
        
        if type(feature) == str:
            feature_temp = ' '.join(feature.strip().split()).upper() # удалены лишние пробелы и переведен к верхнему регистру
            if feature_temp == 'NO': return np.nan
            if 'NO MLS' in feature_temp: return np.nan
            if feature_temp == 'NOMLSID': return np.nan
            return feature_temp
        return feature
    df['MlsId'] = df['MlsId'].apply(clearing_mls)
    df['mls-id'] = df['mls-id'].apply(clearing_mls)
    # дополняем один признак значениями другого
    df['MlsId'] = df['MlsId'].fillna(df['mls-id'])
    # удаляем лишний
    df = df.drop(columns=['mls-id'])
    # преобразуем в категориальный
    df['MlsId'] = df['MlsId'].apply(lambda x: 1 if type(x) == str else 0)

    # status
    def clearing_status(feature):
        # списки синонимов
        foreclosure = ['FORECLOSURE', 
                       'FORECLOSED', 
                       'CONTINGENT FORECLOSURE']
        new_construction = ['NEW CONSTRUCTION', 
                            'NEW']
        backup = ['UNDER CONTRACT W/ BCKP', 
                  'UNDER CONTRACT TAKING BACK UP OFFERS']
        pending = ['P', 
                   'PI', 
                   'PF', 
                   'PS']
        for_sale = ['PRICE CHANGE', 
                    'BACK ON MARKET', 
                    'LISTING EXTENDED']
        under_contract = ['C', 
                          'DUE DILIGENCE PERIOD', 
                          'C CONTINUE SHOW', 
                          'CT', 
                          'ACCEPTED OFFER', 
                          'UC CONTINUE TO SHOW']
        
        if type(feature) == str:
            # удаляем лишние пробелы и переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()

            # замены по списку
            if feature_temp in foreclosure: return 'FORECLOSURE'
            if feature_temp in new_construction: return 'NEW'
            if feature_temp in backup: return 'BACKUP'
            if feature_temp in pending: return 'PENDING'
            if feature_temp in for_sale: return 'FOR SALE'
            if feature_temp in under_contract: return 'UNDER CONTRACT'
            
            # замены по вхождению
            if 'COMING SOON' in feature_temp: return 'COMING SOON'
            if 'PRE-FORECLOSURE' in feature_temp: return 'PRE-FORECLOSURE'
            if 'PENDING' in feature_temp: return 'PENDING'
            if 'CONTRACT' in feature_temp: return 'UNDER CONTRACT'
            if 'AUCTION' in feature_temp: return 'AUCTION'
            if 'BACKUP' in feature_temp: return 'BACKUP'
            if 'OPTION' in feature_temp: return 'OPTION'
            if 'RENT' in feature_temp: return 'RENT'
            if 'LEASE' in feature_temp: return 'RENT'
            if 'CONTINGEN' in feature_temp: return 'CONTINGENT'
            if 'ACTIV' in feature_temp: return 'FOR SALE'
            
            return feature_temp
        else:
            return 'UNDEFINED'
    # применяем функцию очистки
    df['status'] = df['status'].apply(clearing_status)

    # propertyType
    def clearing_property_type(feature):
        # списки синонимов
        single_family = ['SINGLE FAMILY', 
                         'SINGLE-FAMILY', 
                         'SINGLE-FAMILY HOME', 
                         'SINGLE FAMILY HOME', 
                         'DETACHED', 
                         'TINY HOME', 
                         'SINGLE DETACHED', 
                         'SINGLE DETACHED, TRADITIONAL']
        multi_family = ['MULTI-FAMILY', 
                        'MULTI FAMILY', 
                        'MULTI-FAMILY HOME', 
                        'MULTI FAMILY HOME']
        mobile = ['MANUFACTURED HOME (HOUSE)', 
                  'MFD/MOBILE HOME', 
                  'MOBILE', 
                  'PREFAB', 
                  'MODULAR', 
                  'MO2 LE', 
                  'MO2LE', 
                  'MOBILE/MANUFACTURE', 
                  'MOBILE / MANUFACTURE', 
                  'MOBILE / MANUFACTURED', 
                  'MOBILE/MANUFACTURED']
        cabin = ['CABIN', 
                 'CA2 N', 
                 'CA2N']
        midcentury = ['MIDCENTURY', 
                      'MID CENTURY', 
                      'MID-CENTURY']
        townhouse = ['TOWNHOME', 
                     'TOWNHOUSE']
        coop = ['CO-OP', 
                'COOP']
        land = ['LAND', 
                'LOT/LAND']
        one_story = ['ONE STORY', 
                     '1 STORY', 
                     'DETACHED, ONE STORY']
        two_stories = ['2 STORIES', 
                       '2 STORY', 
                       'TWO STORY', 
                       'TWO STORIES', 
                       'DETACHED, TWO STORY']
        modern = ['CONTEMPORARY', 
                  'CONTEMPORARY/MODERN', 
                  'MODERN']

        if type(feature) == str:
            # удаляем лишние пробелы и переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()
            
            # замены по списку
            if feature_temp in single_family: return 'SINGLE FAMILY'
            if feature_temp in multi_family: return 'MULTI FAMILY'
            if feature_temp in mobile: return 'MOBILE'
            if feature_temp in cabin: return 'CABIN'
            if feature_temp in midcentury: return 'MIDCENTURY'
            if feature_temp in townhouse: return 'TOWNHOUSE'
            if feature_temp in coop: return 'COOP'
            if feature_temp in land: return 'LAND'
            if feature_temp in one_story: return '1 STORY'
            if feature_temp in two_stories: return '2 STORIES'
            if feature_temp == '': return 'UNDEFINED'
            if feature_temp in modern: return 'MODERN'
    
            # замены по вхождению
            if '1 STORY' in feature_temp: return '1 STORY'
            if '2 STORY' in feature_temp: return '2 STORIES'
            if '2 STORIES' in feature_temp: return '2 STORIES'
            if 'OTHER' in feature_temp: return 'OTHER'
            if 'RANCH' in feature_temp: return 'RANCH'
            if 'CONDO/TOWNHOME' in feature_temp: return 'CONDO/TOWNHOME'
            if 'COOP' in feature_temp: return 'COOP'
                    
            return feature_temp
        else:
            return 'UNDEFINED'

    df['propertyType'] = df['propertyType'].apply(clearing_property_type)

    list_top_property = df['propertyType'].value_counts()[:20].index
    df['property_type'] = df['propertyType'].apply(lambda x: x if x in list_top_property else 'OTHER')
    df['property_type'].value_counts()

    # state
    # сразу переведем в верхний регистр и потом посмотрим
    df['state'] = df['state'].apply(lambda x: x.upper() if type(x) == str else x)
    # преобразуем в множество
    set_state = set(cities['state_id'])
    # найдем те, что не найдены.
    missing_states = df[~df['state'].isin(set_state)]['state'].unique()
    # удалим не существующие
    df = df[~df['state'].isin(missing_states)]

    # city
    df = df[~df['city'].isnull()]
    # удаляем лишние пробелы и переводим в верхний регистр
    df['city'] = df['city'].apply(lambda x: ' '.join(x.strip().split()).upper())
    # удаляем строки с пустым значением
    df = df[df['city'] != '']

    # исправляем города
    # объявим словарь в котором будем хранить уже найденные значения
    found_city = {}
    def city_search(state_city):
        #global found_city
        if (state_city[0], state_city[1]) in found_city:
            # уже есть в словаре
            return found_city[(state_city[0], state_city[1])]
        else:
            # ищем список городов для штата
            ls_cities = list(cities[cities['state_id'] == state_city[0]]['city_ascii_upper'])
            # подбираем похожий город
            res = difflib.get_close_matches(state_city[1], ls_cities)
            if res == []:
                # не нашли ничего похожего
                return 'NOT FOUND'
            else:
                # нашли похожий город
                # пишем в словарь
                found_city[(state_city[0], state_city[1])] = res[0]
                return res[0]

    df['city_2'] = df[['state', 'city']].apply(city_search, axis=1)
    # Проведем небольшие замены значений
    df['city_2'] = df[['city', 'city_2']].apply(lambda x: 'OTHER' if 'OTHER' in x[0] else x[1], axis=1)
    df['city_2'] = df[['city', 'city_2']].apply(lambda x: x[0] if x[1] == 'NOT FOUND' else x[1], axis=1)

    # baths
    # проведем некоторые замены и преобразования текста.
    def clearing_baths(feature):
        # списки синонимов
        one_bath = ['1 / 1-0 / 1-0 / 1-0']
        two_baths = ['1-2 BATHS']
        three_baths = ['1-0 / 1-0 / 1', '3-1 / 2-2']
        four_baths = ['1 / 1 / 1 / 1', '2-1 / 2-1 / 1-1 / 1-1']
        nan_bath = ['116 / 116 / 116']

        if type(feature) == str:
            # удаляем лишние пробелы, переводим в верхний регистр, меняем запятые на точки
            feature_temp = ' '.join(feature.strip().split()).upper().replace(',', '.')
            
            # замены по списку
            if feature_temp in one_bath: return '1'
            if feature_temp in two_baths: return '2'
            if feature_temp in three_baths: return '3'
            if feature_temp in four_baths: return '4'
            if feature_temp in nan_bath: return np.nan
      
            return feature_temp
        else:
            return feature

    # переведем признак во float обрабатывая ошибки
    def feature_to_float(feature):
        if type(feature) == str:
            try:
                float_num=float(feature)
                return float_num
            except ValueError:
                return np.nan    
        else:
            return feature
        
    # чистим
    df['baths'] = df['baths'].apply(clearing_baths)

    # преобразовываем
    df['baths'] = df['baths'].replace(to_replace=r'[^\d.]+', value='', regex=True)
    df['baths'] = df['baths'].apply(feature_to_float)

    # beds
    # переведем в верхний регистр
    df['beds'] = df['beds'].apply(lambda x: x.upper() if type(x) == str else x)

    mask_1 = df['beds'].astype(str).str.contains('SQFT')
    mask_2 = df['beds'].astype(str).str.contains('ACRES')
    index_list = df[(mask_1 | mask_2)].index
    df.loc[index_list, 'sqft'] = df.loc[index_list, 'beds']
    df.loc[index_list, 'beds'] = np.nan

    # почистим
    def clearing_beds(feature):
        if type(feature) == str:
            # замены по вхождению
            if '3 OR MORE BEDROOMS' in feature: return '3'
            if '2 BEDROOMS' in feature: return '2'
            if "BASED ON REDFIN'S" in feature: return np.nan
                    
        return feature

    # переведем признак в int обрабатывая ошибки
    def feature_to_int(feature):
        if type(feature) == str:
            try:
                int_num=int(feature)
                return int_num
            except ValueError:
                return np.nan    
        else:
            return feature

    # чистим
    df['beds'] = df['beds'].apply(clearing_beds)

    # преобразовываем
    df['beds'] = df['beds'].replace(to_replace=r'[^\d.]+', value='', regex=True)
    df['beds'] = df['beds'].apply(feature_to_int)
    # пропуски заполним медианой
    df['beds'] = df['beds'].fillna(median_beds)

    # sqtf
    # почистим
    def clearing_sqft(feature):
        non_decimal = re.compile(r'[^\d.]+')
        
        if type(feature) == str:
            # переведем значение в верхний регистр
            feature = feature.upper()
            if 'ACR' in feature:
                # 1 акр равно 43 559.995 футов квадратных
                ratio = 43559.995
            else:
                ratio = 1
            try:
                feature = float(non_decimal.sub('', feature)) * ratio
                return feature
            except ValueError:
                return np.nan         
            
        return feature

    df['sqft'] = df['sqft'].apply(clearing_sqft)
    # пропуски заполним медианой
    df['sqft'] = df['sqft'].fillna(median_sqft)

    # снова baths
    # заполним пропущенные значения
    mask_1 = df['baths'].isnull()
    index_list = df[mask_1].index
    df.loc[index_list, 'baths'] = df.loc[index_list, 'beds'] + df.loc[index_list, 'private_pool']

    # исправим значения со значительным превышением
    mask_1 = (df['baths'] // (df['beds'] + df['private_pool'])) > 2
    mask_2 = df['baths'] > 2
    index_list = df[mask_1 & mask_2].index
    df.loc[index_list, 'baths'] = df.loc[index_list, 'beds'] + df.loc[index_list, 'private_pool']

    # zipcode
    df = df.drop(columns=['zipcode'])

    # fireplace
    def clearing_fireplace(feature):
        # списки синонимов
        fireplace_yes = ['YES', 'ONE', 'ELECTRIC', 'DECORATIVE', 'IN LIVING ROOM', 'SPECIAL FEATURES', 
                        'IN GREAT ROOM', 'LIVING ROOM', 'LOCATION', 'FAMILYRM', 'GREAT ROOM', 'FAMILY ROOM', 'BRICK']
        fireplace_no = ['NOT APPLICABLE', 'NO', 'WALK-IN CLOSETS', 'N/K', 'DEN']
        fireplace_q = ['TWO']

        if type(feature) == str:
            # удаляем лишние пробелы, переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()
            
            # замены по списку
            if feature_temp in fireplace_yes: return '1'
            if feature_temp in fireplace_no: return '0'
            if feature_temp in fireplace_q: return '2'
    
            # замены по вхождению
            if 'FIREPLACE' in feature_temp: return '1'
            if 'WOOD' in feature_temp: return '1'
            if 'GAS' in feature_temp: return '1'
            if 'ELECTRIC' in feature_temp: return '1'
            if 'DECORATIVE' in feature_temp: return '1'
            if 'FIRE PIT' in feature_temp: return '1'
            if 'FIREPIT' in feature_temp: return '1'
            if 'CEILING FAN' in feature_temp: return '0'
                    
            return feature_temp
        else:
            return '0'

    # чистим
    df['fireplace'] = df['fireplace'].apply(clearing_fireplace)
    # преобразовываем
    df['fireplace'] = df['fireplace'].replace(to_replace=r'[^\d.]+', value='', regex=True)
    df['fireplace'] = df['fireplace'].apply(feature_to_int)
    df['fireplace'] = df['fireplace'].fillna(0)
    df['fireplace'] = df['fireplace'].apply(lambda x: 1 if x > 0 else 0)

    # stories
    def clearing_stories(feature):
        # списки синонимов
        story_1 = ['GROUND LEVEL', 'RANCH']
        story_2 = ['1-2 STORIES', '1 LEVEL, 1.5 LEVEL', '1 LEVEL, 2 LEVEL', 
                   '2 STORY OR 2 LEVEL', 'TOWNHOUSE-2 STORY, T3-TOWNHOUSE 3+ STORIES', 
                   '1 1/2 STORY', '1 1/2 LEVELS']
        story_3 = ['ONE, THREE OR MORE', 'ONE AND ONE HALF, THREE OR MORE', 
                   'TWO, THREE OR MORE', 'T3-TOWNHOUSE 3+ STORIES', '3-4 STORIES', 
                   '3 - 5 STORIES', '1-2 STORIES, 3-4 STORIES']

        if type(feature) == str:
            # удаляем лишние пробелы, переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()
            
            # замены по списку
            if feature_temp in story_1: return '1'
            if feature_temp in story_2: return '2'
            if feature_temp in story_3: return '3'
    
            # замены по вхождению
            if 'ONE' in feature_temp: return '1'
            if 'TWO' in feature_temp: return '2'
            if 'THREE' in feature_temp: return '3'
            if 'TRI-LEVEL' in feature_temp: return '3'
            if 'QUAD LEVEL' in feature_temp: return '4'
            if 'BI-LEVEL' in feature_temp: return '2'
            if 'FOURPLEX' in feature_temp: return '4'
                    
            return feature_temp
        else:
            return feature

    # чистим
    df['stories_2'] = df['stories'].apply(clearing_stories)
    # преобразовываем
    df['stories_2'] = df['stories_2'].replace(to_replace=r'[^\d.]+', value='', regex=True)
    df['stories_2'] = df['stories_2'].apply(feature_to_float).round()
    
    mask_1 = df['stories_2'] > 3
    mask_2 = df['stories_2'].isnull()
    mask_3 = df['property_type'] == 'LAND'
    
    index_list = df[(mask_1 | mask_2) & mask_3].index
    df.loc[index_list, 'stories_2'] = 0

    mask_1 = df['stories_2'] > 3
    mask_2 = df['stories_2'].isnull()
    mask_3 = df['stories_2'] == 0
    mask_4 = df['property_type'] != 'LAND'
    mask_5 = df['sqft'] <= 2000

    index_list = df[(mask_1 | mask_2 | mask_3) & mask_4 & mask_5].index
    df.loc[index_list, 'stories_2'] = 1

    index_list = df[(mask_1 | mask_2 | mask_3) & mask_4 & ~mask_5].index
    df.loc[index_list, 'stories_2'] = 2

    # street
    df = df.drop(columns=['street'])

    # homeFacts
    # разберем признак и выделим из него новые данные
    def home_facts_analysis(feature):
        try:
            # получим список словарей
            ls_dict = ast.literal_eval(feature)['atAGlanceFacts']
            dict_result = {}
            for home_facts in ls_dict:
                # сразу обработаем пустые значения
                if home_facts['factValue'] == '' or home_facts['factValue'] == None:
                    fact_value = np.nan
                else:
                    fact_value = home_facts['factValue']
                
                dict_result[home_facts['factLabel']] = fact_value
            
            result = pd.Series([
                dict_result['Year built'],
                dict_result['Remodeled year'],
                dict_result['Heating'],
                dict_result['Cooling'],
                dict_result['Parking'],
                dict_result['lotsize'],
                dict_result['Price/sqft']
            ])
            return result
        except (KeyError, SyntaxError, ValueError):
            #print('Ошибка обработки homeFacts:', feature)
            result = pd.Series([
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ])
            return result
    
    df[['Year built',
        'Remodeled year',
        'Heating',
        'Cooling',
        'Parking',
        'lotsize',
        'Price/sqft']] = df['homeFacts'].apply(home_facts_analysis)

    # Year built
    # Переведем признак в числовой вид.
    df['Year built'] = df['Year built'].apply(feature_to_int)

    # удалим заведомо ложные значения
    mask_1 = df['Year built'] < 1000
    mask_2 = df['Year built'] > 2023
    index_list = df[mask_1 | mask_2].index
    df.loc[index_list, 'Year built'] = np.nan

    #пропуски заполним медианой
    df['Year built'] = df['Year built'].fillna(median_year_built)
    # переведем в возраст
    current_year = 2024
    df['age_of_the_object'] = current_year - df['Year built']

    # Remodeled year
    # Переведем признак в числовой вид.
    df['Remodeled year'] = df['Remodeled year'].apply(feature_to_int)

    # удалим заведомо ложные значения
    mask_1 = df['Remodeled year'] < 1000
    mask_2 = df['Remodeled year'] > 2023
    index_list = df[mask_1 | mask_2].index
    df.loc[index_list, 'Remodeled year'] = np.nan

    # Переведем признак в числовой вид.
    df['Remodeled year'] = df['Remodeled year'].apply(feature_to_int)

    # удалим заведомо ложные значения
    mask_1 = df['Remodeled year'] < 1000
    mask_2 = df['Remodeled year'] > 2023
    index_list = df[mask_1 | mask_2].index
    df.loc[index_list, 'Remodeled year'] = np.nan

    mask_1 = df['Remodeled year'].isnull()
    mask_2 = df['Remodeled year'] < df['Year built']
    index_list = df[mask_1 | mask_2].index
    df.loc[index_list, 'Remodeled year'] = df.loc[index_list, 'Year built']
    df['age_of_the_remodeled'] = current_year - df['Remodeled year']

    df = df.drop(columns=['age_of_the_object'])

    # Heating
    def clearing_heating(feature):
        # списки синонимов
        air = ['HEAT PUMP']

        if type(feature) == str:
            # удаляем лишние пробелы, переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()
            
            # замены по списку
            if feature_temp in air: return 'AIR'

            # замены по вхождению
            if 'AIR' in feature_temp: return 'AIR'
            if 'GAS' in feature_temp: return 'GAS'
            if 'ELECTRIC' in feature_temp: return 'ELECTRIC'
            if 'OTHER' in feature_temp: return 'OTHER'
            if 'HEAT PUMP' in feature_temp: return 'AIR'
            if 'REFRIGERATION' in feature_temp: return 'AIR'
            if 'CENTRAL' in feature_temp: return 'CENTRAL'
            if 'BASEBOARD' in feature_temp: return 'BASEBOARD'
            if 'WALL' in feature_temp: return 'WALL'
                    
            return feature_temp
        else:
            return 'NO DATA'

    # чистим
    df['Heating'] = df['Heating'].apply(clearing_heating)

    list_top = df['Heating'].value_counts()[:10].index
    df['Heating'] = df['Heating'].apply(lambda x: x if x in list_top else 'OTHER')
    df['Heating'].value_counts()

    # Cooling
    def clearing_feature(feature):
        if type(feature) == str:
            # удаляем лишние пробелы, переводим в верхний регистр
            feature_temp = ' '.join(feature.strip().split()).upper()
            if 'NO DATA' == feature_temp or 'NONE' == feature_temp or '0' == feature_temp:
                return 0
            return 1
        else:
            return 0

    # чистим
    df['Cooling'] = df['Cooling'].apply(clearing_feature)

    # Parking
    # воспользуемся той же функцией, т.к. алгоритм не отличается
    df['Parking'] = df['Parking'].apply(clearing_feature)

    # lotsize
    df['lotsize'] = df['lotsize'].apply(clearing_sqft)
    # заполним пропуски
    df['lotsize'] = df['lotsize'].fillna(0)

    # Price/sqft
    df = df.drop(columns=['Price/sqft'])

    # schools
    # сделаем функцию для разбора
    def grades_analysis(grades):
        
        def list_grades_to_int(list_grades):
            # заменим текстовые обозначения цифровыми: Preschool (pre-kindergarten) - -1; Kindergarten - 0.
            # и переведем все в числовой формат
            ln_list_grades = len(list_grades)
            for ls in range(ln_list_grades):
                ln_ls = len(list_grades[ls])
                for gr in range(ln_ls):
                    if list_grades[ls][gr] == 'K':
                        list_grades[ls][gr] = 0
                    elif list_grades[ls][gr] == 'PK':
                        list_grades[ls][gr] = -1
                    else:
                        try:
                            list_grades[ls][gr] = int(list_grades[ls][gr])
                        except ValueError:
                            list_grades[ls][gr] = -2
            return list_grades

        def unpack_list_grades(list_grades):
            # раскроем диапазоны
            ln_list_grades = len(list_grades)
            for ls in range(ln_list_grades):
                ln_ls = len(list_grades[ls])
                if ln_ls == 2:
                    diff_ls = list_grades[ls][1] - list_grades[ls][0] + 1
                    ls_grades_new = []
                    for i in range(diff_ls):
                        ls_grades_new.append(list_grades[ls][0] + i)
                    list_grades[ls] = ls_grades_new
            return list_grades

        def flatten(lst):
            # Функция для выпрямления списка
            # Создаём новый пустой список
            result = []
            # Создаём цикл по элементам списка
            for elem in lst:
                # Если элемент списка является списком,
                if type(elem) is list:
                    # Применяем к нему функцию выпрямления и добавляем элементы к результату
                    result += flatten(elem)
                else: # Если элемент не является списоком,
                    # Добавляем элемент в новый список
                    result.append(elem)
            return result

        # удаляем лишние пробелы, переводим в верхний регистр
        if grades == None: grades=''
        grades = ' '.join(grades.strip().split()).upper()
        # Немного преобразований для стандартизации
        grades = grades.replace('PRESCHOOL', 'PK').replace('TO', '-').replace('–', '-').replace(' ', '')
        grades = grades.split(',')
        grades = list(map(lambda x: x.split('-'), grades))
        # преводим в цифровой формат
        grades = list_grades_to_int(grades)
        # раскрываем диапазоны
        grades = unpack_list_grades(grades)
        flatten_grades = flatten(grades)
        if -2 in flatten_grades:
            flatten_grades.remove(-2)
        result = len(flatten_grades) * (1/15)
        
        return grades, flatten_grades, result

    # разберем признак
    def schools_analysis_2(feature):
        try:
            # получим словарь
            school_data = ast.literal_eval(feature)
            
            # функция для перевода в int значений рейтинга школ
            def rating_to_int(rating):
                try:
                    return int(rating.split('/')[0])
                except ValueError:
                    return 1
            # функция для перевода во float значений расстояния
            def distance_to_float(distance):
                non_decimal = re.compile(r'[^\d.]+')
                return float(non_decimal.sub('', distance))

            for data in school_data:
                # разбираем rating
                ls_rating = list(map(rating_to_int, data['rating'])) # получили список
                if len(ls_rating) > 0:
                    av_rating = round(sum(ls_rating) / len(ls_rating), 3) # посчитали среднее
                else:
                    av_rating = np.nan
                
                # разбираем distance
                ls_distance = list(map(distance_to_float, data['data']['Distance'])) # получили список
                if len(ls_distance) > 0:
                    av_distance = round(sum(ls_distance) / len(ls_distance), 3) # посчитали среднее
                else:
                    av_distance = np.nan
                
                # разбираем grades
                ls_shool_grades = []
                for shool_grades in  data['data']['Grades']:
                    _, _, grades = grades_analysis(shool_grades)
                    ls_shool_grades.append(grades) # получили список
                if len(ls_shool_grades) > 0:
                    av_grades = round(sum(ls_shool_grades) / len(ls_shool_grades), 3) # посчитали среднее
                else:
                    av_grades = np.nan
                
                
            return pd.Series([av_rating, av_distance, av_grades])
        
        except (KeyError, SyntaxError, ValueError):
            #print('Ошибка обработки:', feature)
            return pd.Series([np.nan, np.nan, np.nan])
    
    df[['schools_av_rating', 'schools_av_distance', 'schools_av_grades']] = df['schools'].apply(schools_analysis_2)

    # заполним пропуски медианой
    df['schools_av_rating'] = df['schools_av_rating'].fillna(median_schools_av_rating)
    df['schools_av_distance'] = df['schools_av_distance'].fillna(median_schools_av_distance)
    df['schools_av_grades'] = df['schools_av_grades'].fillna(median_schools_av_grades)

    # удаляем лишние колонки
    df = df.drop(columns=['propertyType', 'city', 'stories', 'homeFacts', 'Year built', 'Remodeled year', 'schools'])

    return df

#-----------------------------------------------------------

def encode_features(df):
    # Binary Encoding
    columns_to_change = ['property_type', 'state', 'city_2']
    type_bin = bin_encoder.transform(df[columns_to_change])
    df = pd.concat([df, type_bin], axis=1)
    # удалим лишнее
    df = df.drop(columns=columns_to_change)
    
    # OneHot Encoding
    columns_to_change = ['status', 'Heating']
    type_hot = hot_encoder.transform(df[columns_to_change])
    df = pd.concat([df, type_hot], axis=1)
    # удалим лишнее
    df = df.drop(columns=columns_to_change)
    return df

#-----------------------------------------------------------    

@app.route('/')
def index():
    return 'Test message. The server is running'

#-----------------------------------------------------------

@app.route('/predict', methods=['POST'])
def pred_fun():
    df_json = request.json
    df = pd.read_json(df_json)
    df = clearing_data(df) # очистка входящих данных
    df = encode_features(df) # кодируем категориальные признаки
    predict = list(model.predict(df).round(2))
    print(predict)
    return jsonify({'prediction': predict})

#-----------------------------------------------------------

if __name__ == '__main__':
    
    # Производим десериализацию и извлекаем необходимые компоненты из pkl
    # модель
    file_name = 'model_cbr.pickle'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', file_name)
    with open(path_to_file, 'rb') as pkl_file:
        model = pickle.load(pkl_file)
    
    # медианы
    file_name = 'median.pickle'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', file_name)
    with open(path_to_file, 'rb') as pkl_file:
        median = pickle.load(pkl_file)
    median_beds = median.loc[0, 'median_beds']
    median_sqft = median.loc[0, 'median_sqft']
    median_year_built = median.loc[0, 'median_year_built']
    median_schools_av_rating = median.loc[0, 'median_schools_av_rating']
    median_schools_av_distance = median.loc[0, 'median_schools_av_distance']
    median_schools_av_grades = median.loc[0, 'median_schools_av_grades']
    
    # кодировщики
    # bin_encoder
    file_name = 'bin_encoder.pickle'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', file_name)
    with open(path_to_file, 'rb') as pkl_file:
        bin_encoder = pickle.load(pkl_file)
    # hot_encoder
    file_name = 'hot_encoder.pickle'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', file_name)
    with open(path_to_file, 'rb') as pkl_file:
        hot_encoder = pickle.load(pkl_file)
    
    # эталонное наименование города
    file_name = 'cities.pickle'
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', file_name)
    with open(path_to_file, 'rb') as pkl_file:
        cities = pickle.load(pkl_file)
    
    app.run('localhost', 5000)