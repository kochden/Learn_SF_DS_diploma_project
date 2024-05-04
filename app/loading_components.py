import pickle
import os

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
