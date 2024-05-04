# Задаём базовый образ
FROM tiangolo/uwsgi-nginx-flask:python3.11
# Копируем содержимое папки ./app в рабочую директорию контейнера
COPY ./app ./
# Копируем файл requirements.txt в рабочую директорию контейнера
COPY ./requirements.txt ./
# Копируем файл uwsgi.ini в рабочую директорию контейнера
COPY ./uwsgi.ini ./
# Запускаем установку необходимых зависимостей
RUN pip install --no-cache-dir -r requirements.txt