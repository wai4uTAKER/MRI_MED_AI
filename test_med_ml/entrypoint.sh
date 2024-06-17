#!/bin/bash
set -e

echo "Connection to DB"

while ! nc -z ${SQL_HOST} ${SQL_PORT}; do
  sleep .2s
done

echo "Connected!"

cd /usr/src/web/dj_nnapi

# Отладочный вывод содержимого директории
echo "Directory contents before running manage.py:"
ls -la

# Проверим директорию существования manage.py
if [ ! -f manage.py ]; then
  echo "manage.py not found!"
  exit 1
fi

python manage.py makemigrations nnmodel
python manage.py migrate nnmodel
python manage.py migrate 
export wsgi_start=1
python manage.py runserver 0.0.0.0:8000
python manage.py migrate 
#python3 -m celery -A dj_nnapi worker -P solo -l info --without-heartbeat --concurrency=1
# gunicorn -w 1 -b 0.0.0.0:8000 -t 120 --log-level debug dj_nnapi.wsgi:application

exec "$@"