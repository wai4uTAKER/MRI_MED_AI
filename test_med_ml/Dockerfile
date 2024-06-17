FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Устанавливаем рабочую директорию
WORKDIR /usr/src/web

# Обновляем pip и устанавливаем необходимые системные пакеты
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 netcat

# Копируем и устанавливаем зависимости для нейронных сетей
COPY ./config/nnreq.txt ./config/nnreq.txt
RUN pip install -r ./config/nnreq.txt

# Копируем и устанавливаем общие зависимости
COPY ./config/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# Копируем весь проект в контейнер
COPY . /usr/src/web

# Устанавливаем аргумент и переменную окружения для MEDIA_ROOT
ARG MEDIA_ROOT
ENV MEDIA_ROOT=$MEDIA_ROOT

# Устанавливаем переменную окружения для немедленного вывода
ENV PYTHONUNBUFFERED=1

# Копируем скрипт entrypoint и делаем его исполняемым
RUN chmod +x /usr/src/web/entrypoint.sh

# Используем entrypoint для запуска
ENTRYPOINT ["/usr/src/web/entrypoint.sh"]