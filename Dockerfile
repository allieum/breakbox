FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt requirements.txt
# RUN apt-get install python-pygame
RUN apt-get update
RUN apt-get install -y git curl libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0
RUN apt-get install -y libasound2
# RUN apt-get install -y alsa-utils
# RUN apt-get install -y alsa-utils alsa-lib alsa-plugins
RUN pip install pygame
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
# CMD ["python", "-m" , "break.py", "--host=0.0.0.0"]