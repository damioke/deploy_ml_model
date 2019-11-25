
# Pull image
FROM python:3.7-slim

# create a directory in the container image.
#RUN mkdir /APP_HOME

# Set enviroment variable for the app directory
ENV APP_HOME /app

# switch to the app directory
WORKDIR $APP_HOME

# Copy all files from local to container image
COPY . ./
#COPY ./APP_HOME /APP_HOME

# install production dependencies.
#RUN pip install -r /requirements.txt

RUN pip install pathlib>=1.0.1 numpy>=1.17.4 pandas>=0.25.3 scikit-learn>=0.21.3 Flask>=1.1.1 datetime>=4.3 gunicorn>=20.0.0

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app


