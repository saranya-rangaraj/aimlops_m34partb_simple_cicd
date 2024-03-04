# pull python base image
FROM python:3.10

# copy application files
ADD /iris_model_api /iris_model_api/

# specify working directory
WORKDIR /iris_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]