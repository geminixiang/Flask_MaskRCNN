FROM python:3.7

WORKDIR /workspace/

COPY . /workspace/

RUN pip install -r requirements.txt

EXPOSE 8000
CMD gunicorn -c gunicorn_config.py app:app
