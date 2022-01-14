#!/bin/bash
conda activate mrcnn-flask
gunicorn -c gunicorn_config.py app:app --daemon
