#!/bin/bash
conda activate mrcnn-flask
gunicorn -w 1 -b 0.0.0.0:5000 app:app --daemon
