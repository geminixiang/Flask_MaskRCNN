#!/bin/bash
Program="mrcnn-flask"
pid=`ps ax | grep gunicorn | grep $Program | awk '{split($0,a," "); print a[1]}' | head -n 1`
if [ -z "$pid" ]; then
  echo "no gunicorn deamon - $Program"
else
  kill $pid
  echo "killed gunicorn deamon - $Program, PID=$pid"
fi
