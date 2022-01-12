
bind = '0.0.0.0:8000'
backlog = 2048

workers = 2
worker_class = 'eventlet'
worker_connections = 1000
timeout = 300
keepalive = 5

preload = False

errorlog = '-'
loglevel = 'debug'
accesslog = None
