import os
from celery import Celery
from datetime import timedelta

#Celery, Celery Beat and Redis settings
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = 'django-db'
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_SERIALIZER = 'json'
CELERY_TIMEZONE = "Asia/Bangkok"
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'