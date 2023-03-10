import os
from celery import Celery


# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartTrafficMonitoring.settings')

app = Celery('SmartTrafficMonitoring')

# Set the broker URL and result backend for Celery
app.conf.broker_url = 'redis://localhost:6379/0'
app.conf.result_backend = 'redis://localhost:6379/0'

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()
