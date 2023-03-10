import os
from celery import Celery
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartTrafficMonitoring.settings')
app = Celery('SmartTrafficMonitoring')
app.conf.broker_url = 'redis://localhost:6379/0'

app.conf.timezone = 'Asia/Bangkok'

app.conf.beat_schedule = {
    "every_thirty_seconds": {
        "task": "users.tasks.thirty_second_func",
        "schedule": timedelta(seconds=30),
    },
}
app.autodiscover_tasks()