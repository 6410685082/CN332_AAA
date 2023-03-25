import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartTrafficMonitoring.settings')

app = Celery('SmartTrafficMonitoring')
<<<<<<< HEAD
=======
app.config_from_object('django.conf:settings', namespace='CELERY')
>>>>>>> origin

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

<<<<<<< HEAD
# Load task modules from all registered Django apps.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
=======
app.conf.beat_schedule = {
    "every_thirty_seconds": {
        "task": "engine.scheduler.thirty_second_func",
        "schedule": timedelta(seconds=30),
    },
}
app.autodiscover_tasks()
>>>>>>> origin
