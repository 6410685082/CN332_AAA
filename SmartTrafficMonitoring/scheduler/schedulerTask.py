from .celery import Celery #apscheduler.schedulers.background import BackgroundScheduler
#from django_apscheduler.jobstores import DjangoJobStore
from .schedulerTask import process_scheduler
from django.utils import timezone

scheduler = Celery()
scheduler.autodiscover_tasks()

@scheduler.scheduled_job(trigger="interval", minutes=1)
def check_schedulers():
    for scheduler in process_scheduler.objects.all():
        #if scheduler.start_time <= timezone.now() <= scheduler.end_time:
        process_scheduler(scheduler)

scheduler.start()

def process_scheduler(scheduler):
    # Perform the desired task based on the scheduler data
    print(f"Processing scheduler: {scheduler.task_id}")
    # For example, you could trigger an email or a message to be sent at the scheduled time



