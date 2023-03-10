from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore
from .models import Scheduler
from .tasks import process_scheduler
from django.utils import timezone

scheduler = BackgroundScheduler()
scheduler.add_jobstore(DjangoJobStore(), "default")

@scheduler.scheduled_job(trigger="interval", minutes=1)
def check_schedulers():
    for scheduler in Scheduler.objects.all():
        if scheduler.start_time <= timezone.now() <= scheduler.end_time:
            process_scheduler(scheduler)

scheduler.start()
