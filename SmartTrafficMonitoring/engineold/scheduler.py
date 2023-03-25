from django.conf import settings
from django.contrib.auth.models import User
from task.models import Task
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
 
@shared_task()
def thirty_second_func():
    logger.info("I run every 30 seconds using Celery Beat")
    return "Done"

@shared_task(bind=True)
def process(self, task_id):
    from ooad import Detect
    d = Detect(Task.objects.get(id=task_id))
    d.detect_engine()
    return "Done"

@shared_task
def mul(x, y):
    return x * y