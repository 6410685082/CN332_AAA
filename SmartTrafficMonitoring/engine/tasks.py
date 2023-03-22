from django.conf import settings
from django.contrib.auth.models import User
 
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
 
@shared_task()
def thirty_second_func():
    logger.info("I run every 30 seconds using Celery Beat")
    return "Done"

@shared_task
def process(task):
    from ooad import detect_engine
    d = detect_engine(task)
    d.detect_engine()
    return "Done"

@shared_task
def mul(x, y):
    return x * y