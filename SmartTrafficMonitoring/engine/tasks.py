from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.core.mail import get_connection
from django.contrib.auth.models import User
from engine.arial_car_track.ooad import detect_engine
 
from celery import shared_task
from celery.utils.log import get_task_logger
 
logger = get_task_logger(__name__)
 
@shared_task()
def thirty_second_func():
    logger.info("I run every 30 seconds using Celery Beat")
    return "Done"

@shared_task
def process():
    detect_engine.detect_engine()
    return "Done"

@shared_task
def mul(x, y):
    return x * y