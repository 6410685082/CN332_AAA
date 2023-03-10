from celery import shared_task
from datetime import datetime

@shared_task
def print_hello():
    print('Hello from Celery!')
    print(datetime.now())

@shared_task
def print_info():
    print("Hello, world!")
