from pathlib import Path
import os
from celery import Celery
from datetime import timedelta
from celery import shared_task
from engine.scheduler import Scheduler
from celery.utils.log import get_task_logger

class CeleryAdapter(Scheduler):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartTrafficMonitoring.settings')

        app = Celery('SmartTrafficMonitoring')
        app.config_from_object('SmartTrafficMonitoring.celery', namespace='CELERY')
        
        app.conf.timezone = 'Asia/Bangkok'

        app.autodiscover_tasks()

        def add_install_app(self):
            return ['django_celery_results','django_celery_beat']
        
        
        @shared_task()
        def adapt_thirty_second_func(self,):
            #logger = get_task_logger(__name__)
            #logger.info("I run every 30 seconds using Celery Beat")
            return "Done"

        @shared_task(bind=True)
        def adapt_process(self,task_id):
            from task.models import Task
            from ooad import Detect
            d = Detect(Task.objects.get(id=task_id))
            d.detect_engine()
            return "Done"

        @shared_task
        def adapt_mul(self,x, y):
            return x * y