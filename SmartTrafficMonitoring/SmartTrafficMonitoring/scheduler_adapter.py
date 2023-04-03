from pathlib import Path
import os
from celery import Celery
from datetime import timedelta
from celery import shared_task
from engine.scheduler import Scheduler
from celery.utils.log import get_task_logger
from django.core.files.base import ContentFile
import io

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
            from task.models import Task, Status, UploadFile
            from ooad import Detect, Vehicle
            BASE_DIR = Path(__file__).resolve().parent.parent
            task = Task.objects.get(id=task_id)
            

            d = Detect(Task.objects.get(id=task_id))

            save_path, save_direc = d.detect_engine()


            Task.objects.filter(pk=task_id).update(
                status_id = Status.objects.last(),
            )

            # update timestamp (updated_at)
            Task.objects.get(pk=task_id).save()
            
            
            video_file_path = os.path.join(BASE_DIR, save_path)
            text_file_path = os.path.join(BASE_DIR, save_direc,'loop.txt')

            # Open the files and read their contents
            with open(video_file_path, 'rb') as video_file:
                video_content = io.BytesIO(video_file.read())
            task.output_vdo.save('video.mp4', ContentFile(video_content.getvalue()))
            with open(text_file_path, 'r') as text_file:
                text_content = text_file.read()
            task.report.save('text.txt', ContentFile(text_content))

            video_file.close()
            text_file.close()

            return "Done"

        @shared_task
        def adapt_mul(self,x, y):
            return x * y