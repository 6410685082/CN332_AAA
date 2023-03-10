from django.db import models

class Scheduler(models.Model):
    title = models.CharField(max_length=200)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    def __str__(self):
        return self.title

from .models import Scheduler

def process_scheduler(scheduler):
    # Perform the desired task based on the scheduler data
    print(f"Processing scheduler: {scheduler.title}")
    # For example, you could trigger an email or a message to be sent at the scheduled time
