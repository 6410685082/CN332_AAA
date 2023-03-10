from django.db import models
from datetime import datetime
import random
import string

# Create your models here.

class ScheduledTask(models.Model):
    task_id = models.CharField(max_length=255, unique=True, default=0)
    status = models.CharField(max_length=50, default=0)
    result = models.TextField(blank=True, null=True, default="")
    date_done = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.task_id

class RandomIDField(models.CharField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_length', 20)
        super().__init__(*args, **kwargs)

    def generate_id(self):
        length = self.max_length
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def pre_save(self, model_instance, add):
        value = getattr(model_instance, self.attname)
        if not value:
            value = self.generate_id()
            setattr(model_instance, self.attname, value)
        return value

    