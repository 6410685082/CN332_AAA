from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Status(models.Model):
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name}"
    
# class Video(models.Model):
#     name = models.CharField(max_length=255)
#     path = models.FileField(upload_to='videos/', null=True, verbose_name="")

#     def __str__(self):
#         return f"{self.name}: {self.path}"

class Task(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    input_vdo = models.FileField(upload_to='input')
    output_vdo = models.FileField(upload_to='output', null=True)
    status_id = models.ForeignKey(
        Status, on_delete=models.CASCADE, related_name="status_id")
    note = models.CharField(max_length=255)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_by")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} {self.location} {self.status_id}"
