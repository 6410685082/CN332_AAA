from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Status(models.Model):
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name}"
    
class Video(models.Model):
    name = models.CharField(max_length=255)
    path = models.URLField(max_length=255)

    def __str__(self):
        return f"{self.name}: {self.path}"

class Task(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    input_vdo_id = models.ForeignKey(
        Video, on_delete=models.CASCADE, related_name="input_vdo_id")
    res_vdo_id = models.ForeignKey(
        Video, on_delete=models.CASCADE, related_name="res_vdo_id", null=True)
    status_id = models.ForeignKey(
        Status, on_delete=models.CASCADE, related_name="status_id")
    note = models.CharField(max_length=255)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_by")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} {self.location} {self.status_id}"
