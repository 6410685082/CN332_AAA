from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Status(models.Model):
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name}"

class Task(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    loop = models.FileField(upload_to='loop/')
    input_vdo = models.FileField(upload_to='vdo_input/')
    output_vdo = models.FileField(upload_to='vdo_output/', null=True)
    report = models.FileField(upload_to='report_output/', null=True)
    status_id = models.ForeignKey(
        Status, on_delete=models.CASCADE, related_name="status_id")
    note = models.CharField(max_length=255, blank=True)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_by")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} {self.location} {self.status_id}"
