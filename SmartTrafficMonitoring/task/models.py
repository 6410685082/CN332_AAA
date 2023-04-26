from django.db import models
from django.contrib.auth.models import User

class Status(models.Model):
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name}"

class Task(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    latitude = models.FloatField(max_length=64, null=True, blank=True)
    longitude = models.FloatField(max_length=64, null=True, blank=True)
    loop = models.FileField(upload_to='loop/', null=True, blank=True)
    input_vdo = models.FileField(upload_to='vdo_input/', null=True, blank=True)
    output_vdo = models.FileField(upload_to='vdo_output/', null=True, blank=True)
    report = models.FileField(upload_to='report_output/', null=True, blank=True)
    status_id = models.ForeignKey(
        Status, on_delete=models.CASCADE, related_name="status_id")
    note = models.CharField(max_length=255, blank=True)
    preset = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_by")
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return f"{self.name} {self.location}"

class UploadFile(models.Model):
    video_file = models.FileField(upload_to='videos/')
    loop_txt_file = models.FileField(upload_to='texts/')

    def __str__(self):
        return f"{self.video_file} {self.loop_txt_file}"
    
class Notification(models.Model):
    detail = models.CharField(max_length=255)
    task = models.ForeignKey(
        Task, on_delete=models.CASCADE, related_name="task_id")
    already_read = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="notification_for")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.created_by} {self.detail}"