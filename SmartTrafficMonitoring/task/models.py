from django.db import models
from django.contrib.auth.models import User

class Status(models.Model):
    name = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.name}"

class Task(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    loop = models.FileField(upload_to='loop/')
    input_vdo = models.FileField(upload_to='vdo_input/')
    output_vdo = models.FileField(upload_to='vdo_output/', null=True, blank=True)
    report = models.FileField(upload_to='report_output/', null=True, blank=True)
    status_id = models.ForeignKey(
        Status, on_delete=models.CASCADE, related_name="status_id")
    note = models.CharField(max_length=255, blank=True)
    preset = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_by")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} {self.location}"

class UploadFile(models.Model):
    video_file = models.FileField(upload_to='videos/')
    loop_txt_file = models.FileField(upload_to='texts/')

    def __str__(self):
        return f"{self.video_file} {self.loop_txt_file}"