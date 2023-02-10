from django.db import models

# Create your models here.

class Video(models.Model):
    name = models.CharField()
    time = models.FloatField()
    dir = models.CharField()

    def __str__(self):
        return f"{self.name} {self.time} {self.dir}"

    def creat_video(self):
        pass
    def view_video(self):
        pass
    def update_video(self):
        pass
    def delete_video(self):
        pass

