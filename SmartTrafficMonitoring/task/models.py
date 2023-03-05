from django.db import models

# Create your models here.

class Video(models.Model):
    name = models.CharField(max_length = 200)
    time = models.FloatField()
    dir = models.CharField(max_length = 100)

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

