from django.db import models
from task.models import Task

# Create your models here.

class Drone(models.Model):
    name = models.CharField(max_length=64)
    location = models.CharField(max_length=64)
    status = models.CharField(max_length=64)
    note = models.CharField(max_length=64)
    battery = models.IntegerField()
    schedule = models.DateTimeField(max_length=64)
    height = models.FloatField(max_length=64)


    def __str__(self):
        return f"{self.name} {self.location} {self.status} {self.note} {self.battery} {self.schedule} {self.height}"
        
    def creat_task(self):
        pass
    def view_task(self):
        pass
    def update_task(self):
        pass
    def delete_task(self):
        pass
    def notification(self):
        pass
    def check_weather(self):
        pass

class Weather(models.Model):
    task = models.ForeignKey(Task,null=True, on_delete=models.SET_NULL )
    location = models.CharField(max_length=64)
    latitude = models.FloatField(max_length=64)
    longitude = models.FloatField(max_length=64)
    temp = models.FloatField(max_length=64)
    humidity = models.IntegerField()
    wind_speed = models.FloatField(max_length=64)
    weather_report = models.CharField(max_length=64)
    clouds = models.IntegerField()

    def __str__(self):
        return f"{self.location} {self.temp} {self.humidity} {self.wind_speed} {self.weather_report}"

    def view_weather(self):
        pass

