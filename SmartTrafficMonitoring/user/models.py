from django.db import models

# Create your models here.

class RoleID(models.Model):
    name = models.CharField()

    def __str__(self):
        return f"{self.name} "

    def creat_role(name):
        pass
    def view_role(name):
        pass
    def update_role(name):
        pass
    def delete_role(name):
        pass

class Drone(models.Model):
    name = models.CharField()
    location = models.CharField()
    status = models.CharField()
    note = models.CharField()
    battery = models.IntegerField()
    schedule = models.DateTimeField()
    height = models.FloatField()


    def __str__(self):
        return f"{self.name} {self.location} {self.status} {self.note} {self.battery} {self.schedule} {self.height}"
        
    def creat_task(name):
        pass
    def view_task(name):
        pass
    def update_task(name):
        pass
    def delete_task(name):
        pass
    def notification():
        pass
    def check_weather():
        pass
    
class Video(models.Model):
    name = models.CharField()
    time = models.FloatField()
    dir = models.CharField()

    def __str__(self):
        return f"{self.name} {self.time} {self.dir}"

    def creat_video(name):
        pass
    def view_video(name):
        pass
    def update_video(name):
        pass
    def delete_video(name):
        pass

class Weather(models.Model):
    location = models.CharField()
    temp = models.FloatField()
    humidity = models.IntegerField()
    wind_speed = models.FloatField()
    weather_report = models.CharField()

    def __str__(self):
        return f"{self.location} {self.temp} {self.humidity} {self.wind_speed} {self.weather_report}"

    def check_weather():
        pass