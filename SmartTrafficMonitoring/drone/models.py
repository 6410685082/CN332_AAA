from django.db import models

# Create your models here.

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
    location = models.CharField()
    temp = models.FloatField()
    humidity = models.IntegerField()
    wind_speed = models.FloatField()
    weather_report = models.CharField()

    def __str__(self):
        return f"{self.location} {self.temp} {self.humidity} {self.wind_speed} {self.weather_report}"

    def view_weather(self):
        pass