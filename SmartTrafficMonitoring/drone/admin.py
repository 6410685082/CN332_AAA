from django.contrib import admin
from .models import *

class WeatherAdmin(admin.ModelAdmin):
    list_display = ['location', 'wind_speed']

admin.site.register(Weather, WeatherAdmin)