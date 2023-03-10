from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import Scheduler

class SchedulerAdmin(admin.ModelAdmin):
    list_display = ('title', 'start_time', 'end_time')

admin.site.register(Scheduler, SchedulerAdmin)
