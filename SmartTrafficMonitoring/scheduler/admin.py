from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import ScheduledTask

class SchedulerAdmin(admin.ModelAdmin):
    list_display = ('task_id', 'status', 'result')

admin.site.register(ScheduledTask, SchedulerAdmin)
