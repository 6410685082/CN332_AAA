from django.contrib import admin
from .models import *

class StatusAdmin(admin.ModelAdmin):
    list_display = ['name']

class TaskAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'loop', 'input_vdo', 'output_vdo', 'status_id', 'note', 'created_by', 'created_at', 'updated_at']

admin.site.register(Status, StatusAdmin)
admin.site.register(Task, TaskAdmin)