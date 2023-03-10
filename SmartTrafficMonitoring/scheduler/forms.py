from django import forms
from .models import ScheduledTask

class ScheduledTaskForm(forms.ModelForm):
    class Meta:
        model = ScheduledTask
        fields = ['task_id', 'status', 'result']
        
