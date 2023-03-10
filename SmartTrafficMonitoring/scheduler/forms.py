from django import forms
from .models import Scheduler

class SchedulerForm(forms.ModelForm):
    class Meta:
        model = Scheduler
        fields = ['title', 'start_time', 'end_time']
