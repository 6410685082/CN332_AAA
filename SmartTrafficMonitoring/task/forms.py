from django import forms
from .models import Task

class TaskForm(forms.Form):
    name = forms.CharField(label="Name: ", max_length=255)  
    location  = forms.CharField(label="Location: ", max_length=255) 
    # input_vdo = forms.FileField(label="Video")
    note = forms.CharField(label="Note: ", max_length=255, required=False) 

    class Meta:
        model = Task
        fields = ["name", "location", "input_vdo", "note"]