from django.shortcuts import render

# Create your views here.

def report(request):
    return render(request, 'task/report.html')

def create_task(request):
    return render(request, 'task/createtask.html')