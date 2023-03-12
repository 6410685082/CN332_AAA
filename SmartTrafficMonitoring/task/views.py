from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'index.html')

def report(request):
    return render(request, 'task/report.html')

def create_task(request):
    return render(request, 'task/createtask.html')