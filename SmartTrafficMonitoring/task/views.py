from django.shortcuts import render
import engine.ooad as ooad
import torch
from engine.utils.general import strip_optimizer

# Create your views here.

def index(request):
    return render(request, 'index.html')

def report(request):
    return render(request, 'task/report.html')

def create_task(request):
    return render(request, 'task/createtask.html')

def showtask(request):
    opt_weights = 'yolov7.pt'
    if request.method == 'POST':
        input_text = request.POST.get('input')
        # process input_text and generate output
        output_text = "Output generated from input: " + input_text
        with torch.no_grad():
            for opt_weights in ['yolov7.pt']:
                ooad.Vehicle.detect()
                strip_optimizer(opt_weights)
        # send output_text to template
        return render(request, 'showtask.html', {'output': output_text})
    return render(request, 'showtask.html')