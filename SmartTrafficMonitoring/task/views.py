from django.shortcuts import render
import engine.ooad as ooad

# Create your views here.

def index(request):
    return render(request, 'index.html')

def report(request):
    return render(request, 'task/report.html')

def create_task(request):
    return render(request, 'task/createtask.html')

def showtask(request):
    if request.method == 'POST':
        input_text = request.POST.get('input')
        # process input_text and generate output
        output_text = "Output generated from input: " + input_text
        ooad.Vehicle.detect()
        # send output_text to template
        return render(request, 'showtask.html', {'output': output_text})
    return render(request, 'showtask.html')