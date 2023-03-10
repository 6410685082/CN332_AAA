from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .forms import ScheduledTaskForm
from .models import ScheduledTask
from django.http import JsonResponse
from scheduler.models import ScheduledTask
from celery import shared_task
from django_celery_beat.models import IntervalSchedule, PeriodicTask
from .models import RandomIDField

@shared_task
def scheduler(request):
    if request.method == 'POST':
        form = ScheduledTaskForm(request.POST)
        if (request.methode == "POST"):
            scheduler_create()
            form.save()
    else:
        form = ScheduledTaskForm()

    tasks = ScheduledTask.objects.all()
    showScheduler = PeriodicTask.objects.all()
    return render(request, 'scheduler.html', {'form': form, 'tasks': tasks, 'shows': showScheduler})

@shared_task
def scheduler_create():
    interval = IntervalSchedule.objects.create(every=10, period=IntervalSchedule.SECONDS)
    PeriodicTask.objects.create(
    interval= interval,                  # we created this above.
    name= RandomIDField.generate_id,          # simply describes this periodic task.
    task= RandomIDField.generate_id,  # name of task.
    )
    if request.method == 'POST':
        form = ScheduledTaskForm(request.POST)
        if form.is_valid():
            scheduler = form.save(commit=False)
            scheduler.save()
            return HttpResponseRedirect(reverse('scheduler'))
    else:
        form = ScheduledTaskForm()
    return render(request, 'scheduler_form.html', {'form': form})

def scheduler_edit(request, id):
    scheduler = get_object_or_404(ScheduledTask, task_id=id)
    if request.method == 'POST':
        form = ScheduledTaskForm(request.POST, instance=scheduler)
        if form.is_valid():
            scheduler = form.save(commit=False)
            scheduler.save()
            return HttpResponseRedirect(reverse('scheduler'))
    else:
        form = ScheduledTaskForm(instance=scheduler)
    return render(request, 'scheduler_form.html', {'form': form})

def scheduler_delete(request, id):
    scheduler = get_object_or_404(ScheduledTask, task_id=id)
    scheduler.delete()
    return HttpResponseRedirect(reverse('scheduler'))

def task_progress(request, task_id):
    task = ScheduledTask.objects.get(id=task_id)
    progress = task.get_progress()
    return JsonResponse({'progress': progress})

class TaskResultList():
    queryset = ScheduledTask.objects.all()
