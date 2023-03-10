from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .models import Scheduler
from .forms import SchedulerForm

def scheduler(request):
    schedulers = Scheduler.objects.all().order_by('start_time')
    return render(request, 'scheduler.html', {'schedulers': schedulers})

def scheduler_create(request):
    if request.method == 'POST':
        form = SchedulerForm(request.POST)
        if form.is_valid():
            scheduler = form.save(commit=False)
            scheduler.save()
            return HttpResponseRedirect(reverse('scheduler'))
    else:
        form = SchedulerForm()
    return render(request, 'scheduler_form.html', {'form': form})

def scheduler_edit(request, pk):
    scheduler = get_object_or_404(Scheduler, pk=pk)
    if request.method == 'POST':
        form = SchedulerForm(request.POST, instance=scheduler)
        if form.is_valid():
            scheduler = form.save(commit=False)
            scheduler.save()
            return HttpResponseRedirect(reverse('scheduler'))
    else:
        form = SchedulerForm(instance=scheduler)
    return render(request, 'scheduler_form.html', {'form': form})

def scheduler_delete(request, pk):
    scheduler = get_object_or_404(Scheduler, pk=pk)
    scheduler.delete()
    return HttpResponseRedirect(reverse('scheduler'))
