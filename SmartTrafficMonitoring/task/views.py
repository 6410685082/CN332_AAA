from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.utils import dateformat

import sys
sys.path.append("../user")
sys.path.append("../SmartTrafficMonitoring")

from SmartTrafficMonitoring.scheduler_adapter import CeleryAdapter
from user.models import UserInfo
from .models import *

@login_required(login_url='/user/login')
def index(request):
    user_info = UserInfo.objects.get(user_id=request.user)

    tasks = Task.objects.filter(created_by=request.user)

    for task in tasks:
        task.created_at = dateformat.format(task.created_at, 'd/m/Y')
        task.updated_at = dateformat.format(task.updated_at, 'd/m/Y')

    return render(request, 'task/index.html', {
        'user': request.user,
        'user_info': user_info,
        'tasks': tasks
    })

@login_required(login_url='/user/login')
def create_task(request):
    if request.method == 'POST':
        if request.FILES['loop'] and request.FILES['input_vdo']:
            if request.POST.get('input_type', 'new') == 'new':
                name = request.POST.get('name', None)
                location = request.POST.get('location', None)
            else:
                task_id = request.POST.get('task_id', None)
                task = Task.objects.get(pk=task_id)

                name = task.name
                location = task.location

            fs = FileSystemStorage()

            loop = request.FILES['loop']
            loop_filename = fs.save(loop.name, loop)
            uploaded_loop_url = fs.url(loop_filename)

            input_vdo = request.FILES['input_vdo']
            input_vdo_filename = fs.save(input_vdo.name, input_vdo)
            uploaded_input_vdo_url = fs.url(input_vdo_filename)

            note = request.POST.get('note', None)
            preset = request.POST.get('preset', False)

            task = Task.objects.create(
                name = name,
                location = location,
                loop = uploaded_loop_url,
                input_vdo = uploaded_input_vdo_url,
                status_id = Status.objects.first(),
                note = note,
                preset = preset,
                created_by = request.user
            )
            
            # scheduler = CeleryAdapter()
            # scheduler.process(task.id)

            return HttpResponseRedirect(reverse('task:view_task', args=(task.id,)))
        else:
            return redirect(reverse('task:create_task'))

    else:
        preset_tasks = Task.objects.filter(created_by=request.user, preset=True)
        
        return render(request, 'task/create_task.html', {
            'user': request.user,
            'preset_tasks': preset_tasks
        })
    
@login_required(login_url='/user/login')
def view_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if task is None:
        return redirect(reverse('task:index'))

    return render(request, 'task/view_task.html', {
                'user': request.user,
                'task': task
            })

@login_required(login_url='/user/login')
def update_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if task is None:
        return redirect(reverse('task:index'))
    else:
        if request.method == 'POST':
            name = request.POST.get('name')

            fs = FileSystemStorage()

            try:
                loop = request.FILES['loop']
                loop_filename = fs.save(loop.name, loop)
                uploaded_loop_url = fs.url(loop_filename)
            except:
                uploaded_loop_url = task.loop

            try:
                input_vdo = request.FILES['input_vdo']
                input_vdo_filename = fs.save(input_vdo.name, input_vdo)
                uploaded_input_vdo_url = fs.url(input_vdo_filename)
            except:
                uploaded_input_vdo_url = task.input_vdo

            location = request.POST.get('location')
            note = request.POST.get('note')

            Task.objects.filter(pk=task_id).update(
                name = name,
                loop = uploaded_loop_url,
                input_vdo = uploaded_input_vdo_url,
                location = location,
                note = note,
            )

            # update timestamp (updated_at)
            Task.objects.get(pk=task_id).save()

            return HttpResponseRedirect(reverse('task:view_task', args=(task.id,)))
        else:
            return render(request, 'task/update_task.html', {
                        'user': request.user,
                        'task': task
                    })


@login_required(login_url='/user/login')
def delete_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if task is not None:
        task.delete()

    return redirect(reverse('task:index'))

@login_required(login_url='/user/login')
def search_task(request):
    tasks = Task.objects.filter(created_by=request.user)

    keyword = request.GET.get('keyword', "")

    if keyword == "":
        return redirect(reverse('task:index'))

    tasks_by_name = Task.objects.filter(name__contains=keyword, created_by=request.user)
    tasks_by_location = Task.objects.filter(location__contains=keyword, created_by=request.user)
    tasks = tasks_by_name | tasks_by_location

    for task in tasks:
        task.created_at = dateformat.format(task.created_at, 'd/m/Y')
        task.updated_at = dateformat.format(task.updated_at, 'd/m/Y')

    return render(request, 'task/index.html', {
        'tasks': tasks
    })
