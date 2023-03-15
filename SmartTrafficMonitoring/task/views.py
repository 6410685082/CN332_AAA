from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.utils import timezone
from django.utils import dateformat
from django.db.models import Q

import sys
sys.path.append("../user")

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
            name = request.POST.get('name', None)
            location = request.POST.get('location', None)

            fs = FileSystemStorage()

            loop = request.FILES['loop']
            loop_filename = fs.save(loop.name, loop)
            uploaded_loop_url = fs.url(loop_filename)

            input_vdo = request.FILES['input_vdo']
            input_vdo_filename = fs.save(input_vdo.name, input_vdo)
            uploaded_input_vdo_url = fs.url(input_vdo_filename)

            note = request.POST.get('note', None)

            task = Task.objects.create(
                name = name,
                location = location,
                loop = uploaded_loop_url,
                input_vdo = uploaded_input_vdo_url,
                status_id = Status.objects.first(),
                note = note,
                created_by = request.user
            )

            return HttpResponseRedirect(reverse('task:view_task', args=(task.id,)))
        else:
            return redirect(reverse('task:create_task'))

    else:
        return render(request, 'task/create_task.html', {
            'user': request.user
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
            location = request.POST.get('location')
            note = request.POST.get('note')

            Task.objects.filter(pk=task_id).update(
                name=name,
                location=location,
                note=note
            )

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