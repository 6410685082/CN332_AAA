from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required

import sys
sys.path.append("../user")

from user.models import UserInfo
from django.contrib.auth.models import User
from .forms import TaskForm
from .models import *

@login_required(login_url='/user/login')
def index(request):
    user_info = UserInfo.objects.get(user_id=request.user)

    return render(request, 'task/index.html', {
        'user': request.user,
        'user_info': user_info
    })

@login_required(login_url='/user/login')
def create_task(request):
    # def handle_uploaded_file(f):
    #     with open(f.name, 'wb+') as destination:
    #         for chunk in f.chunks():
    #             destination.write(chunk)

    if request.method == "POST":
        # form = TaskForm(request.POST, request.FILES)
        form = TaskForm(request.POST)

        if form.is_valid():
            name = form.cleaned_data['name']
            location = form.cleaned_data['location']
            # input_vdo = request.FILES['input_vdo']
            note = form.cleaned_data['note']

            task = Task.objects.create(
                name = name,
                location = location,
                # input_vdo = input_vdo,
                status_id = Status.objects.first(),
                note = note,
                created_by = request.user
            )

            return render(request, 'task/index.html', {
                'user': request.user,
            })
        else:
            return render(request, 'task/index.html', {
                'user': request.user,
                'message': 'Data is invalid.'
            })

    else:
        form = TaskForm()

        return render(request, 'task/create_task.html', {
            'user': request.user,
            'form': form
        })
    
@login_required(login_url='/user/login')
def view_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if task is None:
        return redirect(reverse('task:index'))

    return render(request, 'task/task.html', {
                'user': request.user,
                'task': task
            })

@login_required(login_url='/user/login')
def update_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if request.method == 'POST':
        pass
    else:
        form = TaskForm(request.POST)

        print(form)
        return 


@login_required(login_url='/user/login')
def delete_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    task.delete()

    if task is not None:
        task.delete()

    return redirect(reverse('task:index'))