from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.utils import dateformat
from . import loop
import os
from django.conf import settings

import sys
sys.path.append("../user")
sys.path.append("../SmartTrafficMonitoring")

from SmartTrafficMonitoring.scheduler_adapter import CeleryAdapter
from user.models import UserInfo
from .models import *
from drone.views import CheckDrone
from drone.models import Weather

@login_required(login_url='/user/login')
def index(request):
    user_info = UserInfo.objects.get(user_id=request.user)

    # tasks = Task.objects.filter(created_by=request.user)
    tasks = Task.objects.all()

    for task in tasks:
        task.created_at = dateformat.format(task.created_at, 'd/m/Y')
        task.updated_at = dateformat.format(task.updated_at, 'd/m/Y')

    if len(tasks) == 0:
        message = "You need to create task."
    else:
        message = None

    return render(request, 'task/index.html', {
        'user': request.user,
        'user_info': user_info,
        'tasks': tasks,
        'message': message
    })

@login_required(login_url='/user/login')
def create_task(request):
    if request.method == 'POST':
        if request.FILES['input_vdo']:
            if request.POST.get('input_type', 'new') == 'new':
                name = request.POST.get('name', None)
                location = request.POST.get('location', None)
            else:
                task_id = request.POST.get('task_id', None)
                task = Task.objects.get(pk=task_id)

                name = task.name
                location = task.location
            
            fs = FileSystemStorage()
            if request.FILES.get('loop'):
                loop = request.FILES['loop']
                loop_filename = fs.save(loop.name, loop)
            else:
                default_path = os.path.join(settings.BASE_DIR, 'loop.json')
                media_file_path = os.path.join(settings.MEDIA_ROOT, 'loop.json')
                with open(default_path, 'rb') as f:
                    loop_filename = fs.save(media_file_path, f)

            uploaded_loop_url = fs.url(loop_filename)

            input_vdo = request.FILES['input_vdo']
            input_vdo_filename = fs.save(input_vdo.name, input_vdo)
            uploaded_input_vdo_url = fs.url(input_vdo_filename)

            note = request.POST.get('note', None)
            preset = request.POST.get('preset', False)
            latitude = request.POST.get('latitude')
            flag = True
            try:
                float(latitude)
            except ValueError:
                    flag = False
            if flag:
                latitude = float(latitude)
            else:
                latitude = 14.068542
            longitude = request.POST.get('longitude', None)
            flag = True
            try:
                float(longitude)
            except ValueError:
                    flag = False
            if flag:
                longitude = float(longitude)
            else:
                longitude = 100.605965

            task = Task.objects.create(
                name = name,
                location = location,
                loop = uploaded_loop_url,
                input_vdo = uploaded_input_vdo_url,
                status_id = Status.objects.first(),
                note = note,
                preset = preset,
                created_by = request.user,
                latitude = latitude,
                longitude = longitude
            )
            weather = Weather.objects.create(task = task, location="Loading. Please refresh to load"
            , wind_speed=0, latitude=latitude, longitude=longitude, 
            temp=0, weather_report='', humidity=0, clouds = 0)
            
            return HttpResponseRedirect(reverse('task:custom_loop', args=(task.id,)))
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
    weather = Weather.objects.get(task=task_id)
    task = Task.objects.filter(pk=task_id).first()
    CheckDrone(request,task_id = task, lat = weather.latitude ,lon = weather.longitude)

    Notification.objects.filter(task=task).update(
        already_read = True
    )

    my_task = Task.objects.get(id=task_id)
    rows = []
    if my_task.report:
        file_path = my_task.report.path
        
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split(',')
                row_dict = {
                    'loop': fields[0],
                    'track': fields[1],
                    'vehicle': fields[2],
                    'time': fields[3],
                    'direction': fields[4],
                }
                rows.append(row_dict)
        
        report_url = my_task.report.url
    else:
        report_url = None
    if task is None:
        return redirect(reverse('task:index'))

    return render(request, 'task/view_task.html', {
                'rows': rows,
                'user': request.user,
                'task': task,
                'output_vdo': task.output_vdo.url if task.output_vdo else '',
                'report_url': report_url,
                'weather' : weather,
            })

@login_required(login_url='/user/login')
def update_task(request, task_id):
    task = Task.objects.filter(pk=task_id, created_by=request.user).first()

    if task is None:
        return redirect(reverse('task:index'))
    else:
        if request.method == 'POST':
            name = request.POST.get('name')
            latitude = request.POST.get('latitude')
            longitude = request.POST.get('longitude')

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
            weather = Weather.objects.get(task=task_id)
            flag = True
            try:
                float(latitude)
            except ValueError:
                flag = False
            if flag:
                latitude = float(latitude)
            else:
                latitude = weather.latitude
            flag = True
            try:
                float(longitude)
            except ValueError:
                flag = False
            if flag:
                longitude = float(longitude)
            else:
                longitude = weather.longitude
            weather.latitude = latitude
            weather.longitude = longitude
            weather.save()
            Task.objects.filter(pk=task_id).update(
                name = name,
                loop = uploaded_loop_url,
                input_vdo = uploaded_input_vdo_url,
                location = location,
                note = note,
                latitude = latitude,
                longitude = longitude
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

    tasks_by_name = Task.objects.filter(name__contains=keyword)
    tasks_by_location = Task.objects.filter(location__contains=keyword)
    tasks = tasks_by_name | tasks_by_location

    for task in tasks:
        task.created_at = dateformat.format(task.created_at, 'd/m/Y')
        task.updated_at = dateformat.format(task.updated_at, 'd/m/Y')

    if len(tasks) == 0:
        message = "Sorry, we couldn't find any results"
    else:
        message = None

    return render(request, 'task/index.html', {
        'tasks': tasks,
        'message': message
    })

@login_required(login_url='/user/login')
def custom_loop(request,task_id):
    task = Task.objects.get(pk = task_id)
    video = os.path.join(settings.MEDIA_ROOT, str(task.input_vdo).split("/")[-1])
    loop_path = os.path.join(settings.MEDIA_ROOT, str(task.loop).split("/")[-1])
    frame_path = os.path.join(settings.MEDIA_ROOT, "capture")
    isExist = os.path.exists(frame_path)
    if not isExist:
        os.makedirs(frame_path)

    if request.method == 'POST':
        name = request.POST.get("name")
        loop_id = request.POST.get("id")

        x1 = int(request.POST.get("x1"))
        y1 = int(request.POST.get("y1"))

        x2 = int(request.POST.get("x2"))
        y2 = int(request.POST.get("y2"))

        x3 = int(request.POST.get("x3"))
        y3 = int(request.POST.get("y3"))

        x4 = int(request.POST.get("x4"))
        y4 = int(request.POST.get("y4"))

        if request.POST.get("clock-choice") == "clockwise":
            clock = True
        else:
            clock = False

        x = [x1,x2,x3,x4]
        y = [y1,y2,y3,y4]

        loop.write_json(loop_path,name,loop_id,x,y,clock)

    frame = os.path.join("capture",loop.draw_loop(loop_path,video,frame_path))
    return render(request, 'task/custom_loop.html', {
        'frame': frame,
        'task_id': task_id,
        'loop_path': task.loop
            })

def clear_loop(request,task_id):
    task = Task.objects.get(pk = task_id)
    video = os.path.join(settings.MEDIA_ROOT, str(task.input_vdo).split("/")[-1])
    loop_path = os.path.join(settings.MEDIA_ROOT, str(task.loop).split("/")[-1])
    frame_path = os.path.join(settings.MEDIA_ROOT, "capture")

    loop.clear_loop(loop_path)

    frame = os.path.join("capture",loop.draw_loop(loop_path,video,frame_path))
    return render(request, 'task/custom_loop.html', {
        'frame': frame,
        'task_id': task_id,
        'loop_path': task.loop
            })

def schedule(request, task_id):
    scheduler = CeleryAdapter()
    scheduler.process(task_id)
    return HttpResponseRedirect(reverse('task:view_task', args=(task_id,)))

def view_notification(request):
    user_info = UserInfo.objects.get(user_id=request.user)
    notifications = Notification.objects.filter(created_by=request.user)

    for noti in notifications:
        noti.task.created_at = dateformat.format(noti.task.created_at, 'd/m/Y')
        noti.already_read = 'read' if noti.already_read else 'unread'

    return render(request, 'task/notifications.html', {
        'user': request.user,
        'user_info': user_info,
        'notifications': notifications
    })

def map(request):
    return render(request,'drone/map.html')