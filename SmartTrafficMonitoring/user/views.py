from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import  UpdateUserForm, UserCreationForm
from .models import UserInfo
from django.contrib.auth.views import PasswordChangeView
from django.contrib.messages.views import SuccessMessageMixin

import sys
sys.path.append("..")
from task.models import *

log_in = 'user/login.html'

def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse('task:index'))
        else:
            return render(request, log_in, {
                'message': 'Invalid credentials.'
            })
    return render(request, log_in)

def logout_view(request):
    logout(request)
    return render(request, log_in, {
                'message': 'You are logged out.'
            })


@login_required(login_url='/user/login')
def profile(request):    
    user_info = UserInfo.objects.get(user_id=request.user)
    phone_number = user_info.phone_number
    role_id = user_info.role_id.name
    context = {'phone_number': phone_number, 'role_id': role_id}
    return render(request, 'user/profile.html', context)

def create_user(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'User created successfully!')
            return redirect('user:profile')
    else:
        form = UserCreationForm()
    return render(request, 'user/create_user.html', {'form': form})

def edit_profile(request):
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'User created successfully!')
            return redirect('user:profile')
    else:
        user_form = UserCreationForm()
    return render(request, 'user/create_user.html', {'user_form': user_form})

class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'user/change_pw.html'
    success_message = "Successfully Changed Your Password"
    success_url = '/user/login'
