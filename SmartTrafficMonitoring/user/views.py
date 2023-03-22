from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import  UpdateUserForm
from .models import UserInfo


# Create your views here.
log_in = 'user/login.html'


def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse('user:profile'))
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
    role_id = user_info.role_id_id
    
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)

        if user_form.is_valid():
            user_form.save()

            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='/user/profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
    context = {'user_form': user_form, 'phone_number': phone_number, 'role_id': role_id}

    return render(request, 'user/profile.html', context)


