from django.shortcuts import render, redirect
from django.contrib.auth.models import User

def index(request):
    # if not request.user.is_authenticated:
    #     return render(request, 'users/login.html', status=403)

    # user = User.objects.get(pk=request.user.id)
    # user_info = UserInfo.objects.get(user_id=user)

    # return render(request, 'index.html', {
    #     'user': user,
    #     'user_info': user_info,
    # })

    return render(request, 'task/index.html')