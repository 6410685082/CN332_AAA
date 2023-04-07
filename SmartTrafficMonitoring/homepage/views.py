from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.utils import timezone
from django.utils import dateformat
from django.db.models import Q

#import sys
#sys.path.append("../user")

#from user.models import UserInfo

#@login_required(login_url='/user/login')
def index(request):
    return render(request, 'homepage/homepage.html')
