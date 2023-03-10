from django.urls import path
from . import views

urlpatterns = [
    path('report', views.report, name='report'),
    path('createtask', views.create_task, name='createtask'),
]