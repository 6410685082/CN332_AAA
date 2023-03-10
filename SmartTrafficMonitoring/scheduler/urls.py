from django.urls import path
from .views import TaskResultList, scheduler

urlpatterns = [
    path('task-results/', TaskResultList),
    path('scheduler' , scheduler, name='scheduler'),
]
