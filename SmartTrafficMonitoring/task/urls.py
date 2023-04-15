from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'task'

urlpatterns = [
    path('index/', views.index, name='index'),
    path('create-task/', views.create_task, name='create_task'),
    path('view-task/<int:task_id>', views.view_task, name='view_task'),
    path('update-task/<int:task_id>', views.update_task, name='update_task'),
    path('delete-task/<int:task_id>', views.delete_task, name='delete_task'),
    path('search-task', views.search_task, name='search_task'),
    path('custom-loop/<int:task_id>', views.custom_loop, name='custom_loop'),
    path('clear-loop/<int:task_id>', views.clear_loop, name='clear_loop'),
    path('schedule/<int:task_id>', views.schedule, name='schedule'),
    path('notification/', views.view_notification, name='view_notification')
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
