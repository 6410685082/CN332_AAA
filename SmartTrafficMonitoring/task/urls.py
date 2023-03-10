from django.urls import path
from . import views
# from django.conf import settings
# from django.conf.urls.static import static

app_name = 'task'

urlpatterns = [
    path('', views.index, name='index'),
    path('create-task/', views.create_task, name='create_task'),
    path('view-task/<int:reserve_id>', views.view_task, name='view_task'),
    path('update-task/', views.update_task, name='update_task'),
    path('delete-task/<int:reserve_id>', views.delete_task, name='delete_task'),

] 
# + static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)