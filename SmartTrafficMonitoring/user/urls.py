from django.urls import path
from user.views import ChangePasswordView


from . import views
app_name = 'user'

urlpatterns = [
    #path('', views.index, name='index'),
    path('login', views.login_view, name='login'),
    path('logout', views.logout_view, name='logout'),
    path('profile', views.profile, name='profile'),
    path('change_pw', ChangePasswordView.as_view(), name='change_pw'),
    path('create_user', views.create_user, name='create_user'),
    path('edit_profile', views.edit_profile, name='edit_profile'),
]