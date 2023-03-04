from django.contrib import admin
from .models import UserInfo, RoleID

class UserInfoAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'role_id_id', 'phone_number')

class RoleIDAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')

admin.site.register(RoleID, RoleIDAdmin)
admin.site.register(UserInfo, UserInfoAdmin)