from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class RoleID(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} "

    def creat_role(self):
        pass
    def view_role(self):
        pass
    def update_role(self):
        pass
    def delete_role(self):
        pass


class UserInfo(models.Model):
    user_id = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="user_id")
    role_id = models.ForeignKey(
        RoleID, on_delete=models.CASCADE, related_name="role")
    phone_number = models.CharField(max_length=10)
    

    def __str__(self):
        return f'{ self.user_id.username } { self.role_id }'
    
    def creat_user(self):
        pass
    def view_user(self):
        pass
    def update_user(self):
        pass
    def delete_user(self):
        pass
    def login(self):
        pass
    def logout(self):
        pass