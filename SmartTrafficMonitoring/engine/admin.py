from django.contrib import admin
from .models import LoopInfo

# Register your models here.

class LoopInfoAdmin(admin.ModelAdmin):
    list_display = ('name', 'loop_id')

admin.site.register(LoopInfo, LoopInfoAdmin)