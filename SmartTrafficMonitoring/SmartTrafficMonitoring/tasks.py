import torch
import ooad
from celery import shared_task

@shared_task
def detect():
    opt_weights = 'yolov7.pt'
    with torch.no_grad():
            for opt_weights in ['yolov7.pt']:
                ooad.Vehicle.detect()
                #strip_optimizer(opt_weights)