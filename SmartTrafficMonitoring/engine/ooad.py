import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

#from experload import attempt_load
from datasets import LoadStreams, LoadImages
from general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from plots import plot_one_box
from torch_utils import select_device, load_classifier, \
    time_synchronized, TracedModel
from download_weights import download

# For SORT tracking
import skimage
from sort import *
from line_intersect import isIntersect
import json
import count_table
from models import LoopInfo


def check_clock_wise(p1, p2, p3):
    vec1 = (p2[0]-p1[0], p2[1]-p1[1])
    vec2 = (p3[0]-p2[0], p3[1]-p2[1])
    cross = vec2[0] * vec1[1] - vec2[1] * vec1[0]
    if cross >= 0:
        return True
    else:
        return False

count_boxes = []
loop_boxes = []  # loop statistics
time_stamp = 0  # time in second
save_dir = ""
names = ""

# ............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class Box:
    """" Calculates the relative bounding box from absolute pixel values. """
    def bbox_rel(*xyxy):
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    """Function to Draw Bounding boxes"""
    def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0]+box[2])/2), (int((box[1]+box[3])/2)))
            label = str(id) + ":" + str(cat) + ":" + names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, [255, 255, 255], 1)
            # cv2.circle(img, data, 6, color,-1)
        return img


class Loop:
    def __init__(self, data):
        self.name = data['name']
        self.id = data['id']
        self.points = data['points']
        self.orientation = data['orientation']
        self.summary_location = data['summary_location']

        # check if item entering or exit loop
    def check_enter_exit_loop(self, track):
        # loops = count_boxes["loops"]
        for loop in loops:
            # print(loop)
            pt0, pt1, pt2, pt3 = loop.points
            # check entering line
            if len(track.centroidarr) > 20:
                tp2, tp1 = track.centroidarr[-1], track.centroidarr[-20]
                # check entering line
                self.line_enter_check_and_set(loop, track, tp1, tp2, pt0, pt1)

                # check exit line left straight and right
                self.line_exit_check_and_set(
                    loop, track, tp1, tp2, pt1, pt2, "left")
                self.line_exit_check_and_set(
                    loop, track, tp1, tp2, pt2, pt3, "straight")
                self.line_exit_check_and_set(
                    loop, track, tp1, tp2, pt3, pt0, "right")

    # draw bouncing box to loop
    def draw_loops(img):
        # loops = count_boxes["loops"]
        for loop in loops:
            pt0, pt1, pt2, pt3 = loop.points

            cv2.line(img, (pt0["x"], pt0["y"]), (pt1["x"],
                     pt1["y"]), (255, 0, 0), 2)  # entering line
            cv2.line(img, (pt1["x"], pt1["y"]), (pt2["x"],
                     pt2["y"]), (255, 255, 0), 2)  # left line
            cv2.line(img, (pt2["x"], pt2["y"]), (pt3["x"],
                     pt3["y"]), (255, 255, 0), 2)  # straight
            cv2.line(img, (pt3["x"], pt3["y"]), (pt0["x"],
                     pt0["y"]), (255, 255, 0), 2)  # right
            cv2.putText(img, loop.name, (pt0["x"], pt0["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [
                        0, 255, 0], 2)

    def append_to_file(self, filename, text):
        with open(filename, "a") as file:
            file.write(text + "\n")

    def line_enter_check_and_set(self, loop, track, tp1, tp2, line_start, line_end):
        if isIntersect(tp1, tp2, line_start, line_end):
            if loop.id not in track.aoi_entered:
                track.aoi_entered.append(loop.id)
                msg = f'{loop.id},{track.id},{names[int(track.detclass)]},{time_stamp},ENTERED'
                self.append_to_file(str(save_dir)+"\\loop.txt", msg)
                print(
                    f'track {track.id} of type {track.detclass} entered loop {loop.id} at time ...{time_stamp}')

    # check if the object exit the line, if the first time mark it and prevent the re entry by setting the flag
    # line_side is the left or right border
    def line_exit_check_and_set(self, loop, track, tp1, tp2, line_start, line_end, line_side):
        if isIntersect(tp1, tp2, line_start, line_end):
            if loop.id in track.aoi_entered and loop.id not in track.aoi_exited:
                track.aoi_exited.append(loop.id)  # means already exit
                print(
                    f'track {track.id} of type {track.detclass} exit loop {loop.id } at time ...{time_stamp}')
                if (loop.orientation == "clockwise" and line_side == "left" or
                        loop.orientation == "counterclockwise" and line_side == "right"):  # turn left
                    loop_boxes[int(loop.id)].add_left(int(track.detclass))
                    msg = f'{loop.id},{track.id},{names[int(track.detclass)]},{time_stamp}, LEFT'
                    self.append_to_file(str(save_dir)+"\\loop.txt", msg)

                if (loop.orientation == "clockwise" and line_side == "right" or
                        loop.orientation == "counterclockwise" and line_side == "left"):  # turn right
                    loop_boxes[int(loop.id)].add_right(
                        int(track.detclass))  # turn right
                    msg = f'{loop.id},{track.id},{names[int(track.detclass)]},{time_stamp}, RIGHT'
                    self.append_to_file(str(save_dir)+"\\loop.txt", msg)

                if line_side == "straight":
                    loop_boxes[int(loop.id)].add_straight(int(track.detclass))
                    msg = f'{loop.id},{track.id},{names[int(track.detclass)]},{time_stamp}, STRAIGHT'
                    self.append_to_file(str(save_dir)+"\\loop.txt", msg)


class Vehicle:
    def detect(save_img=False):

        global save_dir
        global time_stamp
        global names
        global loop_boxes
        source, weights, view_img, save_txt, imgsz, trace, colored_trk = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk
        save_img = not opt.nosave and not source.endswith(
            '.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # .... Initialize SORT ....
        # .........................
        sort_max_age = 25  # 5
        sort_min_hits = 5  # 2
        sort_iou_thresh = 0.5  # 0.2
        sort_tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh)
        # .........................
        # Directories
        save_dir = Path(increment_path(Path(opt.project) /
                        opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        for loop in loops:
            loop_boxes.append(count_table.LoopCount(
                len(names), loop.summary_location, loop))
    #    for i in range(len(names)):
    #        cnts.append([0,0,0].copy()) #prepare array the same size as class type
    #    print(cnts)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        # ........Rand Color for every trk.......
        rand_color_list = []
        for i in range(0, 5005):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            rand_color_list.append(rand_color)
        # .........................
        img = None
        counttable = count_table.CountTable(img, None,
                                            list(names), ["Straight", "Left", "Right"], border_color=(0, 255, 0), text_color=(0, 0, 255))
        frame_count = 0
        for path, img, im0s, vid_cap in dataset:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            time_stamp = frame_count/fps  # calculate time stamp
            frame_count += 1
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # ..................USE TRACK FUNCTION....................
                    # pass an empty array to sort
                    dets_to_sort = np.empty((0, 6))

                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort,
                                                  np.array([x1, y1, x2, y2, conf, detclass])))

                    # Run SORT
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()

                    for track in tracks:
                        # tracking object passing line check and update
                        for loop in loops:
                            loop.check_enter_exit_loop(track)

                        # color = compute_color_for_labels(id)
                        # draw colored tracks
                        if colored_trk:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                      (int(track.centroidarr[i+1][0]),
                                       int(track.centroidarr[i+1][1])),
                                      rand_color_list[track.id], thickness=1)
                             for i, _ in enumerate(track.centroidarr)
                             if i < len(track.centroidarr)-1]
                        # draw same color tracks
                        else:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                      (int(track.centroidarr[i+1][0]),
                                       int(track.centroidarr[i+1][1])),
                                      (255, 0, 0), thickness=1)
                             for i, _ in enumerate(track.centroidarr)
                             if i < len(track.centroidarr)-1]

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        Box.draw_boxes(
                            im0, bbox_xyxy, identities, categories, names)

                    # ........................................................

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # add bounding box to the main image
            # line 1 entering line
            # cv2.line(im0s, (200,200),(400,20),(255,0,0),5)
            # line2 left turn line
            # cv2.line(im0s, (400,20),(600,200),(0,255,0),5)
            # line3  straigh line
            # cv2.line(im0s, (600,200),(400,400),(0,255,0),5)
            # line4 right turn line
            # cv2.line(im0s, (400,400),(200,200),(0,255,0),5)

            # add counting table
            counttable.img = im0s

            for lb in loop_boxes:
                lb.draw(counttable)
            # cv2.rectangle(im0s,(500,400),(900,550),(0,0,0),cv2.FILLED)
            # cv2.putText(im0s,"        Straight    Left        Right", (500, 420),cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, [0, 255, 0], 2)
            # cv2.putText(im0s,f"Car      {cnts[0][0]}           {cnts[0][1]}           {cnts[0][2]} ", (500, 450),cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, [0, 255, 0], 2)
            # cv2.putText(im0s,f"Bike      {cnts[1][0]}           {cnts[1][1]}           {cnts[1][2]} ", (500, 480),cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, [0, 255, 0], 2)
            # cv2.putText(im0s,f"Pickup   {cnts[2][0]}           {cnts[2][1]}           {cnts[2][2]} ", (500, 510),cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, [0, 255, 0], 2)

            Loop.draw_loops(im0s)
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    raise StopIteration

                # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_dir, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        # if save_txt or save_img:
            # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            # print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',help='augmented inference')
    parser.add_argument('--update', action='store_true',help='update all models')
    parser.add_argument('--project', default='runs/detect',help='save results to project/name')
    parser.add_argument('--name', default='object_tracking',help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--loop', default="loop.json",type=str, help='loop setting file')
    parser.add_argument('--loop-txt', action='store_true',  help='save history for each loop')
    parser.add_argument('--summary-txt', action='store_true', help='save summary for each loop')  # todo later

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)

    count_boxes = LoopInfo

    loops = [Loop(data) for data in count_boxes["loops"]]

    # check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                Vehicle.detect()
                strip_optimizer(opt.weights)
        else:
            Vehicle.detect()

import numpy as np
import random
import torch
import torch.nn as nn

from common import Conv, DWConv
from google_utils import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output





class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


