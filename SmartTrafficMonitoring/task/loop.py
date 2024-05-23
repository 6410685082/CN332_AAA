import cv2
from django.core.files.storage import FileSystemStorage
import os
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import json

def cap_picture(vdo, frame_path):
    vdo_name = os.path.basename(vdo)
    name = 'frame_' + vdo_name + '.jpg'
    video = cv2.VideoCapture(vdo)
    path = os.path.join(frame_path,name)
    while(True):
        ret, frame = video.read()
        if ret:
            cv2.imwrite(path,frame)
        break
    
    video.release()
    cv2.destroyAllWindows()
    return path

def draw_loop(filename,vdo, path):
    vdo_name = os.path.basename(vdo)
    name = 'scale-frame_' + vdo_name + '.jpg'
    capture = cap_picture(vdo, path)
    path = os.path.join(path,name)
    im = plt.imread(capture)
    fig, ax = plt.subplots()
    ax.imshow(im)

    with open(filename,'r') as file:
        file_data = json.load(file)
        if file_data["loops"] == []:
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            plt.savefig(path)
        for data in file_data["loops"]:
            pt1, pt2, pt3, pt4 = data["points"]
            verts = [
        (pt1["x"], pt1["y"]),  # left, bottom
        (pt2["x"], pt2["y"]),  # left, top
        (pt3["x"], pt3["y"]),  # right, top
        (pt4["x"], pt4["y"]),  # right, bottom
        (0, 0),  # ignored
            ]

            point = verts[:3]
            p_text = min(point)

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            path_poly = Path(verts, codes)

            patch = patches.PathPatch(path_poly, facecolor='none', lw=2, edgecolor='lightblue')
            ax.add_patch(patch)
            ax.plot([pt1["x"],pt2["x"]], [pt1["y"],pt2["y"]], color='blue')
            ax.annotate(data["name"]+" id: "+data["id"], p_text, color="green")
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            plt.savefig(path)
    file.close()

    return name

# add new loop to existed loop.json file?
def write_json(filename, name, id, x, y, clock):
    id_f = int(id)
    y_sum = 20*((id_f+1)**2)
    if clock == True:
        clock_direc = "clockwise" 
    else:
        clock_direc = "counterclockwise" 
    data = {
        "name": name,
        "id": id,
        "points": [
            {"x": x[0], "y": y[0]},
            {"x": x[1], "y": y[1]},
            {"x": x[2], "y": y[2]},
            {"x": x[3], "y": y[3]}
        ],
        "orientation": clock_direc,
        "summary_location":{"x":20,"y":f'{y_sum}'}
    }
    
    # get loops file
    try:
        with open(filename, 'r+') as file:
            file_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        file_data = {"loops": []}

    # get the loops section
    loops = file_data.get("loops", [])
    for idx, obj in enumerate(loops):
        if obj.get('id') == id:
            loops.pop(idx)
            break
        
    # append the new loop to the loop section
    loops.append(data)

    # save the file
    try:
        with open(filename, 'w') as file:
            # replace the old loops section with the new one
            json.dump({"loops": loops}, file, indent=4)
    except json.JSONEncodeError:
        print("Error: could not encode JSON data.")

def clear_loop(filename):
    try:
        with open(filename, 'w') as file:
            json.dump({"loops": []}, file, indent=4)
    except json.JSONEncodeError:
        print("Error: could not encode JSON data.")

def delete_loop(filename, loop_id):
    # look in the loop file
    # find the loop with that id

    # get loops file
    try:
        with open(filename, 'r+') as file:
            file_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        file_data = {"loops": []}
    
    # get the loops section
    loops = file_data.get("loops", []) #list

    # find the loop with the chosen id
    for loop in loops:
        if loop['id'] == str(loop_id) :
            # remove the chosen loop
            loops.remove(loop)
            break

    # save the file
    try:
        with open(filename, 'w') as file:
            # replace the old loops section with the new one
            json.dump({"loops": loops}, file, indent=4)
    except json.JSONEncodeError:
        print("Error: could not encode JSON data.")