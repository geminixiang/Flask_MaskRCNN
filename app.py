# -*- coding: UTF-8 -*-
import os
import sys
import uuid
import json

import numpy as np
from PIL import Image
from flask import Flask, render_template, make_response, request
import tensorflow as tf
import skimage
import datetime
# Image Render
import random
import colorsys
from PIL import Image, ImageDraw, ImageFont
import fnmatch

from mrcnn.config import Config
import mrcnn.model as modellib

global graph, model

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

app = Flask(__name__, static_folder='static')
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = os.urandom(24)
# class_names = ['BG', '人', '房子', '樹', '時鐘', '人力車', '轎子', '船', '貓', '羊', '牛', '象', '斑馬', '傘', '領帶', '扇', '水瓶', '碗', '椅', '盆栽植物', '桌', '水桶', '書', '花瓶', '馬匹', '炮', '火車', '杯', '畫作', '燈籠', '路燈', '槍', '氣球', '馬車', '十字架', '橋', '刀具', '狗', '蠟燭', '鏢', '孩童', '軍隊', '電線桿', '望遠鏡', '雕像', '鼓', '軍旗', '城牆', '旗幟', '城門', '蒸汽機', '儀表', '畫像', '賽馬場', '馬廄', '琵琶', '匾額', '掛燈', '珊瑚', '船帆', '門', '守望台', '藤牌', '魚', '水稻', '亭子', '煙囪', '鐵路', '帳篷', '煤', '十字鎬', '眼鏡', '水龍', '棺材', '官員', '電報局', '英皇子', '日本國旗', '美國國旗', '西醫', '法國國旗', '絞刑', '水雷', '西妓', '教堂', '木馬', '窗', '壁燈', '大鐘', '花窗玻璃', '英國國旗', '船錨', '水煙斗', '天文台', '砲彈', '雞', '猴子', '香爐', '牌位', '軍官', '軍人', '女性', '潛水艇', '消防員', '屍體', '罪犯', '警官', '庸醫', '曾國藩', '當舖', '鳥籠', '濕版攝影相機', '井', '大水缸', '報紙', '公主', '鞭子', '戎克船/䑸', '龍舟', '長喇叭', '鑼', '玉皇宮']
class_names = ['BG', '人', '房子', '樹', '船', '扇', '椅', '桌', '孩童', '門', '窗', '軍人', '女性', '帽', '圖像標題', '圖像文字', '印章文字']
# class_names = ['BG', '人', '房子', '樹', '船', '扇', '椅', '盆栽植物', '桌', '杯', '燈籠', '槍', '刀具', '孩童', '旗幟', '門', '窗', '軍人', '女性', '帽', '官兵', '官員', '官役', '蠟燭', '蛤蠣', '圖像標題', '圖像文字', '印章文字']
model_file = "./models/500x20_all.h5"
try:
    import time, stat
    model_modify_time = time.ctime(os.stat(model_file)[stat.ST_MTIME])
except:
    model_modify_time = "NULL"

class InferenceConfig(Config):
    NAME = "dian"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.8

graph = tf.get_default_graph()
model = modellib.MaskRCNN( mode="inference", 
        config=InferenceConfig(), 
        model_dir=ROOT_DIR)
try:
    model.load_weights(model_file, by_name=True)
    print('><><><><> Model load success <><><><><')
except:
    print("can't find .h5 model file")

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def save_image(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
               scores_thresh=0.1, save_dir=None, mode=0):
    """
        https://github.com/matterport/Mask_RCNN/pull/38/files
        image: image array
        image_name: image name
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        save_dir: (optional) the path to store image
        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
    mode_list = [0, 1, 2, 3]
    assert mode in mode_list, "mode's value should in mode_list %s" % str(mode_list)

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    useful_mask_indices = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        useful_mask_indices.append(i)

    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return

    colors = random_colors(len(useful_mask_indices))

    if mode != 3:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)

    if mode != 1:
        for index, value in enumerate(useful_mask_indices):
            masked_image = apply_mask(masked_image, masks[:, :, value], colors[index])

    masked_image = Image.fromarray(masked_image)

    if mode == 3:
        masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
        return

    draw = ImageDraw.Draw(masked_image)
    colors = np.array(colors).astype(int) * 255

    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        score = scores[value]
        label = class_names[class_id]

        y1, x1, y2, x2 = boxes[value]
        if mode != 2:
            color = tuple(colors[index])
            draw.rectangle((x1, y1, x2, y2), outline=color)

        # Label
        font = ImageFont.truetype("./NotoSansTC-Regular.otf", size=25)
        draw.text((x1, y1), "%s %f" % (label, score), (255, 0, 0), font)

    masked_image.save(os.path.join(save_dir, image_name))

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # print(image)
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

@app.route('/', methods=['GET'])
def index():
    global result_text
    result_text = []
    results = filter(lambda x: ".jpg" in x, os.listdir(ROOT_DIR + "/static/sample"))
    resp = make_response(render_template('index.html', imgs=results, model=model_file, model_modify_time=model_modify_time, class_names=class_names))

    return resp


@app.route('/results', methods=['GET'])
def result():
    # <><><><><><><><><><>  Get Today's Files  <><><><><><><><><><>
    folder = "./static/results/"
    folderContent = os.listdir(folder)
    # 資料夾內容先排序，以便後續顯示有個邏輯性
    folderContent.sort()
    results = []
    today = "{:%Y%m%d}".format(datetime.datetime.now())

    for i, file in sorted(enumerate(folderContent)):
        # print(file)
        if fnmatch.fnmatch(file, today + "*") and fnmatch.fnmatch(file, "*" + "_mask.jpg"):
            results.append(file)

    # results = filter(lambda x: ".png" in x, os.listdir(ROOT_DIR + "/static/results"))
    resp = make_response(render_template('results.html', imgs=results[::-1]))

    return resp

class ObjectDetection:
    name = ""
    category = []
    url = ""

    def __init__(self, name, category, url):
        self.name = name
        self.category = list(set(category))
        self.url = url
        self.total = json.dumps({ 'name': self.name, 'category': self.category, 'url': self.url }, ensure_ascii = False)

    def data(self):
        return self.total


@app.route('/api/maskrcnn', methods=['POST'])
def MaskRCNN():
    resp = dict()
    resp["ok"] = True
    request_image_name = request.files['image'].filename
    
    # Image open
    image = Image.open(request.files['image'])
    # 避免PNG會多一個透明通道(長,寬,4)，預設是(長,寬,3)，這邊轉換一下。
    image = image.convert("RGB")
    image_array = np.array(image)

    print(image_array.shape)
  
    # 確認是原始graph
    with graph.as_default():
        r = model.detect([image_array], verbose=1)[0]
        splash = color_splash(image_array, r['masks'])
    
    time_code = "/static/results/{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    file_name = ROOT_DIR + time_code + "_splash.jpg"
    file_name_origin = ROOT_DIR + time_code + "_origin.jpg"
    file_name_mask = ROOT_DIR + time_code + "_mask.jpg"
    
    # ROIS output
    # for i in range(len(r['rois'])):
    #     mask_result = Image.fromarray(r['masks'][:, :, i])
    #     mask_result.save(ROOT_DIR + "/static/results/" + file_name + "_" + str(i) + ".png")

    # return image & detect objects
    predict_object = ObjectDetection(name = request_image_name, category = [class_names[i] for i in r['class_ids']], url = time_code + "_mask.jpg")
    result_text.append(predict_object.data())
    resp["predict"] = result_text

    # Save image 
    skimage.io.imsave(file_name, splash)
    image.save(file_name_origin)
    save_image(image_array, file_name_mask, r['rois'], r['masks'],
            r['class_ids'], r['scores'], class_names,
            scores_thresh=0.8, mode=0)

    return make_response(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=False)

