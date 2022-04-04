# -*- coding: UTF-8 -*-
import os
import sys
import json
import fnmatch
import skimage
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, make_response, request

import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn.visualize import save_images, color_splash


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


class ObjectDetection:
    def __init__(self, name, category, url):
        self.name = name
        self.category = list(set(category))
        self.url = url

    @property
    def data(self):
        return json.dumps({ 'name': self.name, 'category': self.category, 'url': self.url }, ensure_ascii = False)

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
        if fnmatch.fnmatch(file, today + "*") and fnmatch.fnmatch(file, "*" + "_mask.jpg"):
            results.append(file)

    # results = filter(lambda x: ".png" in x, os.listdir(ROOT_DIR + "/static/results"))
    resp = make_response(render_template('results.html', imgs=results[::-1]))

    return resp

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
    result_text.append(predict_object.data)
    resp["predict"] = result_text

    # Save image 
    skimage.io.imsave(file_name, splash)
    image.save(file_name_origin)
    save_image(image_array, file_name_mask, r['rois'], r['masks'],
            r['class_ids'], r['scores'], class_names,
            scores_thresh=0.8, mode=0)

    # Close image
    image.close()

    return make_response(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=False)
