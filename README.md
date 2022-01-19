# Flask_Mask_RCNN

***
## Description
`Folder`
* `static` contains js、css、Results/Sample images
* `mrcnn` [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
* `templates` for Front-end page of Flask

`File`
* `model.h5` tensorflow model file (就不上傳了)
* `NotoSansTC-Regular.otf` for Pillow to display Chinese
* `app.py` for Flask api serve
* `gunicorn_config.py` for gunicorn setting
* `start.sh` for quick start script

## Quick Start
```bash
conda create -n mrcnn-flask python=3.7
conda activate mrcnn-flask
pip install -r requirements.txt
bash -i start.sh
```

## Docker
```bash
## Build image
docker build -t f416720001/flask-mrcnn:0.0.1 .

## Run container with port 8000
docker run -p 8000:8000 f416720001/flask-mrcnn:0.0.1

## Or Run with cpu/memory limits
docker run --cpus="4.0" --memory="2g" -p 8000:8000 f416720001/flask-mrcnn:0.0.1
```

## 有坑必學
[Pillow 中文顯示](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html)

全域graph、model，避免以下狀況
[ValueError: Tensor Tensor("mrcnn_detection/Reshape_1:0", shape=(1, 100, 6), dtype=float32) is not an element of this graph](https://github.com/keras-team/keras/issues/2397#issuecomment-254919212)

## Ref.
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

[Kuludu/Mask_RCNN_TSR](https://github.com/Kuludu/Mask_RCNN_TSR)

[gunicorn](https://docs.gunicorn.org/en/stable/settings.html)
