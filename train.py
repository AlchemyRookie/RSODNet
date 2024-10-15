import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/RSODNet.yaml')
    # model.load('yolov8s.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=800,
                epochs=150,
                lr0=0.001,
                lrf=0.01,  # (float) final learning rate (lr0 * lrf)
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='AdamW',
                project='runs/train',
                name='RSOD',
                )