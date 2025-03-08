from ultralytics import YOLO

model = YOLO('models/yolov8x.pt')

results = model.predict('video/football_video.mp4', save=True)

print(results[0])

print('==================================')
for box in results[0].boxes:
    print(box)

