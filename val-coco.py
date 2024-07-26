from ultralytics import YOLO

print("========== Baseline ==========")
model = YOLO('yolov8x-seg.pt')
result = model.val(data = 'coco.yaml', device = 0, plots=False)

print("mP:", result.seg.mp)
print("mR:", result.seg.mr)
print("mAP50: ", result.seg.map50)
print("mAP75: ", result.seg.map75)