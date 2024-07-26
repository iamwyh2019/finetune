from ultralytics import YOLO

model = YOLO('runs/segment/affordance-nomosaic/weights/best.pt')

result = model.val(data='affordance.yaml', device = 0, plots=False)

print("mAP:", result.seg.map)
print("mAP50: ", result.seg.map50)
print("mAP75: ", result.seg.map75)
