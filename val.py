from ultralytics import YOLO
from create_mapping import coco_to_vistas, simplified_ids, simplified_names

# print("========== YOLO on COCO ==========")
# model = YOLO('yolov8x-seg.pt')
# result = model.val(data = 'coco.yaml', device = 0, plots=False)

# print("mAP:", result.seg.map)
# print("mAP50: ", result.seg.map50)
# print("mAP75: ", result.seg.map75)


print("========== Baseline ==========")
model = YOLO('yolov8x-seg.pt')
result = model.val(data = 'data-vistas.yaml', device = 0, plots=False,
                   custom_mapping=coco_to_vistas, class_mapping = simplified_ids, class_mapping_names = simplified_names)

print("mAP:", result.seg.map)
print("mAP50: ", result.seg.map50)
print("mAP75: ", result.seg.map75)



print("========== Finetuned ==========")
model = YOLO('best-outdoor.pt')
result = model.val(data = 'data-vistas.yaml', device = 0, plots=False, class_mapping = simplified_ids, class_mapping_names = simplified_names)

print("mAP:", result.seg.map)
print("mAP50: ", result.seg.map50)
print("mAP75: ", result.seg.map75)


# print("========== Baseline + Freeze Layer ==========")
# model = YOLO('runs/segment/freeze-layer/weights/best.pt')
# result = model.val(data = 'data-vistas.yaml', device = 0, plots=False)

# print("mAP:", result.seg.map)
# print("mAP50: ", result.seg.map50)
# print("mAP75: ", result.seg.map75)