from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

model = YOLO('yolov8x-seg.pt')

if __name__ == '__main__':
    # train with lower LR (lr0=0.002, momentum=0.9)
    # freeze the backbone (the first 10 layers)
    model.train(data='data-vistas-removed.yaml', pretrained=True, epochs=150, device=0, exist_ok=True,
                freeze=10, cache=True, name='vistas-lr0.002', mosaic=0, lr0=0.002, momentum=0.9,
                optimizer='SGD')
    
    # plot training results
    # plot_results('runs/segment/vistas-lr0.002/results.csv', segment=True)