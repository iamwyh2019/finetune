from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')

if __name__ == '__main__':
    # for X model, it has 23 layers, so freeze 22 layers
    # model.train(data='affordance.yaml', pretrained=True, epochs=300, save_period=20, device=0,
    #             freeze=22, cache=True, name='affordance-freezeall', exist_ok=True)
    # model.train(resume=True)

    # train without mosaic
    # model.train(data='affordance.yaml', pretrained=True, epochs=300, save_period=20, device=0,
    #             freeze=22, cache=True, name='affordance-nomosaic', exist_ok=True, mosaic=0)

    # train with the backbone frozen (freeze=10)
    # model.train(data='affordance.yaml', pretrained=True, epochs=300, save_period=20, device=0,
    #             freeze=10, cache=True, name='affordance-freezebackbone', exist_ok=True, mosaic=0)

    # train with lower LR (lr0=0.001, momentum=0.9)
    model.train(data='affordance.yaml', pretrained=True, epochs=300, save_period=20, device=0,
                freeze=10, cache=True, name='affordance-lr0.001', exist_ok=True, mosaic=0, lr0=0.001, momentum=0.9,
                optimizer='SGD')