from ultralytics import YOLO

def main():
    
    modelname = "yolo11s.pt"
    task = "detect"
    datafile = "Images/Cyclist/data_tsinghua/data.yaml"
    epoch = 40
    device = "cuda"
    imgsz = 640
    batch = 8


    model = YOLO(modelname, task, True)

    model.train(data = datafile, epochs=epoch, batch=batch, device=device, imgsz=imgsz)






if __name__ == "__main__":
    main()