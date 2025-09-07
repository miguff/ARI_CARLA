import carla
from Vehicles import CustomVehicle
from Utils import Lidar
import time
import open3d as o3d
from queue import Queue
from queue import Empty
import cv2
import traceback
from ultralytics import YOLO
import numpy as np
import supervision as sv

DELTA = 0.05

def main():
    """Main of the Script"""
    #SETUP The World
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    world.apply_settings(settings)
    model = YOLO("best.pt")
    tracker = sv.ByteTrack()
    tracker.reset()

    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )

    try:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[94]
        blueprint_library = world.get_blueprint_library()

        #summmon cyclist
        cyclist = bicycleorigin(world, spawn_points)

        #PointCloud Setup
        point_list = o3d.geometry.PointCloud()

        
        
        #Camera Bluebrints
        camera_bp = blueprint_library.filter("sensor.camera.rgb")[0]
        vehicle_bp = blueprint_library.filter('*mini*')[0]
        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "420")


        


        EgoVehicle = CustomVehicle(world, vehicle_bp, spawn_point)
        lidarClass = Lidar(world, blueprint_library, DELTA, points_per_second=170000)
        lidar_bp = lidarClass.generate_lidar_bp()
        EgoVehicle.attach_lidar(lidar_bp, lidarClass, point_list, name="OnlyLidar")
        EgoVehicle.attach_camera(camera_bp)
        EgoVehicle.send_signal()
        EgoVehicle.set_autopilot()

        #Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True



    
        frame = 0
        while frame < 600:
            world.tick()
            cyclist.apply_control(carla.VehicleControl(throttle=1))

            image, lidar_data, plain_image = EgoVehicle.camera_to_lidar()
            plain_image = np.ascontiguousarray(plain_image)
            if frame == 2:
                vis.add_geometry(lidar_data)
            vis.update_geometry(lidar_data)
            
            #Cyclist detection
            result = model(plain_image, conf=0.1, verbose=False)[0]
            # print(result)
            detections = sv.Detections.from_ultralytics(result)
            
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            detections = tracker.update_with_detections(detections=detections)
            

            # labels = [
            #     f"{class_name} {confidence:.2f}"
            #     for class_name, confidence
            #     in zip(detections['class_name'], detections.confidence)
            # ]

            annotated_frame = image.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections)
            # for detect in detections:
            #     print(detect)
                # for box in result.boxes:
                #     # Extract box coordinates and other details
                #     x1, y1, x2, y2 = box.xyxy[0]
                #     center_x = int((x1 + x2) / 2)  # x-center of the bicycle
                #     center_y = int((y1 + y2) / 2)  # y-center of the bicycle
                #     conf = box.conf[0]            # Confidence score
                #     cls = box.cls[0]
                #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                #     label = f"{model.names[int(cls)]}: {conf:.2f}"
                #     cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



            cv2.imshow("Lidar to Camera", plain_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            vis.poll_events()
            vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            frame += 1
    except Exception as e:
        print(traceback.print_exc(e))
    finally:
        print("Delete Everything")
        EgoVehicle.destroy()
        cyclist.destroy()

def bicycleorigin(world, spawn_poiunts):
    bicycle_bp = world.get_blueprint_library().filter('*crossbike*')
    bicycle_start_point = spawn_poiunts[1]

    bicycle = world.try_spawn_actor(bicycle_bp[0], bicycle_start_point)
    bicyclepos = carla.Transform(bicycle_start_point.location + carla.Location(x=-3, y=3.5))
    bicycle.set_transform(bicyclepos)
    for _ in range(40):  # wait for half a second
        #"Still Falling - Cyclist")
        world.tick()
        time.sleep(0.05)
    return bicycle



if __name__ == "__main__":
    main()