import carla
from Vehicles import CustomVehicle
from Utils import Lidar
import time
import open3d as o3d

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

    #PointCloud Setup
    point_list = o3d.geometry.PointCloud()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*mini*')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[2]
    print(spawn_point)

    EgoVehicle = CustomVehicle(world, vehicle_bp, spawn_point)
    print(type(EgoVehicle))
    lidarClass = Lidar(world, blueprint_library, DELTA)
    lidar_bp = lidarClass.generate_lidar_bp()
    EgoVehicle.attach_lidar(lidar_bp, lidarClass, point_list, name="OnlyLidar")
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
    while frame < 200:
        if frame == 2:
            vis.add_geometry(point_list)
        vis.update_geometry(point_list)

        vis.poll_events()
        vis.update_renderer()
        # # This can fix Open3D jittering issues:
        time.sleep(0.005)
        world.tick()
        frame += 1

    EgoVehicle.destroy()


if __name__ == "__main__":
    main()