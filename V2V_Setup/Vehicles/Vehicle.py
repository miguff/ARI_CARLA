from .Network import VehicleNetwork
import carla
from Utils.LidarUrils import Lidar
import open3d as o3d

#TODO: Add Noise
#TODO: Add Packet Loss
#TODO: Add Encryption



class CustomVehicle:
    def __init__(self, world, blueprint, spawn_point):
        self.world = world
        self.vehicle = world.try_spawn_actor(blueprint, spawn_point)  # Composition: vehicle is a member variable
        self.camera = []
        self.connected_vehicles = []

        self.attached_sensors = {}

        VehicleNetwork.register(self)

    def apply_control(self, control):
        self.vehicle.apply_control(carla.VehicleControl(throttle=control))  # Delegate call to member vehicle

    def get_location(self):
        return self.vehicle.get_location()

    def attach_camera(self, camera_bp, transform, callback=None):
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        if callback:
            camera.listen(callback)
        self.camera.append(camera)
        return camera
    
    def attach_lidar(self, lidar_bp, lidarclass: Lidar, point_list: o3d.geometry.PointCloud, lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8)), name = "Name", ):
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar.listen(lambda data: lidarclass.lidar_callback(data, point_list))
        self.attached_sensors[name] = lidar

    def set_autopilot(self, value=True):
        self.vehicle.set_autopilot(value)

    
    def get_speed(self):
        return self.vehicle.get_speed()

    def destroy(self):
        for sensor in self.camera:
            sensor.stop()
            sensor.destroy()
        self.vehicle.destroy()

    def send_signal(self):
        print("BroadCasting: Speed")
        VehicleNetwork.broadcast({"Speed": self.get_speed}, sender=self)

    def recieve_signal():
        pass
