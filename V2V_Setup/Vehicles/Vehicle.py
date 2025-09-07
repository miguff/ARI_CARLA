from .Network import VehicleNetwork
import carla
from Utils.LidarUrils import Lidar
import open3d as o3d
import numpy as np
from queue import Queue
from matplotlib import cm
import cv2

#TODO: Add Noise
#TODO: Add Packet Loss
#TODO: Add Encryption



class CustomVehicle:
    def __init__(self, world, blueprint, spawn_point):
        self.world = world
        self.vehicle = world.try_spawn_actor(blueprint, spawn_point)  # Composition: vehicle is a member variable
        self.connected_vehicles = []

        self.attached_sensors = {}
        self.queue = Queue()
        self.camera_attribute = {}

        VehicleNetwork.register(self)

        self.VIRIDIS = np.array(cm.get_cmap('viridis').colors)
        self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])

    def apply_control(self, control):
        self.vehicle.apply_control(carla.VehicleControl(throttle=control))  # Delegate call to member vehicle

    def get_location(self):
        return self.vehicle.get_location()

    def attach_camera(self, camera_bp, transform = carla.Transform(carla.Location(x=1.6, z=1.6)), name="RGB Cam"):
        """
        Setup the camera to record the video
        """
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        attribute_list = [image_w, image_h, fov]

        
        camera.listen(lambda data: self.sensor_callback(data, self.queue, name))
        self.attached_sensors[name] = camera
        self.camera_attribute[name] = attribute_list
    
    def attach_lidar(self, lidar_bp, lidarclass: Lidar, point_list: o3d.geometry.PointCloud, lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8)), name = "Name", ):
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        

        lidar.listen(lambda data: lidarclass.lidar_callback(data, self.queue, point_list, name))
        self.attached_sensors[name] = lidar
    def set_autopilot(self, value=True):
        #self.vehicle.set_autopilot(value)
        self.vehicle.apply_control(carla.VehicleControl(throttle=1))

    
    def get_speed(self):
        return self.vehicle.get_speed()

    def destroy(self):
        for sensor in self.attached_sensors.keys():
            self.attached_sensors[sensor].stop()
            self.attached_sensors[sensor].destroy()
        self.vehicle.destroy()
        self.world.tick()

    def send_signal(self):
        print("BroadCasting: Speed")
        VehicleNetwork.broadcast({"Speed": self.get_speed}, sender=self)

    def recieve_signal():
        pass

    def sensor_callback(self, data, queue, sensor_name):
        """
        This simple callback just stores the data on a thread safe Python Queue
        to be retrieved from the "main thread".
        """
        queue.put((sensor_name, data))


    def camera_to_lidar(self):
            
        #Lidar, this will work for now because we have the one camera
        sensor_dict = {}
        for i in range(2):
            sensors = self.queue.get(True, 1.0)
            sensor_dict[sensors[0]] = sensors[1]
        lidar_data = sensor_dict["OnlyLidar"]
        image_data = sensor_dict["RGB Cam"]

        #lidar_queue = self.queues["OnlyLidar"]
        #lidar_data = lidar_queue.get(True, 1.0)
        lidar = self.attached_sensors["OnlyLidar"]

        #image_queue = self.queues["RGB Cam"]
        #image_data = image_queue.get(True, 1.0)
        camera = self.attached_sensors["RGB Cam"]

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w, image_h, fov = self.camera_attribute["RGB Cam"]

        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0


        # Get the raw BGRA buffer and convert it to an array of RGB of
        # shape (image_data.height, image_data.width, 3).
        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
        im_array = im_array[:, :, :3][:, :, ::-1]

        
        # Get the lidar data and convert it to a numpy array.
        # print(np.asarray(lidar_data.points))
        # p_cloud_size = len(np.asarray(lidar_data.points))
        # print(p_cloud_size)
        # p_cloud = np.copy(np.frombuffer(np.asarray(lidar_data.points), dtype=np.dtype('f4')))
        # print(p_cloud)
        # print(len(p_cloud))
        # p_cloud = np.reshape(p_cloud, (p_cloud_size, 3))

        # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
        # focus on the 3D points.
        intensity = np.asarray(lidar_data.colors)
        p_cloud = np.asarray(lidar_data.points)
        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = lidar.get_transform().get_matrix()

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)

        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(K, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        #intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]
        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(int)
        v_coord = points_2d[:, 1].astype(int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        #intensity = 4 * intensity - 3
        #color_map = np.array([
        #    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 0]) * 255.0,
        #    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 1]) * 255.0,
        #    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 2]) * 255.0]).astype(int).T

        #if args.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
        plain_image = im_array.copy()
        dot_size = 1
        intensity = intensity*255.0
        for i in range(len(points_2d)):
            im_array[v_coord[i]-dot_size : v_coord[i]+dot_size,
                    u_coord[i]-dot_size : u_coord[i]+dot_size] = intensity[i]

        plain_image = plain_image[:, :, ::-1]
        return im_array, lidar_data, plain_image
        # Save the image using Pillow module.
        # image = Image.fromarray(im_array)
        # image.save("_out/%08d.png" % image_data.frame)