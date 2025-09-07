import numpy as np
from matplotlib import cm
import open3d as o3d


class Lidar:

    def __init__(self, world, blueprint_libray, delta,
                 dropoff_general_rate: float = 0.0, dropoff_intensity_limit: float =1.0, dropoff_zero_intensity: float = 0.0,
                 noise_stddev: float = 0.2, upper_fov: float = 15, lower_fov: float = -25.0, channels: float = 64 , range: float = 100, 
                 points_per_second: float = 200000):
        self.world = world
        self.blueprint_library = blueprint_libray
        self.delta = delta
        self.VIRIDIS = np.array(cm.get_cmap('plasma').colors)
        self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])
        self.LABEL_COLORS = np.array([
            (255, 255, 255), # None
            (70, 70, 70),    # Building
            (100, 40, 40),   # Fences
            (55, 90, 80),    # Other
            (220, 20, 60),   # Pedestrian
            (153, 153, 153), # Pole
            (157, 234, 50),  # RoadLines
            (128, 64, 128),  # Road
            (244, 35, 232),  # Sidewalk
            (107, 142, 35),  # Vegetation
            (0, 0, 142),     # Vehicle
            (102, 102, 156), # Wall
            (220, 220, 0),   # TrafficSign
            (70, 130, 180),  # Sky
            (81, 0, 81),     # Ground
            (150, 100, 100), # Bridge
            (230, 150, 140), # RailTrack
            (180, 165, 180), # GuardRail
            (250, 170, 30),  # TrafficLight
            (110, 190, 160), # Static
            (170, 120, 50),  # Dynamic
            (45, 60, 150),   # Water
            (145, 170, 100), # Terrain
        ]) / 255.0 # normalize each channel [0-1] since is what Open3D uses
        
        self.dropoff_general_rate = dropoff_general_rate
        self.dropoff_intensity_limit = dropoff_intensity_limit
        self.dropoff_zero_intensity = dropoff_zero_intensity
        self.noise_stddev = noise_stddev
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.channels = channels
        self.range = range
        self.points_per_second = points_per_second

    def sensor_callback(data, queue):
        """
        This simple callback just stores the data on a thread safe Python Queue
        to be retrieved from the "main thread".
        """
        queue.put(data)

    def generate_lidar_bp(self, arg=None):
        """Generates a CARLA blueprint based on the script parameters"""
        if arg==None:
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('noise_stddev', str(self.noise_stddev))
        elif  arg.semantic:
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        else:
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            if arg.no_noise:
                lidar_bp.set_attribute('dropoff_general_rate', str(self.dropoff_general_rate))
                lidar_bp.set_attribute('dropoff_intensity_limit', str(self.dropoff_intensity_limit))
                lidar_bp.set_attribute('dropoff_zero_intensity', str(self.dropoff_zero_intensity))
            else:
                lidar_bp.set_attribute('noise_stddev', str(self.noise_stddev))

        lidar_bp.set_attribute('upper_fov', str(self.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(self.lower_fov))
        lidar_bp.set_attribute('channels', str(self.channels))
        lidar_bp.set_attribute('range', str(self.range))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / self.delta))
        lidar_bp.set_attribute('points_per_second', str(self.points_per_second))
        return lidar_bp
    
    def lidar_callback(self, point_cloud, queue, point_list, sensor_name):
        """Prepares a point cloud with intensity
        colors ready to be consumed by Open3D"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 0]),
            np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 1]),
            np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 2])]

        # Isolate the 3D data
        points = data[:, :-1]

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        points[:, :1] = -points[:, :1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        # points = points[:, :-1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        
        queue.put((sensor_name, point_list))
