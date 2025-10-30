#!/usr/bin/env python3

import carla
import rclpy
from rclpy.node import Node
import random
import time
import sys
import math

# --- IMPORTS FOR MAP PUBLISHING ---
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
# --- Import for QoS (Latched Publisher) ---
from rclpy.qos import QoSProfile, DurabilityPolicy

# --- IMPORTS FOR TF ---
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from carla_driver.Utils import Lidar
#from Utils import Lidar
import open3d as o3d
import numpy as np
from queue import Queue
DELTA = 0.05

class CarlaClientNode(Node):

    def __init__(self):
        super().__init__("carla_client_node")
        self.get_logger().info("CARLA Client Node started.")
        
        # Initialize class members
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)  # 5-second timeout
        self.world = self.client.get_world()
        self.ego_actor = None
        self.sensor_list = []
        self.queue = Queue()
        #// Setup world for testing
            
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = DELTA
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.server_version = self.client.get_server_version()

            #// Lidare Blueprint setup
        blueprint_library = self.world.get_blueprint_library()
        self.lidarClass = Lidar(self.world, blueprint_library, DELTA, points_per_second=170000)
        self.lidar_bp = self.lidarClass.generate_lidar_bp()
        # --- Granularity Setting ---
        # Controls the distance between map points (in meters)
        # Smaller = smoother curves, but more markers
        self.map_granularity = 2.0 
        self.get_logger().info(f"Using map granularity: {self.map_granularity}m")
        # --- End ---

        # --- DEFINE A LATCHED QoS ---
        # This makes the map "latched", so new subscribers (like RViz)
        # will receive the last published map message automatically.
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- ADD MAP PUBLISHER (with new QoS) ---
        self.map_publisher = self.create_publisher(
            MarkerArray, 
            '/carla/map', 
            map_qos  # Use the latched QoS profile
        )
        self.ego_marker_publisher = self.create_publisher(
            Marker,
            '/carla/ego_vehicle_marker',
            map_qos
        )
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            'carla/pointcloud',
            map_qos
        )
        # --- END ---

        # --- ADD DYNAMIC TF BROADCASTER ---
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # --- Connect to CARLA ---
        self.connect_to_carla()
        
    
    def publish_ego_tf(self):
        """Publishes the ego vehicle's transform."""
        if not self.ego_actor:
            return
            
        # Get the vehicle's full transform from CARLA
        transform = self.ego_actor.get_transform()
        loc = transform.location
        rot = transform.rotation

        # Create the transform message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"        # Parent frame
        t.child_frame_id = "base_link" # Child frame (vehicle's base)

        # --- Apply CARLA (LHS, Z-up) to ROS (RHS, Z-up, ENU) conversion ---

        # 1. Translation: x=x, y=-y, z=z
        t.transform.translation.x = loc.x
        t.transform.translation.y = -loc.y
        t.transform.translation.z = loc.z

        # 2. Rotation: R=R, P=-P, Y=-Y
        # Convert degrees to radians
        roll_rad = math.radians(rot.roll)
        pitch_rad = math.radians(-rot.pitch) # Negate pitch
        yaw_rad = math.radians(-rot.yaw)     # Negate yaw

        # Convert Euler to Quaternion (using ZYX order for yaw, pitch, roll)
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        # --- End Conversion ---

        # Send the dynamic transform
        self.tf_broadcaster.sendTransform(t)
    
    def publish_ego_marker(self):
        """
        Publishes a CUBE Marker representing the ego vehicle.
        This marker is static relative to the 'base_link' frame.
        """
        if not self.ego_actor:
            return

        bbox = self.ego_actor.bounding_box
        center = bbox.location
        extent = bbox.extent

        marker = Marker()
        marker.header.frame_id = "base_link" # <-- ATTACHED TO THE TF
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ego_vehicle"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # --- Pose (relative to "base_link") ---
        # The marker's pose is the center of the bounding box relative
        # to the actor's transform (which is "base_link").
        # We apply the same CARLA->ROS coordinate conversion to the offset.
        marker.pose.position.x = float(center.x)
        marker.pose.position.y = float(-center.y)
        marker.pose.position.z = float(center.z)
        marker.pose.orientation.w = 1.0 # No rotation relative to base_link

        # --- Scale (full size) ---
        # The scale is the full size (extent is half-size)
        marker.scale.x = float(extent.x * 2.0)
        marker.scale.y = float(extent.y * 2.0)
        marker.scale.z = float(extent.z * 2.0)

        # --- Color ---
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8) # Blue

        # --- Lifetime ---
        # 0 = infinite lifetime (since it's latched)
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        self.get_logger().info(f"Publishing ego vehicle marker to /carla/ego_vehicle_marker")
        self.ego_marker_publisher.publish(marker)


    def connect_to_carla(self):
        try:
            # Try to connect to the CARLA server
            self.get_logger().info('Attempting to connect to CARLA server at localhost:2000...')
            
            
            
            
            self.get_logger().info(f'Successfully connected to CARLA server!')
            self.get_logger().info(f'  Server Version: {self.server_version}')
            self.get_logger().info(f'  Current Map: {self.world.get_map().name}')

            # --- PUBLISH MAP ONCE ---
            self.get_logger().info('Publishing static map topology...')
            
            # --- END ---

            # --- Spawn and set autopilot *after* successful connection ---
            self.spawn_ego()
            
            if self.ego_actor:
                # --- FIX: Get the Traffic Manager and set the port ---
                self.get_logger().info('Setting ego vehicle to autopilot...')
                self.timer_period = DELTA
                
                tm = self.client.get_trafficmanager(8001) # Get TM on default port
                tm_port = tm.get_port()
                self.ego_actor.set_autopilot(True, tm_port) # Tell autopilot to use this TM
                self.get_logger().info(f'Ego vehicle using Traffic Manager on port {tm_port}')
                self.timer = self.create_timer(self.timer_period, self.on_timer_step)
                # --- End Fix ---

                rclpy.get_default_context().on_shutdown(self.on_shutdown)
            else:
                self.get_logger().error("Failed to spawn ego vehicle. Shutting down.")
                rclpy.shutdown()


        except RuntimeError as e:
            self.get_logger().error('Failed to connect to CARLA server.')
            self.get_logger().error('Is the CARLA server running? (e.g., ./CarlaUE4.sh --ros2)')
            self.get_logger().error(f'Error details: {e}')
            rclpy.shutdown() # Shutdown if connection fails
        except Exception as e:
            self.get_logger().error(f'An unexpected error occurred: {e}')
            rclpy.shutdown() # Shutdown on other errors

    def on_timer_step(self):
        """
        Main timer callback. Publishes dynamic data.
        """
        self.world.tick()
        self.publish_map()
        self.publish_ego_marker()
        
        # --- PUBLISH DYNAMIC TF ---
        self.publish_ego_tf()
        # --- END ---

        #// Visualize lidar
        sensors = self.queue.get(True, 1.0)
        lidar_data = sensors[1]

        intensity = np.asarray(lidar_data.colors)
        p_cloud = np.asarray(lidar_data.points)
        self.publish_lidar(p_cloud, intensity)
    
    def publish_lidar(self, pcloud, intensity):
        pcloud[:, 1] = -pcloud[:, 1]
        intensity = (intensity * 255).astype(np.uint8)

        # Pack into uint32
        intensity = np.left_shift(intensity[:,0].astype(np.uint32), 16) | \
                    np.left_shift(intensity[:,1].astype(np.uint32), 8)  | \
                    intensity[:,2].astype(np.uint32)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pcloud.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12  # 3 float32 values * 4 bytes each
        msg.row_step = msg.point_step * pcloud.shape[0]

        # Combine XYZ and RGB into Nx4 array
        cloud_data = np.zeros((pcloud.shape[0], 4), dtype=np.float32)
        cloud_data[:, 0:3] = pcloud.astype(np.float32)
        # Trick: view RGB uint32 as float32
        cloud_data[:, 3] = intensity.view(np.float32)

        # Convert to bytes
        msg.data = np.asarray(pcloud, np.float32).tobytes()
        self.point_cloud_publisher.publish(msg)
        self.get_logger().info('Published LIDAR PointCloud2')
        
    def publish_map(self):
        """
        Publishes the static CARLA map topology as a MarkerArray.
        Samples the map with a finer granularity for smooth curves.
        """
        carla_map = self.world.get_map()
        topology = carla_map.get_topology()
        marker_array = MarkerArray()
        id_counter = 0
        
        for segment in topology:
            w1 = segment[0]
            w2 = segment[1]
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "carla_map_topology"
            marker.id = id_counter
            id_counter += 1
            
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.3  # line width
            marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            
            # --- START GRANULARITY MODIFICATION ---
            # Sample the segment at the specified granularity
            
            marker.points = []
            
            # Add the first waypoint
            loc1 = w1.transform.location
            marker.points.append(Point(x=float(loc1.x), y=float(-loc1.y), z=float(loc1.z)))
            
            # Get all waypoints from w1 until the end of the lane segment
            waypoint_list = w1.next_until_lane_end(self.map_granularity)

            for w in waypoint_list:
                loc = w.transform.location
                marker.points.append(Point(x=float(loc.x), y=float(-loc.y), z=float(loc.z)))

            # Add the final waypoint (w2)
            loc2 = w2.transform.location
            marker.points.append(Point(x=float(loc2.x), y=float(-loc2.y), z=float(loc2.z)))
            
            # --- END GRANULARITY MODIFICATION ---
            
            marker_array.markers.append(marker)
            
        self.map_publisher.publish(marker_array)
        self.get_logger().info(f"Published {id_counter} map segments to /carla/map")

    def spawn_ego(self):
        self.get_logger().info('Spawning ego vehicle...')
        blueprint_library = self.world.get_blueprint_library()

        try:
            ego_bp = blueprint_library.find('vehicle.dodge.charger_2020')
        except IndexError:
             self.get_logger().error("Blueprint 'vehicle.dodge.charger_2020' not found!")
             return

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
             self.get_logger().error("No spawn points found in map!")
             return

        spawn_point = random.choice(spawn_points)
        
        # Spawn the actor
        self.ego_actor = self.world.try_spawn_actor(ego_bp, spawn_point)

        if self.ego_actor:
            self.get_logger().info(f'Ego vehicle spawned with ID: {self.ego_actor.id}')
            # Give the simulator a moment to settle
            
            time.sleep(0.5)
        else:
            self.get_logger().error(f"Failed to spawn ego vehicle at {spawn_point.location}")


        #// Attach sensors
        #Lidar
        #PointCloud Setup
        self.point_list = o3d.geometry.PointCloud()
        self.lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8))
        self.lidar = self.world.spawn_actor(self.lidar_bp, self.lidar_transform, attach_to=self.ego_actor)
        self.lidar.listen(lambda data: self.lidarClass.lidar_callback(data, self.queue, self.point_list, sensor_name="OnlyLidar"))

    def on_shutdown(self):
        """
        Called by rclpy.on_shutdown.
        Cleans up the spawned actor.
        """
        self.get_logger().info('Node is shutting down, destroying ego actor...')

        if self.ego_actor:
            # Must disable autopilot before destroying
            if self.ego_actor.is_alive:
                self.ego_actor.set_autopilot(False)
                self.ego_actor.destroy()
            self.get_logger().info('Ego actor destroyed.')

def main(args=None):
    rclpy.init(args=args)
    
    # Use a try-finally block to ensure shutdown
    node = None
    try:
        node = CarlaClientNode()
        if rclpy.ok(): # Only spin if the node initialized correctly
            rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('KeyboardInterrupt received, shutting down node.')
            # We let the on_shutdown hook handle the cleanup
    except Exception as e:
        if node:
            node.get_logger().error(f"Spin failed with error: {e}")
    finally:
        if node:
            # Explicitly call on_shutdown just in case
            node.on_shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        
if __name__ == '__main__':
    main()