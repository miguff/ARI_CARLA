import carla
import rclpy
from rclpy.node import Node
import random
import time
import sys

# --- ADD IMPORTS FOR MAP PUBLISHING ---
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math


class CarlaClientNode(Node):

    def __init__(self):
        super().__init__("carla_client_node")
        self.get_logger().info("CARLA Client Node started.")
        
        # Initialize class members
        self.client = None
        self.world = None
        self.ego_actor = None
        self.sensor_list = []
        # --- ADD THIS ---
        # Controls the distance between map points (in meters)
        # Smaller = smoother curves, but more markers
        self.map_granularity = 2.0 
        self.get_logger().info(f"Using map granularity: {self.map_granularity}m")
        # --- END ADD ---
        self.get_logger().info("Connect to carla.")
        # --- ADD MAP PUBLISHER ---
        self.map_publisher = self.create_publisher(MarkerArray, '/carla/map', 1)
        ...

        # --- ADD DYNAMIC TF BROADCASTER INSTEAD ---
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create TF broadcaster

    
        self.connect_to_carla()
        
    
    def publish_ego_tf(self):
        """Publishes the ego vehicle's transform."""
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
        


    def connect_to_carla(self):
        try:
            # Try to connect to the CARLA server
            self.get_logger().info('Attempting to connect to CARLA server at localhost:2000...')
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)  # 5-second timeout
            
            self.world = self.client.get_world()
            server_version = self.client.get_server_version()
            
            self.get_logger().info(f'Successfully connected to CARLA server!')
            self.get_logger().info(f'  Server Version: {server_version}')
            self.get_logger().info(f'  Current Map: {self.world.get_map().name}')

            # --- PUBLISH MAP ONCE ---
            self.get_logger().info('Publishing static map topology...')
            self.publish_map()
            # --- END ---
            # --- Spawn and set autopilot *after* successful connection ---
            self.spawn_ego()
            
            if self.ego_actor:
                # --- FIX: Get the Traffic Manager and set the port ---
                self.get_logger().info('Setting ego vehicle to autopilot...')
                self.timer_period = 0.1
                
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


        location = self.ego_actor.get_location()
        velocity = self.ego_actor.get_velocity()
        
        # --- ADD THIS CALL ---
        if self.ego_actor:
            self.publish_ego_tf()
        # --- END ADD ---
        #self.get_logger().info(f"This is my location: {location}")
        #self.get_logger().info(f"This is my velocity: {velocity}")

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
            marker.scale.x = 0.3  # line width (made slightly thicker)

            # Set color (e.g., white)
            marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            
            # --- START MODIFICATION ---
            # Instead of just 2 points, get all points in between
            
            marker.points = []
            
            # Add the first waypoint
            loc1 = w1.transform.location
            marker.points.append(Point(x=float(loc1.x), y=float(-loc1.y), z=float(loc1.z)))
            
            # Get all waypoints from w1 until the end of the lane segment
            # using the specified granularity
            waypoint_list = w1.next_until_lane_end(self.map_granularity)

            for w in waypoint_list:
                loc = w.transform.location
                marker.points.append(Point(x=float(loc.x), y=float(-loc.y), z=float(loc.z)))

            # Add the final waypoint (w2)
            loc2 = w2.transform.location
            marker.points.append(Point(x=float(loc2.x), y=float(-loc2.y), z=float(loc2.z)))
            
            # --- END MODIFICATION ---
            
            marker_array.markers.append(marker)
            
        self.map_publisher.publish(marker_array)
        self.get_logger().info(f"Published {id_counter} map segments to /carla/map")
    def spawn_ego(self):
        self.get_logger().info('Spawning a ego...')
        blueprint_library = self.world.get_blueprint_library()

        ego_bp = blueprint_library.find('vehicle.dodge.charger_2020')

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
             self.get_logger().error("No spawn points found in map!")
             return

        spawn_point = random.choice(spawn_points)
        # Spawn the actor
        self.ego_actor = self.world.try_spawn_actor(ego_bp, spawn_point)

        if self.ego_actor:
            self.get_logger().info(f'Ego spawned with ID: {self.ego_actor.id}')
            self.get_logger().info(f'Ego spawned with ID: {self.ego_actor.id}')
            
            time.sleep(0.5)

    def on_shutdown(self):
        """
        Called by rclpy.on_shutdown.
        Cleans up the spawned actor.
        """
        self.get_logger().info('Node is shutting down, destroying ego actor...')

        if self.ego_actor:
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
            node.on_shutdown()
    except Exception as e:
        if node:
            node.get_logger().error(f"Spin failed with error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()
            
if __name__ == '__main__':
    main()
