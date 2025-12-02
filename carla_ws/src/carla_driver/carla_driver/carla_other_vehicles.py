import carla
import rclpy
from rclpy.node import Node
import random
import time
import sys


class CyclistControllerNode(Node):
    """
    A ROS2 node that spawns a cyclist in CARLA and sets it to
    autopilot using the shared Traffic Manager.
    """
    def __init__(self):
        super().__init__('cyclist_controller_node')
        self.get_logger().info('Cyclist Controller Node started.')
        
        self.client = None
        self.world = None
        self.cyclist_actor = None
        
        try:
            # 1. Connect to CARLA
            self.get_logger().info('Connecting to CARLA server at localhost:2000...')
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            
            # 2. Spawn a cyclist
            self.spawn_cyclist()

            if self.cyclist_actor:
                # --- FIX: Get the Traffic Manager and set the port ---
                self.get_logger().info('Setting cyclist to autopilot...')
                tm = self.client.get_trafficmanager(8000) # Get TM on default port
                tm_port = tm.get_port()
                self.cyclist_actor.set_autopilot(True, tm_port) # Tell autopilot to use this TM
                self.get_logger().info(f'Cyclist using Traffic Manager on port {tm_port}')
                # --- End Fix ---
                
                # 4. Register a shutdown hook to clean up the actor
                rclpy.get_default_context().on_shutdown(self.on_shutdown)
            else:
                self.get_logger().error("Failed to spawn cyclist. Shutting down.")
                rclpy.shutdown()


        except RuntimeError as e:
            self.get_logger().error(f'Failed to connect to CARLA server: {e}')
            self.get_logger().error('Is the CARLA server running? (e.g., ./CarlaUE4.sh --ros2)')
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'An unexpected error occurred: {e}')
            rclpy.shutdown()

    def spawn_cyclist(self):
        self.get_logger().info('Spawning a cyclist...')
        blueprint_library = self.world.get_blueprint_library()
        
        # Find the bicycle blueprint
        try:
            cyclist_bp = blueprint_library.find('vehicle.bicycle.normal')
        except IndexError:
            self.get_logger().error("Could not find 'vehicle.bicycle.normal' blueprint.")
            self.get_logger().error("Trying 'vehicle.diamondback.century'...")
            try:
                cyclist_bp = blueprint_library.find('vehicle.diamondback.century')
            except IndexError:
                self.get_logger().error("Could not find any bicycle blueprint. Exiting.")
                return

        # Find a suitable spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            self.get_logger().error('No spawn points found in the current map!')
            return
            
        spawn_point = random.choice(spawn_points)
        
        # Spawn the actor
        self.cyclist_actor = self.world.try_spawn_actor(cyclist_bp, spawn_point)
        
        if self.cyclist_actor:
            self.get_logger().info(f'Cyclist spawned with ID: {self.cyclist_actor.id}')
            # Set a short delay to let the actor settle
            time.sleep(0.5)
        else:
            self.get_logger().error('Failed to spawn cyclist.')


    def on_shutdown(self):
        """
        Called by rclpy.on_shutdown.
        Cleans up the spawned actor.
        """
        self.get_logger().info('Node is shutting down, destroying cyclist actor...')
        if self.cyclist_actor:
            self.cyclist_actor.destroy()
            self.get_logger().info('Cyclist actor destroyed.')


def main(args=None):
    rclpy.init(args=args)
    
    # Use a try-finally block to ensure shutdown
    node = None
    try:
        node = CyclistControllerNode()
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
