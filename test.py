import carla
import numpy as np
import cv2
import time

# Initialize the CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world
world = client.get_world()

# Function to spawn a vehicle
def spawn_vehicle(blueprint, spawn_point):
    return world.spawn_actor(blueprint, spawn_point)

# Function to create a depth camera
def create_depth_camera(ego_vehicle):
    blueprint_library = world.get_blueprint_library()
    depth_camera_bp = blueprint_library.find('sensor.camera.depth')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))  # Adjust position and orientation as needed
    depth_camera = world.spawn_actor(depth_camera_bp, camera_transform, attach_to=ego_vehicle)
    return depth_camera

# Function to process the depth image
def depth_callback(depth_image):
    depth_array = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
    depth_array = np.reshape(depth_array, (depth_image.height, depth_image.width, 4))  # RGBA
    # Convert to meters (assuming the depth values are normalized)
    depth_meters = depth_array[:, :, 0] / 255.0 * 100.0  # Scale depth to meters (assuming max distance is 100m)
    return depth_meters

# Main function
def main():
    # Get the vehicle blueprints
    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    target_vehicle_bp = blueprint_library.find('vehicle.audi.a2')

    # Set the spawn points
    ego_spawn_point = carla.Transform(carla.Location(x=0, y=0, z=1))
    target_spawn_point = carla.Transform(carla.Location(x=10, y=0, z=1))  # 10 meters in front of ego vehicle

    # Spawn vehicles
    target_vehicle = spawn_vehicle(target_vehicle_bp, target_spawn_point)
    targetchange = carla.Transform(target_spawn_point.location + carla.Location(x=1))
    target_vehicle.set_transform(targetchange)
    ego_vehicle = spawn_vehicle(ego_vehicle_bp, ego_spawn_point)

    

    # Create the depth camera
    depth_camera = create_depth_camera(ego_vehicle)

    # Set up the callback for depth images
    depth_camera.listen(lambda image: depth_callback(image))

    # Allow some time for the camera to capture data
    print("Starting depth measurement...")
    time.sleep(5)  # Capture depth data for 5 seconds

    # Clean up
    depth_camera.stop()
    ego_vehicle.destroy()
    target_vehicle.destroy()
    depth_camera.destroy()

    # Close the client
    print("Test complete. Closing the CARLA client.")
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.camera.depth')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])

if __name__ == '__main__':
    main()
