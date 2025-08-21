import carla
from Vehicles import CustomVehicle
import time

client = carla.Client('localhost', 2000)

world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('*mini*')[0]
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[2]
print(spawn_point)
EgoVehicle = CustomVehicle(world, vehicle_bp, spawn_point)
EgoVehicle.send_signal()

time.sleep(500)

EgoVehicle.destroy()