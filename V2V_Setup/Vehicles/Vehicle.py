from .Network import VehicleNetwork


#TODO: Add Noise
#TODO: Add Packet Loss
#TODO: Add Encryption



class CustomVehicle:
    def __init__(self, world, blueprint, spawn_point):
        self.world = world
        self.vehicle = world.try_spawn_actor(blueprint, spawn_point)  # Composition: vehicle is a member variable
        self.sensors = []
        self.connected_vehicles = []

        VehicleNetwork.register(self)

    def apply_control(self, control):
        self.vehicle.apply_control(control)  # Delegate call to member vehicle

    def get_location(self):
        return self.vehicle.get_location()

    def attach_camera(self, camera_bp, transform, callback=None):
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        if callback:
            camera.listen(callback)
        self.sensors.append(camera)
        return camera
    def get_speed(self):
        return self.vehicle.get_speed()

    def destroy(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        self.vehicle.destroy()

    def send_signal(self):
        print("BroadCasting: Speed")
        VehicleNetwork.broadcast({"Speed": self.get_speed}, sender=self)

    def recieve_signal():
        pass
