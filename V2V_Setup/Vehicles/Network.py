class VehicleNetwork:
    vehicles = []

    @classmethod
    def register(cls, vehicle):
        cls.vehicles.append(vehicle)

    @classmethod
    def unregister(cls, vehicle):
        if vehicle in cls.vehicles:
            cls.vehicles.remove(vehicle)

    @classmethod
    def broadcast(cls, signal_data, sender, range_limit=50.0):
        sender_location = sender.get_location()

        for vehicle in cls.vehicles:
            if vehicle != sender:
                distance = vehicle.get_location().distance(sender_location)
                if distance <= range_limit:
                    vehicle.receive_signal(signal_data, from_vehicle=sender)
