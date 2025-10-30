import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/master/Documents/Programming/ARI_CARLA/carla_ws/src/install/carla_driver'
