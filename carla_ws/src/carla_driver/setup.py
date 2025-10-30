from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'carla_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, 'launch'), glob(os.path.join("launch", 
                                                                          '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', "open3d"],
    zip_safe=True,
    maintainer='master',
    maintainer_email='master@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_client_node = carla_driver.carla_client_node:main',
            'carla_other_vehicles = carla_driver.carla_other_vehicles:main',
            'test_client = carla_driver.test_files:main'
        ],
    },
)
