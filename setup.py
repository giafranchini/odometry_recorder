from setuptools import find_packages, setup
import os
from glob2 import glob

package_name = 'odometry_recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params/', '*'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch/', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='giacomo.franchini@polito.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odometry_recorder_node = odometry_recorder.odometry_recorder_node:main'
        ],
    },
)
