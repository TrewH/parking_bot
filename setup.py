from setuptools import setup, find_packages

package_name = 'parking_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@ucsd.edu',
    description='Parallel parking robot vision and control',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = parking_bot.vision_node:main',
            'test_parking_client = parking_bot.test_parking_client:main',
            "orchestrator = parking_bot.orchestrator:main",
        ],
    },
)
