import glob
import os

from setuptools import find_packages, setup

package_name = "ros2_sam"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob.glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "data"),
            glob.glob("data/*.png") + glob.glob("data/*.jpg"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adrian Röfer, mhubii",
    maintainer_email="aroefer@cs.uni-freiburg.de, martin.huber@kcl.ac.uk",
    description="ROS 2 wrapper for Meta's Segment-Anything model.",
    license="GNU",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sam_client_node = scripts.sam_client_node:main",
            "sam_server_node = scripts.sam_server_node:main",
        ],
    },
)
