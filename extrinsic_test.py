import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def get_camera_extrinsic_coordinates(bag_path):
    """Extract camera extrinsic parameters from the ROS bag."""
    param_topic = "/rosparam_dump"
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[param_topic]):
            # Extract the JSON data from the message
            data = msg.data
            # Load JSON data
            sensor_data = json.loads(data)

    # Get camera pose information
    camera_pose = sensor_data["vehicle"]["sensors"]["poses"]["camera_front"]
    roll = camera_pose.get("roll")
    pitch = camera_pose.get("pitch")
    yaw = camera_pose.get("yaw")
    tx = camera_pose.get("x")
    ty = camera_pose.get("y")
    tz = camera_pose.get("z")
    return roll, pitch, yaw, tx, ty, tz

def get_extrinsic_matrix(roll, pitch, yaw, tx, ty, tz):
    """Generate the extrinsic transformation matrix from camera pose parameters."""
    r = (R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_matrix()
    # Transform the extrinsic matrix from FLU frame to RDF frame
    T = np.zeros((4, 4), np.double)
    T[0][0] = -r[0][1]
    T[0][1] = -r[1][1]
    T[0][2] = -r[2][1]
    T[0][3] = ty
    T[1][0] = -r[0][2]
    T[1][1] = -r[1][2]
    T[1][2] = -r[2][2]
    T[1][3] = tz
    T[2][0] = r[0][0]
    T[2][1] = r[1][0]
    T[2][2] = r[2][0]
    T[2][3] = -tx
    T[3][0] = 0
    T[3][1] = 0
    T[3][2] = 0
    T[3][3] = 1
    return T

def get_extrinsic_matrix2(roll, pitch, yaw, tx, ty, tz):
    """Generate the extrinsic transformation matrix from camera pose parameters."""
    r = (R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_matrix()
    flu_rdf=(R.from_euler("zyx", [-np.pi/2, np.pi/2 , 0], degrees=False)).as_matrix()
    r=  r @ flu_rdf
    # Transform the extrinsic matrix from FLU frame to RDF frame
    T = np.zeros((4, 4), np.double)
    t = np.zeros((3, 1))
    t[0][0] = tx
    t[1][0] = ty
    t[2][0] = tz
    r_inv = np.transpose(r)
    # Get world to vehicle extrinsic matrix
    T[:3, :3] = r_inv
    t_inv = -np.matmul(r_inv, t)
    T[0][3] = t_inv[0][0]
    T[1][3] = t_inv[1][0]
    T[2][3] = t_inv[2][0]
    T[3][3] = 1
    return T

bag_path="/home/rosuser/shared/extracted_images/2022-12-08.14-56-17.3.bag"
roll, pitch, yaw, tx, ty, tz=get_camera_extrinsic_coordinates(bag_path)
T=get_extrinsic_matrix(roll, pitch, yaw, tx, ty, tz)
T2=get_extrinsic_matrix2(roll, pitch, yaw, tx, ty, tz)
print(T, " \n ", T2)




        