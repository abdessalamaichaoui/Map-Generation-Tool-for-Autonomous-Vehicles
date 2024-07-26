import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import json




def cart_to_hom(X):
    """Convert Cartesian coordinates to homogeneous coordinates"""
    shape = X.shape
    X_h = np.ones((shape[0] + 1, shape[-1]), dtype=X.dtype)
    X_h[:-1, ...] = X
    return X_h

def rot_roll_error(roll, delta_roll=0):
    delta_rot_roll=delta_roll*np.array([[0, 0, 0],[0, -np.sin(roll), -np.cos(roll)],[0, np.cos(roll), -np.sin(roll)]])
    return delta_rot_roll

def rot_pitch_error(pitch, delta_pitch=0):
    delta_rot_pitch=delta_pitch*np.array([[-np.sin(pitch), 0, np.cos(pitch)],[0, 0, 0],[-np.cos(pitch), 0, -np.sin(pitch)]])
    return delta_rot_pitch


def rot_yaw_error(yaw, delta_yaw=0):
    delta_rot_yaw=delta_yaw*np.array([[-np.sin(yaw), -np.cos(yaw), 0],[np.cos(yaw), -np.sin(yaw), 0],[0, 0, 0]])
    return delta_rot_yaw

def compute_delta_rot(roll, pitch, yaw, delta_R_x, delta_R_y, delta_R_z):
    R_x = (R.from_euler("zyx", [0, 0, roll], degrees=False)).as_matrix()
    R_y = (R.from_euler("zyx", [0, pitch, 0], degrees=False)).as_matrix()
    R_z = (R.from_euler("zyx", [yaw, 0, 0], degrees=False)).as_matrix()
    term_1 = R_z @ R_y @ delta_R_x
    term_2 = R_z @ delta_R_y @ R_x
    term_3 = R_z @ delta_R_y @ delta_R_x
    term_4 = delta_R_z @ R_y @ R_x
    term_5 = delta_R_z @ R_y @ delta_R_x
    term_6 = delta_R_z @ delta_R_y @ R_x
    term_7 = delta_R_z @ delta_R_y @ delta_R_x
    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7


def compute_delta_E(delta_rot, delta_tran, rot, tran):
    delta_E = np.zeros((4, 4), np.double)
    flu_rdf=(R.from_euler("zyx", [-np.pi/2, np.pi/2 , 0], degrees=False)).as_matrix()
    rot_inv=np.transpose(rot)
    delta_rot=  delta_rot @ flu_rdf
    delta_rot_inv = np.transpose(delta_rot)
    delta_tran_inv=-np.matmul(delta_rot_inv,delta_tran)-np.matmul(delta_rot_inv,tran)-np.matmul(rot_inv,delta_tran)
    delta_E[:3, :3] = delta_rot_inv
    delta_E[0][3] = delta_tran_inv[0][0]
    delta_E[1][3] = delta_tran_inv[1][0]
    delta_E[2][3] = delta_tran_inv[2][0]
    delta_E[3][3] = 1
    return delta_E

def compute_delta_V(delta_rot, delta_tran):
    delta_V = np.zeros((4, 4), np.double)
    delta_rot_inv = np.transpose(delta_rot)
    delta_tran_inv=-np.matmul(delta_rot_inv,delta_tran)
    delta_V[:3, :3] = delta_rot_inv
    delta_V[0][3] = delta_tran_inv[0][0]
    delta_V[1][3] = delta_tran_inv[1][0]
    delta_V[2][3] = delta_tran_inv[2][0]
    delta_V[3][3] = 1
    return delta_V


    


class Homography_Calculator:
    def __init__(self):
        """
        Constructor
        """

        self.scale_matrix = None
        self.image_width = None
        self.image_height = None
        self.intrinsic = None
        self.extrinsic = None
        self.extrinsic_w_v=None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.blur = None

    def get_camera_info(self):
        """Extract camera intrinsic parameters from a ROS bag."""
        # camera informations topic
        info = "/camera_front/camera_info"
        # read ROS bag
        bag_path="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-26.09-54-17.1.bag"
        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[info]):
                # read camera infos topic
                info_msg = msg

                self.intrinsic = np.array(info_msg.K).reshape([3, 3])
                self.distortion = np.array(info_msg.D)
                self.image_width = info_msg.width
                self.image_height = info_msg.height
        # return self.intrinsic
    
    def get_camera_extrinsic_coordinates(self):
        """Extract camera extrinsic parameters from the ROS bag."""
        param_topic = "/rosparam_dump"
        bag_path="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-26.09-54-17.1.bag"
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

    def get_extrinsic_matrix(self, roll, pitch, yaw, tx, ty, tz):
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
        self.extrinsic = T
        return T

    def get_extrinsic_world_to_vehicle(self, x, y, rotx, roty, rotz, w):
        """Generate the extrinsic matrix from world to vehicle frame."""
        # Define the translation vector of vehicle to world frame origin
        t = np.zeros((3, 1))
        t[0][0] = x
        t[1][0] = y
        t[2][0] = 0
        # Get Rotation matrix from Quaternions

        r = R.from_quat([rotx, roty, rotz, w])
        r = r.as_matrix()
        # Inverse rotation matrix
        r_inv = np.transpose(r)
        extrinsic_w_v = np.zeros((4, 4))
        # Get world to vehicle extrinsic matrix
        extrinsic_w_v[:3, :3] = r_inv
        t_inv = -np.matmul(r_inv, t)
        extrinsic_w_v[0][3] = t_inv[0][0]
        extrinsic_w_v[1][3] = t_inv[1][0]
        extrinsic_w_v[2][3] = t_inv[2][0]
        extrinsic_w_v[3][3] = 1
        return extrinsic_w_v

    def set_scale_matrix(self, Delta_x, Delta_y):
        self.scale_matrix = np.array(
            [[Delta_x, 0, self.x_min], [0, -Delta_y, self.y_max], [0, 0, 0], [0, 0, 1]]
        )
        
    def get_map_to_camera_homography_matrix(self, extrinsic, extrinsic_w_v):
        """Compute the homography matrix for mapping from world to camera frame."""
        return self.intrinsic @ extrinsic[:3, :] @ extrinsic_w_v @ self.scale_matrix
    
    def set_map_limits(self):
        """Find the map limits by extracting vehicle positions from a ROS bag."""
        # define two empty arrays
        loc_topic = "/vehicle_pose"
        x = []
        y = []
        # read the ros bag and recover the vehicle's trajectory
        bag_path="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-26.09-54-17.1.bag"
        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[loc_topic]):
                x.append(msg.pose.pose.position.x)
                y.append(msg.pose.pose.position.y)
            # read the ros bag and recover the vehicle's trajectory

        # get the limits of the vehicle's trajectory
        self.x_min = np.min(x) - 20  # north east point x coordinate
        self.y_max = np.max(y) + 20  # north east point y coordinate
        self.x_max = np.max(x) + 20  # we subtract/add 20 to avoid projection getting outside of map
        self.y_min = np.min(y) - 20

    def find_blur_starting_line(self, Delta):
        """Determine the starting line of image blur based on the scale factor value."""
        # define two grids for each pixel coordinate
        u, v = np.meshgrid(
            np.linspace(0, self.image_height - 1, self.image_height),
            np.linspace(0, self.image_width - 1, self.image_width),
        )
        # make a matrix containing image pixels coordiantes
        x = np.zeros((2, self.image_height * self.image_width), np.double)
        x[0, ...] = u.flatten()
        x[1, ...] = v.flatten()
        # get the reverse projection of each image point on the ground plane (z=0)
        ground_points = Homography_Calculator.get_ground_coords(self, x)
        # compute the distance of each ground point to the vehicle
        distances = np.sqrt(
            np.add(
                np.multiply(ground_points[0, :], ground_points[0, :]),
                np.multiply(ground_points[1, :], ground_points[1, :]),
            )
        )
        distances = np.reshape(distances, u.shape)
        distances_diff = np.zeros(self.image_width)
        y_axis = int(self.intrinsic[0][2])
        # compute the distance between each two consecutive pixels in the forward direction
        for i in range(self.image_width - 1, 0, -1):
            distances_diff[i] = distances[i, y_axis] - distances[i - 1, y_axis]
            # the blur limit is where the distance between two consecutive points
            # is above the scale factor
            if np.abs(distances_diff[i]) > Delta:
                blur = i
                break
        self.blur = blur

    def get_ground_coords(self, point):
        """Get ground coordinates from image coordinates using the transformation matrix."""
        # Retrieve the rotation matrix and translation vectors
        R = self.extrinsic[:3, :3]
        tvec = self.extrinsic[:3, 3]
        # Inverse(transpose) the rotation
        R_inv = np.transpose(R)
        # Retrieve the reverse projection translation
        t = np.matmul(R_inv, tvec)
        # Move from Cartesian coordinates to homogeneous coordinates
        uv_coordinates = cart_to_hom(point)
        # Computing the reverse projection matrix
        X_h = np.matmul(R_inv, np.linalg.solve(self.intrinsic, uv_coordinates))
        # Finding the scale factor (homogeneous coordinates weight w) for Z=0
        w = np.divide(t[2], X_h[2, ...])
        ground_points = np.zeros((X_h.shape[0], X_h.shape[-1]))
        # Calculating the 3D points of image points
        ground_points[0, ...] = np.subtract(np.multiply(X_h[0, ...], w), t[0])
        ground_points[1, ...] = np.subtract(np.multiply(X_h[1, ...], w), t[1])
        return ground_points
    
    def delta_H(self, delta_K, delta_E, delta_V):
        term_1 = self.intrinsic @ self.extrinsic[:3, :] @ delta_V
        term_2 = self.intrinsic @ delta_E[:3, :] @ self.extrinsic_w_v
        term_3 = self.intrinsic @ delta_E[:3, :] @ delta_V
        term_4 = delta_K @ self.extrinsic[:3, :] @ self.extrinsic_w_v
        term_5 = delta_K @ self.extrinsic[:3, :] @ delta_V
        term_6 = delta_K @ delta_E[:3, :] @ self.extrinsic_w_v
        term_7 = delta_K @ delta_E[:3, :] @ delta_V
        delta_H= (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7) @ self.scale_matrix
        return delta_H

    
def main():
    loc_topic = "/vehicle_pose"
    HC=Homography_Calculator()
    Delta_y = 0.03
    Delta_x = 0.03
    Delta = 0.03
    HC.get_camera_info()
    roll, pitch, yaw, tx, ty, tz = HC.get_camera_extrinsic_coordinates()
    rot_pitch1= (R.from_euler("zyx", [0, pitch, 0], degrees=False)).as_matrix()
    rot_pitch2= (R.from_euler("zyx", [0, pitch+0.1, 0], degrees=False)).as_matrix()


    tran=np.array([[tx],[ty],[tz]])
    rot=(R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_matrix()
    rot2=(R.from_euler("zyx", [yaw+0.1, pitch, roll], degrees=False)).as_matrix()
    # print(rot2-rot)
    extrinsic1=HC.get_extrinsic_matrix(roll, pitch, yaw, tx, ty, tz)
    print(np.shape(extrinsic1))
    extrinsic2=HC.get_extrinsic_matrix(roll, pitch, yaw+0.1, tx, ty, tz)
    HC.set_map_limits()
    HC.set_scale_matrix(Delta_x, Delta_y)
    HC.find_blur_starting_line(Delta)
    bag_path="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-26.09-54-17.1.bag"
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[loc_topic]): 
            x=msg.pose.pose.position.x
            y=msg.pose.pose.position.y
            z=msg.pose.pose.position.z
            rotx=msg.pose.pose.orientation.x
            roty=msg.pose.pose.orientation.y
            rotz=msg.pose.pose.orientation.z
            w=msg.pose.pose.orientation.w   
            extrinsic_w_v=HC.get_extrinsic_world_to_vehicle(x, y, rotx, roty, rotz, w)
            H1=HC.get_map_to_camera_homography_matrix(extrinsic1, extrinsic_w_v)
            H2=HC.get_map_to_camera_homography_matrix(extrinsic2, extrinsic_w_v)
            #camera angles errors
            cam_delta_roll=0.1
            cam_delta_pitch=0.1
            cam_delta_yaw=0.1
            #camera position error
            delta_tran=np.zeros((3, 1))         #np.array([[0],[0],[0]])
            #compute camera rotation error based on angle errors
            delta_rot_roll=rot_roll_error(roll, cam_delta_roll)
            print(rot_pitch2-rot_pitch1)
            delta_rot_pitch=rot_yaw_error(pitch, cam_delta_pitch)
            print(delta_rot_pitch)

            delta_rot_yaw=rot_pitch_error(yaw, cam_delta_yaw)
            rot_error=compute_delta_rot(roll, pitch, yaw, delta_rot_roll, delta_rot_pitch, delta_rot_yaw)
            # print(rot_error)
            #compute extrinsic matrix error based on angle and position errors
            delta_V=np.zeros((4,4))
            delta_K=np.zeros((3,3))
            delta_E=compute_delta_E(rot_error, delta_tran, rot, tran)
            # print(delta_E)
            # print(extrinsic2-extrinsic1)
            delta_H=HC.delta_H(delta_K, delta_E, delta_V)
            print(delta_H)
            print(H2-H1)
if __name__ == '__main__':
    main()

    




