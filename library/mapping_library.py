#!/usr/bin/env python3
"""
    2D grid map functions library

"""
import rosbag
import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct
import os
from scipy.spatial.transform import Rotation as R
import json
import cv2 as cv
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from math import floor

# Load the video encoder shared library
video_encoder = npct.load_library(
    "libvideo_encoder_py", "/home/rosuser/build/Executables/ROS/blackbox/video_encoder/"
)

# Define buffer type and size type for ctypes
buffer_type = npct.ndpointer(dtype=np.uint8, ndim=1, flags=["C_CONTIGUOUS"])
SizeType = ct.c_ulonglong

# Specify argument types for the to_rgb24 function in the video encoder library
video_encoder.to_rgb24.argtypes = [
    ct.c_char_p,
    buffer_type,
    SizeType,
    ct.POINTER(ct.POINTER(ct.c_uint8)),
    ct.POINTER(SizeType),
    ct.POINTER(SizeType),
]


def cart_to_hom(X):
    """Convert Cartesian coordinates to homogeneous coordinates"""
    shape = X.shape
    X_h = np.ones((shape[0] + 1, shape[-1]), dtype=X.dtype)
    X_h[:-1, ...] = X
    return X_h


def hom_to_cart(X):
    """Convert homogeneous coordinates to Cartesian coordinates."""
    return X[:-1, ...] / X[-1, ...]


def map_to_camera(H, X):
    """Map coordinates from the 2d grid map to camera image coordinates."""
    return hom_to_cart(H @ cart_to_hom(X))


def camera_to_map(H, X):
    """Map coordinates from camera image coordinates to the 2d grid map."""
    return hom_to_cart(np.linalg.solve(H, cart_to_hom(X)))


def map_to_vehicle(M, X):
    """Map coordinates from the 2d grid map to vehicle image coordinates."""
    return hom_to_cart(M @ cart_to_hom(X))


def calculate_distance(loc1, loc2):
    """
    Calculate the Euclidean distance between two localization points.
    """
    return np.sqrt(np.square(loc2["x"] - loc1["x"]) + np.square(loc2["y"] - loc1["y"]))


def write_image_on_disk(path, image):
    """
    Write an image double in 0 to 1 scale BGR image encoding on disk

    """
    cv.imwrite(path, image, [cv.IMWRITE_JPEG_QUALITY, 100])


def get_distances(M, flat_map_x_coord, flat_map_y_coord):
    """
    calculate the Euclidean distances (squared Euclidean distance) of map coordinates
    from the vehicle.
    Parameters:
    - M: The transformation matrix from map to vehicle coordinates.
    - flat_map_x_coord: Flattened array of x-coordinates in the map.
    - flat_map_y_coord: Flattened array of y-coordinates in the map.

    Returns:
    - distances: an array containing the Euclidean distances of each point from the vehicle.
    """

    # Stack the flat map coordinates into a single matrix
    coords = np.vstack((flat_map_y_coord, flat_map_x_coord))

    # Transform the map coordinates to vehicle coordinates
    ground_coords = map_to_vehicle(M, coords)

    # Compute the squared Euclidean distance for each coordinate
    distances = np.add(
        np.multiply(ground_coords[0, :], ground_coords[0, :]),
        np.multiply(ground_coords[1, :], ground_coords[1, :]),
    )

    # Return the distances
    return distances
def image_illumination_normalization(image):
    """
    Apply image illumination normalization to the input image and return the result

    """
    # compute image histogram 
    hist, _ = np.histogram(
        (image).flatten(), bins=int(255 * np.max(image)), range=[0, np.max(image)]
    )
    # compute image cumulative histogram 
    cum_hist = np.cumsum(hist)
    cumulative_hist_norm = cum_hist / np.sum(hist)
    # compute the histogram quantiles
    q1 = np.argmax(cumulative_hist_norm >= 0.01) / 255    
    #the quantiles can be modified depending on the image illumination
    q3 = np.argmax(cumulative_hist_norm >= 0.9) / 255
    return (image - q1) / (q3 - q1)
def road_segmentation(source,frame):
    masks_dir="/home/rosuser/shared/iamap/masks"
    model = FastSAM("FastSAM.pt")  # or FastSAM-x.pt
    # Run inference on an image
    everything_results = model(source, device="cuda", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(source, everything_results, device="cuda")
    # results = prompt_process.point_prompt(points=[[500, 700]], pointlabel=[1])
    results=prompt_process.text_prompt(text="a photo of the road")
    masks = results[0].masks # The actual mask data, a tensor of shape (N, H, W)
    mask_data = masks.data 
    for i, mask in enumerate(mask_data):
        mask = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_file = os.path.join(masks_dir, f"mask_{i}{frame}.png")
        cv.imwrite(mask_file, mask)
        return mask

class MapGenerator:
    """
    Generate a BEV Map from ROS Bag Data
    """

    def __init__(self,args):
        """
        Constructor
        """
        self.bag_path = args.bag_file
        self.image_save_folder=args.images_save_dir
        self.map_folder=args.map_save_dir
        self.scale_matrix = None
        self.image_width = None
        self.image_height = None
        self.intrinsic = None
        self.extrinsic = None
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
        with rosbag.Bag(self.bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[info]):
                # read camera infos topic
                info_msg = msg

                self.intrinsic = np.array(info_msg.K).reshape([3, 3])
                self.distortion = np.array(info_msg.D)
                self.image_width = info_msg.width
                self.image_height = info_msg.height

    def get_camera_extrinsic_coordinates(self):
        """Extract camera extrinsic parameters from the ROS bag."""
        param_topic = "/rosparam_dump"
        with rosbag.Bag(self.bag_path, "r") as bag:
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

    # def get_extrinsic_matrix(self, roll, pitch, yaw, tx, ty, tz):
    #     """Generate the extrinsic transformation matrix from camera pose parameters."""
    #     r = (R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_matrix()
    #     # Transform the extrinsic matrix from FLU frame to RDF frame
    #     T = np.zeros((4, 4), np.double)
    #     T[0][0] = -r[0][1]
    #     T[0][1] = -r[1][1]
    #     T[0][2] = -r[2][1]
    #     T[0][3] = ty
    #     T[1][0] = -r[0][2]
    #     T[1][1] = -r[1][2]
    #     T[1][2] = -r[2][2]
    #     T[1][3] = tz
    #     T[2][0] = r[0][0]
    #     T[2][1] = r[1][0]
    #     T[2][2] = r[2][0]
    #     T[2][3] = -tx
    #     T[3][0] = 0
    #     T[3][1] = 0
    #     T[3][2] = 0
    #     T[3][3] = 1
    #     self.extrinsic = T
    
    def get_extrinsic_matrix(self, roll, pitch, yaw, tx, ty, tz):
        """Generate the extrinsic transformation matrix from camera pose parameters."""
        r = (R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_matrix()
        # Transform the extrinsic matrix from FLU frame to RDF frame
        flu_rdf=(R.from_euler("zyx", [-np.pi/2, np.pi/2 , 0], degrees=False)).as_matrix()
        r=  r @ flu_rdf
        extrinsic = np.zeros((4, 4), np.double)
        t = np.zeros((3, 1))
        t[0][0] = tx
        t[1][0] = ty
        t[2][0] = tz
        r_inv = np.transpose(r)
        extrinsic[:3, :3] = r_inv
        t_inv = -np.matmul(r_inv, t)
        extrinsic[0][3] = t_inv[0][0]
        extrinsic[1][3] = t_inv[1][0]
        extrinsic[2][3] = t_inv[2][0]
        extrinsic[3][3] = 1
        self.extrinsic = extrinsic

    def set_map_limits(self):
        """Find the map limits by extracting vehicle positions from a ROS bag."""
        # define two empty arrays
        loc_topic = "/vehicle_pose"
        x = []
        y = []
        # read the ros bag and recover the vehicle's trajectory
        bag_path_2="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-28.06-49-30.4.bag"
        bag_path_3="/home/rosuser/shared/VJRD1A10230000060.2024-05-26.04-12-44.4.bag"

        with rosbag.Bag(self.bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[loc_topic]):
                x.append(msg.pose.pose.position.x)
                y.append(msg.pose.pose.position.y)
            # read the ros bag and recover the vehicle's trajectory
        # with rosbag.Bag(bag_path_2, "r") as bag:
        #     for _, msg, _ in bag.read_messages(topics=[loc_topic]):
        #         x.append(msg.pose.pose.position.x)
        #         y.append(msg.pose.pose.position.y)
        #     # read the ros bag and recover the vehicle's trajectory
        # with rosbag.Bag(bag_path_3, "r") as bag:
        #     for _, msg, _ in bag.read_messages(topics=[loc_topic]):
        #         x.append(msg.pose.pose.position.x)
        #         y.append(msg.pose.pose.position.y)
            # read the ros bag and recover the vehicle's trajectory
        # get the limits of the vehicle's trajectory
        self.x_min = np.min(x) - 20  # north east point x coordinate
        self.y_max = np.max(y) + 20  # north east point y coordinate
        self.x_max = np.max(x) + 20  # we subtract/add 20 to avoid projection getting outside of map
        self.y_min = np.min(y) - 20

    def set_scale_matrix(self, Delta_x, Delta_y):
        self.scale_matrix = np.array(
            [[Delta_x, 0, self.x_min], [0, -Delta_y, self.y_max], [0, 0, 0], [0, 0, 1]]
        )

    def image_distortion_correction(self, image):
        """
        Apply image distortion correction to the input image.
        """
        mapx, mapy = cv.initUndistortRectifyMap(
            self.intrinsic,
            self.distortion,
            None,  # R matrix
            self.intrinsic,  # New camera matrix
            (self.image_width, self.image_height),
            cv.CV_32FC1,
        )
        return cv.remap(image, mapx, mapy, cv.INTER_LINEAR, cv.INTER_LINEAR, borderValue=(0, 0, 0))

    def get_extrinsic_world_to_vehicle(self, x, y, z, rotx, roty, rotz, w):
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

    def get_map_to_camera_homography_matrix(self, extrinsic_w_v):
        """Compute the homography matrix for mapping from world to camera frame."""
        return self.intrinsic @ self.extrinsic[:3, :] @ extrinsic_w_v @ self.scale_matrix

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
        ground_points = MapGenerator.get_ground_coords(self, x)
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

    def image_vignetting_correction(self, image):
        """
        Apply image vignetting correction to the input image and return the result

        """
        # Create meshgrid with pixel indexes
        u, v = np.meshgrid(np.arange(self.image_width), np.arange(self.image_height))

        # Normalized coordinates calculation
        coordinates_normalized = np.linalg.solve(
            self.intrinsic, np.stack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))
        )
        coordinates_normalized /= coordinates_normalized[2]
        x_norm = (
            coordinates_normalized[0].reshape(self.image_height, self.image_width).astype(np.double)
        )
        y_norm = (
            coordinates_normalized[1].reshape(self.image_height, self.image_width).astype(np.double)
        )

        # Radius calculation
        r = np.sqrt(np.square(x_norm) + np.square(y_norm))

        # Angle calculation
        alpha = np.arctan(r)
        # Vignetting mask computation
        correction_mask = np.power(np.cos(alpha), 4).astype(np.double)
        # applying vignetting correction mask and return resutl
        return np.divide(image, np.tile(correction_mask[:, :, np.newaxis], (1, 1, 3)))


    def save_images_from_bag2(self,args,bag_path_2):
        """
        Reads a ROS bag file, extracts images from a specific topic,
        and saves them to disk with filenames as their timestamps.

        """
        image_topic = "/camera_front/image/compressed"
        # Ensure save folder exists
        if not os.path.exists(self.image_save_folder):
            os.makedirs(self.image_save_folder)

        with rosbag.Bag(bag_path_2, "r") as bag:
            for topic, msg, _ in bag.read_messages(topics=[image_topic]):
                if topic == image_topic:
                    # Extract image data from the ROS bag message
                    image_stamp = msg.header.stamp.to_sec()

                    input_stream = np.frombuffer(msg.data, np.uint8)
                    buffer = ct.pointer(ct.c_uint8())
                    width = SizeType()
                    height = SizeType()

                    video_encoder.to_rgb24(
                        msg.format.encode("utf-8"),
                        input_stream,
                        input_stream.size,
                        buffer,
                        width,
                        height,
                    )
                    image = npct.as_array(buffer, shape=(height.value, width.value, 3))

                    # correct image from BGR format to RGB
                    extracted_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                    temp_image = MapGenerator.image_distortion_correction(
                        self, extracted_image
                    ).astype(np.double)
                    # correct vignetting if needed
                    if args.vignetting:
                        temp_image = MapGenerator.image_vignetting_correction(
                            self,
                            extracted_image,
                        ).astype(np.double)
                    # normalize image illumination if needed
                    if args.normalization:
                        temp_image= image_illumination_normalization(temp_image)*255
                    # Save the image with the timestamp as filename
                    img_filename = f"{image_stamp:.6f}.jpg"
                    img_path = os.path.join(self.image_save_folder, img_filename)
                    write_image_on_disk(img_path, temp_image)







    def save_images_from_bag(self,args):
        """
        Reads a ROS bag file, extracts images from a specific topic,
        and saves them to disk with filenames as their timestamps.

        """
        image_topic = "/camera_front/image/compressed"
        # Ensure save folder exists
        if not os.path.exists(self.image_save_folder):
            os.makedirs(self.image_save_folder)

        with rosbag.Bag(self.bag_path, "r") as bag:
            for topic, msg, _ in bag.read_messages(topics=[image_topic]):
                if topic == image_topic:
                    # Extract image data from the ROS bag message
                    image_stamp = msg.header.stamp.to_sec()

                    input_stream = np.frombuffer(msg.data, np.uint8)
                    buffer = ct.pointer(ct.c_uint8())
                    width = SizeType()
                    height = SizeType()

                    video_encoder.to_rgb24(
                        msg.format.encode("utf-8"),
                        input_stream,
                        input_stream.size,
                        buffer,
                        width,
                        height,
                    )
                    image = npct.as_array(buffer, shape=(height.value, width.value, 3))

                    # correct image from BGR format to RGB
                    extracted_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                    temp_image = MapGenerator.image_distortion_correction(
                        self, extracted_image
                    ).astype(np.double)
                    # correct vignetting if needed
                    if args.vignetting:
                        temp_image = MapGenerator.image_vignetting_correction(
                            self,
                            extracted_image,
                        ).astype(np.double)
                    # normalize image illumination if needed
                    if args.normalization:
                        temp_image= image_illumination_normalization(temp_image)*255
                    # Save the image with the timestamp as filename
                    img_filename = f"{image_stamp:.6f}.jpg"
                    img_path = os.path.join(self.image_save_folder, img_filename)
                    write_image_on_disk(img_path, temp_image)

    def extract_loc_data(self):
        """
        Extracts localization data from a ROS bag file.

        """
        loc_topic = "/vehicle_pose"
        localization_data = []
        # bag_path_2="/home/rosuser/shared/extracted_images/VJRD1A10230000074.2024-05-28.06-49-30.4.bag"
        # bag_path_3="/home/rosuser/shared/VJRD1A10230000060.2024-05-26.04-12-44.4.bag"

        with rosbag.Bag(self.bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[loc_topic]):
                loc_stamp = msg.header.stamp.to_sec()
                loc_info = {
                    "stamp": loc_stamp,
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z,
                    "rot_x": msg.pose.pose.orientation.x,
                    "rot_y": msg.pose.pose.orientation.y,
                    "rot_z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w,
                }
                localization_data.append(loc_info)

        # with rosbag.Bag(bag_path_2, "r") as bag:
        #     for _, msg, _ in bag.read_messages(topics=[loc_topic]):
        #         loc_stamp = msg.header.stamp.to_sec()
        #         loc_info = {
        #             "stamp": loc_stamp,
        #             "x": msg.pose.pose.position.x,
        #             "y": msg.pose.pose.position.y,
        #             "z": msg.pose.pose.position.z,
        #             "rot_x": msg.pose.pose.orientation.x,
        #             "rot_y": msg.pose.pose.orientation.y,
        #             "rot_z": msg.pose.pose.orientation.z,
        #             "w": msg.pose.pose.orientation.w,
        #         }
        #         localization_data.append(loc_info)
        # with rosbag.Bag(bag_path_3, "r") as bag:
        #     for _, msg, _ in bag.read_messages(topics=[loc_topic]):
        #         loc_stamp = msg.header.stamp.to_sec()
        #         loc_info = {
        #             "stamp": loc_stamp,
        #             "x": msg.pose.pose.position.x,
        #             "y": msg.pose.pose.position.y,
        #             "z": msg.pose.pose.position.z,
        #             "rot_x": msg.pose.pose.orientation.x,
        #             "rot_y": msg.pose.pose.orientation.y,
        #             "rot_z": msg.pose.pose.orientation.z,
        #             "w": msg.pose.pose.orientation.w,
        #         }
        #         localization_data.append(loc_info)
        return localization_data

    def join_images_with_localization(self, localization_data):
        """
        Reads images from a folder and assigns localization information to each image.

        """
        image_files = sorted(os.listdir(self.image_save_folder))
        images_with_localization = []

        for image_file in image_files:
            image_path = os.path.join(self.image_save_folder, image_file)
            image_stamp = float(os.path.splitext(image_file)[0])  # Extract timestamp from filename

            # Find the closest localization stamp
            closest_loc_info = min(
                localization_data, key=lambda loc: abs(loc["stamp"] - image_stamp)
            )

            # Add localization information to the image
            images_with_localization.append(
                {"image_path": image_path, "localization_info": closest_loc_info}
            )

        return images_with_localization

    def SET_BEV_MAP(self, H, left, right, bottom, top):
        """
        sets up the Bird's Eye View (BEV) map by transforming map coordinates
        to camera coordinates using the given homography matrix H.
        """

        # Set the map dimensions
        output_height = floor(top) - floor(bottom) + 1
        output_width = floor(right) - floor(left) + 1

        # Define map_x and map_y with default value -1
        map_x = np.full(shape=(output_height, output_width), fill_value=-1, dtype=np.float32)
        map_y = np.full(shape=(output_height, output_width), fill_value=-1, dtype=np.float32)

        # Create a range of coordinates
        u = np.linspace(floor(left), floor(right), floor(right) - floor(left) + 1).astype(int)
        v = np.linspace(floor(bottom), floor(top), floor(top) - floor(bottom) + 1).astype(int)
        map_u, map_v = np.meshgrid(u, v)

        # Flatten map coordinates into row vectors
        flat_map_x_coord = map_v.flatten()
        flat_map_y_coord = map_u.flatten()

        # Concatenate map_x and map_y to a matrix containing all
        #  the map coordinates for each image pixel
        coords = np.vstack((flat_map_y_coord, flat_map_x_coord))

        # Get camera coordinates of each grid map point
        uv_coords = map_to_camera(H, coords).astype(np.float32)

        # Create a mask to filter valid camera coordinates
        mask = (
            (uv_coords[0] > 1)
            & (uv_coords[0] < self.image_width - 1)
            & (uv_coords[1] > self.blur)
            & (uv_coords[1] < self.image_height - 1)
        )

        # Map each grid map point to its correspondence in the camera image
        map_x[flat_map_x_coord - flat_map_x_coord[0], flat_map_y_coord - flat_map_y_coord[0]] = (
            uv_coords[0]
        )
        map_y[flat_map_x_coord - flat_map_x_coord[0], flat_map_y_coord - flat_map_y_coord[0]] = (
            uv_coords[1]
        )

        return map_x, map_y, flat_map_x_coord[mask], flat_map_y_coord[mask]

    def get_map_to_vehicle_matrix(self, extrinsic_w_v):

        return extrinsic_w_v @ self.scale_matrix

    def make_2d_grid_map(
        self,
        images_with_localization,  # List of dictionaries containing image paths
        # and localization info
        Delta_x,  # Horizontal resolution of the grid map
        Delta_y,  # Vertical resolution of the grid map
        distance_threshold,  # Minimum distance between consecutive images to be considered
        args
    ):
        """
        This function generates a 2D grid map by processing a series of images
        with associated localization information. It:
        1. selects images according to distance_threshold
        2. Calculates the extrinsic and homography matrices for projecting image points to
          the map coordinate system.
        3. Creates a grid map and remaps image points onto this map.
        4. Stacks images by considering the closest points from the vehicle.
        5. Saves the final 2D grid map and metadata to the specified folder.
        """

        processed_images = []  # List to store images that meet the distance threshold
        stacked_image = None  # Variable to hold the final stacked image
        count = 0  # Counter for image filenames

        # Paths to ROS bag files

        # Set map limits based on the ROS bag files
        MapGenerator.set_map_limits(self)

        # Calculate the dimensions of the grid map
        map_height = int((self.y_max - self.y_min) / Delta_y)
        map_width = int((self.x_max - self.x_min) / Delta_x)

        # Define the points representing the corners of an image
        P1 = np.array([[0], [self.blur]])
        P2 = np.array([[0], [self.image_height]])
        P3 = np.array([[self.image_width], [self.blur]])
        P4 = np.array([[self.image_width], [self.image_height]])

        # Process the first image and set it as the initial reference point
        if images_with_localization:
            previous_loc = images_with_localization[0]["localization_info"]
            processed_images.append(images_with_localization[0])

            # Process the rest of the images based on the distance threshold
            for item in images_with_localization[1:]:
                current_loc = item["localization_info"]
                distance = calculate_distance(previous_loc, current_loc)

                if distance >= distance_threshold:
                    processed_images.append(item)
                    previous_loc = current_loc

        # Iterate over the processed images
        for item in processed_images:
            image_path = item["image_path"]
            loc_info = item["localization_info"]

            # Read the image from the file system
            image = cv.imread(image_path)

            # Get the extrinsic matrix from the world to the vehicle
            extrinsic_w_v = MapGenerator.get_extrinsic_world_to_vehicle(
                self,
                loc_info["x"],
                loc_info["y"],
                loc_info["z"],
                loc_info["rot_x"],
                loc_info["rot_y"],
                loc_info["rot_z"],
                loc_info["w"],
            )

            # Calculate the homography matrix from the map to the camera
            H = MapGenerator.get_map_to_camera_homography_matrix(self, extrinsic_w_v)

            # Project the image corners to the map coordinate system
            P1_map = camera_to_map(H, P1)
            P2_map = camera_to_map(H, P2)
            P3_map = camera_to_map(H, P3)
            P4_map = camera_to_map(H, P4)
            
            road_mask_map=camera_to_map(H,np.squeeze(road_mask_camera))

            # Determine the boundaries of the projected image in the map coordinate system
            top = np.max(
                [P1_map[1][0], np.max([P2_map[1][0], np.max([P3_map[1][0], P4_map[1][0]])])]
            )
            bottom = np.min(
                [P1_map[1][0], np.min([P2_map[1][0], np.min([P3_map[1][0], P4_map[1][0]])])]
            )
            left = np.min(
                [P1_map[0][0], np.min([P2_map[0][0], np.min([P3_map[0][0], P4_map[0][0]])])]
            )
            right = np.max(
                [P1_map[0][0], np.max([P2_map[0][0], np.max([P3_map[0][0], P4_map[0][0]])])]
            )

            # Create a grid of points corresponding to the image
            u, v = np.meshgrid(
                np.linspace(0, self.image_width - 1, self.image_width),
                np.linspace(self.blur, self.image_height - 1, self.image_height - self.blur),
            )
            camera_u_v = np.zeros(
                (2, self.image_width * (self.image_height - self.blur)), np.double
            )
            camera_u_v[0, ...] = u.flatten()
            camera_u_v[1, ...] = v.flatten()

            # Get the 2D grid map coordinates of each image point
            map_u, map_v, flat_map_x_coord, flat_map_y_coord = MapGenerator.SET_BEV_MAP(
                self, H, left, right, bottom, top
            )

            # Remap the image to the 2D grid map
            new_image = cv.remap(image, map_u, map_v, cv.INTER_LINEAR)
            road_mask_map= cv.remap(road_mask_camera, map_u, map_v, cv.INTER_LINEAR)

            # Initialize an empty grid map
            grid_map = np.zeros(shape=(map_height, map_width, 3), dtype=np.uint8)

            # Define the section of the grid map that corresponds to the projected image
            map_section = grid_map[
                floor(bottom) : floor(top) + 1, floor(left) : floor(right) + 1, :
            ]
            map_section[:, :, :] = new_image

            # Get the matrix to transform map coordinates to vehicle coordinates
            map_to_vehicle_matrix = MapGenerator.get_map_to_vehicle_matrix(self, extrinsic_w_v)

            # Calculate the distances for each point in the grid map
            distances = get_distances(map_to_vehicle_matrix, flat_map_x_coord, flat_map_y_coord)

            # Initialize a distance map with a high default value
            distance_map = np.full(shape=(map_height, map_width), fill_value=500, dtype=np.float16)
            distance_map[flat_map_x_coord, flat_map_y_coord] = distances

            # Stack the images based on the distances to keep the closest points
            road_map = np.zeros((map_height, map_width),dtype=np.double)
            road_map_section = road_map[floor(bottom) : floor(top) + 1, floor(left) : floor(right) + 1]
            road_map_section[:, :] = road_mask_map
            if stacked_image is None:
                stacked_image = grid_map
                stacked_distances = distance_map
                stacked_road_map=road_map

            else:
                mask = (
                    distance_map[flat_map_x_coord, flat_map_y_coord]
                    < stacked_distances[flat_map_x_coord, flat_map_y_coord]
                )
                stacked_image[flat_map_x_coord[mask], flat_map_y_coord[mask]] = grid_map[
                    flat_map_x_coord[mask], flat_map_y_coord[mask]
                ]
                stacked_distances[flat_map_x_coord[mask], flat_map_y_coord[mask]] = distance_map[
                    flat_map_x_coord[mask], flat_map_y_coord[mask]
                ]
                stacked_road_map[road_map>0] = road_map[road_map>0]
            #save/do not save generated map after each iteration (for debugging)
            if args.save_map:
                stacked_output_path = os.path.join(self.map_folder, "frame%06i.jpg" % count)
                write_image_on_disk(stacked_output_path, stacked_image)
                road_path = os.path.join(self.map_folder, "road_map%06i.jpg" % count)
                write_image_on_disk(road_path, stacked_road_map)
            count += 1

        # Save the final stacked map
        stacked_output_path = os.path.join(self.map_folder, "frame%06i.jpg" % count)
        road_path = os.path.join(self.map_folder, "road_map%06i.jpg" % count)
        write_image_on_disk(stacked_output_path, stacked_image)
        write_image_on_disk(stacked_road_map, stacked_road_map)

        # Save the world file with the grid map metadata
        wld_path = os.path.join(self.map_folder, "frame%06i.wld" % count)
        f = open(wld_path, "w")
        f.write("{} \n0 \n0 \n{}\n{}\n{}".format(Delta_x, -Delta_y, self.x_min, self.y_max))
        f.close() 
