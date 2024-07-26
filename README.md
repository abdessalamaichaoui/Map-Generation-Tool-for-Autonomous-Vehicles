# Bird's Eye View (BEV) Map Generator

  

## Overview

The primary objective of this project is to assist deployers in determining the limits of the navigable corridor by creating highly precise texture maps of the ground for various sites, this code generates a 2D Bird's Eye View (BEV) grid map using a vehicle's front camera images and localization data from ROS bag files. The map is built by processing a series of images with associated localization information, remapping image points onto a grid, and stacking images based on the closest points to the vehicle.

we can create a global georeferenced map of a site with a resolution of 1cm/1 pixel.

  

## Features

  

- **Image Extraction**: Extracts images from ROS bag files and saves them with timestamps.

- **Vignetting Correction**: Optional correction of image vignetting if needed.

- **Illumination Normalization**: Optional normalization of image illumination if needed.

- **Localization Data Extraction**: Extracts localization data from ROS bag files.

- **Image and Localization Matching**: Matches extracted images with corresponding localization data.

- **BEV Map Generation**: Generates a 2D grid map by processing images and localization data.

- **Adding Image Metadata**: Adds geo metadata of generated map in wld file.

  

## Prerequisites

  

- Python 3

- OpenCV

- NumPy

- ROS Bag (for reading ROS bag files)

- Scipy

  
  

## Usage

  

1. **Run the BEV map generator:**

  

```bash

python3 make_map.py <bag_file> <images_save_dir> <map_save_dir> [options]

```

  

- `<bag_file>`: Path to the input ROS bag file.

- `<images_save_dir>`: Directory to save the extracted images with timestamps.

- `<map_save_dir>`: Directory to save the generated BEV map.

  

**Options:**

  

- `-vn`, `--vignetting`: Enable vignetting correction.

- `-nr`, `--normalization`: Enable illumination normalization.

- `-sm`, `--save_map`: Save the map after each iteration (for debugging).

  

Example:

  

```bash

python3 make_map.py my_ros_bag.bag ./images ./map -vn -nr -sm

```

  

2. **Explanation of the Main Steps:**

  

- **Camera Info Retrieval**: The script retrieves camera intrinsic parameters and distortion coefficients, as well as extrinsic parameters (pose) using the `get_camera_info()` and `get_camera_extrinsic_coordinates()` methods.

- **Extrinsic Matrix Calculation**: The camera extrinsic matrix is computed using the retrieved pose parameters with `get_extrinsic_matrix()`.

- **Blur Line Detection**: The starting line of image blur is detected using `find_blur_starting_line()`.

- **Map Limits**: The script sets the limits of the map from the bag file using `set_map_limits()`, this function takes the output of `find_blur_starting_line()` as an input .

- **Scale Matrix Setup**: The scale matrix is set using `set_scale_matrix()`.

- **Image Saving**: Images are saved in a folder with their timestamps using `save_images_from_bag()`.

- **Localization Data Extraction**: Localization data is extracted from the ROS bag file using `extract_loc_data()`.

- **Image and Localization Matching**: Images are matched with their corresponding localization data using `join_images_with_localization()`.

- **2D Grid Map Generation**: A 2D grid map is generated using `make_2d_grid_map()`.

  

## Outputs

  

- **Extracted Images**: The images extracted from the ROS bag file are saved in the specified directory with filenames as their timestamps.

- **Generated BEV Map**: The final 2D grid map is saved in the specified directory as `frameXXXXXX.jpg`.

- **Metadata**: A metadata file (`frameXXXXXX.wld`) is saved alongside the BEV map, containing the grid map metadata.

  

## How It Works

  

This project involves several steps to generate a BEV map from vehicle front camera images and localization data. The main steps include:

  

1. **Homography Matrix Computing** :

2. **Blur Detection**: Identify blurred areas in the map and adjust the image projection accordingly to avoid these areas.

3. **Defining The Zone of Interest** : Identify the part of original image that we want to project, which is a rectangle limited by 0:image width and blur line pixel:image height.

4. **Point Mapping**: Map each pixel of an image to its projection on the grid map using Inverse Perspective Mapping (IPM) technique.

5. **Image and Localization Data Matching**: Match extracted images with corresponding localization data.

6. **BEV Map Generation**: Generate and save the final BEV map with a configurable cell size.

  

### Homography Matrix Computing

  

To apply inverse perspective mapping, first, we need to calculate the Homography matrix \(H\).

$$ \lambda.\left(\begin{array}{c}u \\v\\1\end{array}\right)=K.\left(\begin{array}{cccc}1 & 0 & 0 & 0 \\0 & 1 & 0 & 0 \\0 & 0 & 1 & 0\end{array}\right).T_{FLU \rightarrow RDF}.T_{V \rightarrow C}.T_{W \rightarrow V} .S. \left(\begin{array}{c}i \\j\\1\end{array}\right)$$

This requires extracting the necessary matrices from the ROS bags:

1. **Intrinsic Matrix \(K\)**:

- Retrieved from the topic `/camera_front/camera_info` in the ROS bags, containing internal parameters of the camera.

  

2. **Extrinsic Matrix**:

- The camera pose (rotation and translation) in the vehicle frame is stored in the topic `/rosparam_dump`. The messages are in JSON format and can be accessed in Python using `json.loads(msg.data)`. The topic stores different information about the vehicle, including sensors' extrinsic parameters.

- Compute the matrix $T_v_c$ from the extracted parameters, then get $H$.

  

3. **Transformation Matrix from FLU to RDF Frame**:

- Camera extrinsic calibration is done in the FLU frame, requiring conversion to RDF frame.

  
  

4. **Transformation Matrix from World Frame to Vehicle Frame**:

- Vehicle localization information is stored in the topics `/vehicle_pose` and `/vehicle_pose_global`, which publish the vehicle’s absolute pose.

- Compute the transformation matrix $T_w_v$, then use it to get $H$.

  

5. **Scaling Matrix $S$**:

- Defines the relationship between world coordinates of the ground plane and BEV image coordinates. $\Delta x$ and $\Delta y$ are chosen based on the desired image resolution and computation time, while $X_0$ and $Y_0$ are the map’s top left coordinates in the world frame (in meters).

$$S=\left(\begin{array}{ccc}s_{x} & 0 & t_x \\0 & s_{y} & t_y \\0 & 0 & 0 \\0 & 0 & 1 \end{array}\right)$$

### Blurred Zone Detection

  

To determine the zone of interest in the image for projection onto the BEV map, it is necessary to compute the blur line, which defines the region in the image where the projection starts to blur. The blur line marks the point beyond which pixels in the original image begin to overlap. To identify this line, establish a threshold such that the distance between each pair of successive pixels in the original image corresponds to a distance in the BEV that is not less than the chosen threshold. This ensures the projected area on the BEV map maintains high detail and avoids pixel overlapping.

  

### Inverse Perspective Mapping (IPM)

  

IPM assumes the world to be flat on a plane at Z=0 and maps all pixels from a given viewpoint onto this flat plane through homography projection. This technique helps in avoiding holes in the target image and ensuring that each pixel in the BEV map has a corresponding pixel in the original image.

  

#### IPM Steps :

  

1. **Compute Scale Matrix**: Determine the relationship between the BEV image and world coordinates.

2. **Compute Homography Matrix**: Calculate the homography matrix for projecting the image onto the BEV map.

3. **Determine Zone of Interest**: Define the zone of interest in the image, based on the blur line.

4. **Remap Points**: Project the zone of interest onto the BEV map and compute Map_x and Map_y to map each point on the BEV map with its correspondence in the original image.

5. **Generate BEV Image**: Use OpenCV’s remap function to create the BEV image.

  

### BEV Map Stitching

  

Once we generate a BEV image from a single image, the next step is to stitch or stack these images. Two techniques have been explored (the second one has used in this code):

  

1. **Pasting Only New Parts**: This method pastes only the new part at each iteration, resulting in fewer artifacts.

2. **Overlaying New Images**: This method overlays each new image on the previous one, offering better precision.

  

### Minimal Distance Reconstruction

  

To improve map accuracy and handle multiple passes over the same area, we choose the closest pixel to the camera when overlapping occurs. This involves computing a distance map, where each time pixels overlap, the one with the smallest distance to the camera is selected.

  

### Adding Generated Map to SiteEditor

  

To verify the geo-positioning of our map and compare it to other maps (e.g., Lidar scans) and to help deployers draw the **Navigable Corrdior**, we can add the generated map to SiteEditor. SiteEditor allows adding georeferenced images to existing maps:

  

1. Go to "Map" in the top bar.

2. Choose "Edit GeoImage" background.

3. Select "Open a GeoImage".

4. The GeoImage can be in `.tif` or `.geotiff` format with added metadata or in `.jpeg` format with metadata saved in a `.wld` file with the same name. The `.jpeg` format is preferred for efficiency(smaller in size) and simplicity.

  

the generated `.wld` file is text file contains the following parameters:

  

- Pixel X size

- Rotation about the Y-axis (usually 0.0)

- Rotation about the X-axis (usually 0.0)

- Negative pixel Y size

- X coordinate of the upper left pixel center

- Y coordinate of the upper left pixel center

  

These parameters can also be manually configured in SiteEditor.

  
  

## How to Use the Resulting Map

The primary use of the generated Map is to display it in SiteEditor to help deployers draw the Navigable corridor.

  

### Adding the BEV Map to SiteEditor

  

#### Steps to display the Map in SiteEditor

  

1. **Open SiteEditor**: Launch SiteEditor on your computer.

2. **Navigate to the Map Section**: In the top menu, click on the "Map" tab to access map-related options.

3. **Edit GeoImage Background**: Choose the "Edit GeoImage" background option. This allows you to add georeferenced images to your existing map.

4. **Open a GeoImage**: Select the option to open a GeoImage. This will prompt you to select your image file and the corresponding world file.

- **Select the BEV Map Image**: Choose the `.jpg` or `.tif` file of your generated BEV map.

- **Select the World File**: Choose the `.wld` file you created with the georeferencing information.

5. **Configure the GeoImage Settings**: If necessary, adjust the settings for the GeoImage, such as the transparency, to better visualize it over the existing map layers.

6. **Save the Changes**: Once the BEV map is properly placed and configured, save the changes to your map.

  

#### Tips for Verification

  

- **Overlay Comparison**: Use transparency to compare the BEV map with other map layers.

- **Check Key Features**: Verify that key features (e.g., roads, nearby buildings, landmarks...) align correctly with the corresponding features in other map layers


![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/site01.gif)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/site02.gif)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/site03.gif)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/image.jpg)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/mask.png)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/frame000475.jpg)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/mask.jpg)
![Alt text](/home/abdessalamaichaoui/Downloads/2d_map_generation/images/segmentation result.png)





