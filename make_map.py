"""
Make a 2d Grid Map based on vehicle's front camera images and its localization
"""

import library.mapping_library as map
import argparse
import time

def read_parameters_from_file(filename):
    parameters = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 1:
            parameters['Delta_x'] = float(lines[0].strip())
        if len(lines) >= 2:
            parameters['Delta_y'] = float(lines[1].strip())
        if len(lines) >= 3:
            parameters['Delta'] = float(lines[2].strip())
        if len(lines) >= 4:
            parameters['distance_threshold'] = float(lines[3].strip())
    return parameters


def main():
    parser = argparse.ArgumentParser(description="make BEV map.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("images_save_dir", help="Extracted images with timestamps directory.")
    parser.add_argument("map_save_dir", help="generated map save directory.")
    parser.add_argument("parameters_file", help="File containing Map parameters")
    parser.add_argument("-vn", "--vignetting", help="Enable/Disable Vignetting correction",
                        action="store_true")
    parser.add_argument("-nr", "--normalization", help="Enable/Disable illumination normalization",
                        action="store_true")
    parser.add_argument(
        "-sm", "--save_map", help="save / do not save map after each iteration (for debugging)",
        action="store_true")
    
    args = parser.parse_args()

    parameters = read_parameters_from_file(args.parameters_file)
    Delta_x = parameters.get('Delta_x', 0.015)  # Default value if not found in file
    Delta_y = parameters.get('Delta_y', 0.015)  
    Delta = parameters.get('Delta', 0.015)      
    distance_threshold = parameters.get('distance_threshold', 0.1)  

    MG = map.MapGenerator(args)
    MG.get_camera_info()
    roll, pitch, yaw, tx, ty, tz = MG.get_camera_extrinsic_coordinates()
    MG.get_extrinsic_matrix(roll, pitch, yaw, tx, ty, tz)
    MG.set_map_limits()
    MG.set_scale_matrix(Delta_x, Delta_y)
    MG.find_blur_starting_line(Delta)
    MG.save_images_from_bag(args)
    localization_data = MG.extract_loc_data()
    images_with_localization = MG.join_images_with_localization(localization_data)
    MG.make_2d_grid_map(images_with_localization, Delta_x, Delta_y, distance_threshold, args)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(time.time() - start_time)

