import os
import glob
import numpy as np
from PIL import Image

import tqdm
import cv2
import exr

np.set_printoptions(precision=4, suppress=True)


def my_print(print_func):
    from datetime import datetime
    import traceback
    import os

    def wrap(*args, **kwargs):
        i = -2
        call = traceback.extract_stack()[i]
        while call[2] in ('log', 'show'):
            i -= 1
            call = traceback.extract_stack()[i]
        print_func(
            f'\x1b[0;96;40m[{datetime.now().strftime("%H:%M:%S")} {os.path.relpath(call[0])}:{call[1]}]\x1b[0;37;49m ', end='')
        print_func(*args, **kwargs)
    return wrap


pr = print
print = my_print(print)


def read_exr_as_Image(file_name):
    img = exr.read(file_name)
    # ! this tonemapping, and all possible transformation here (and corresponding in save_Image_as_exr()) won't change the generated data
    img = img / (img + 1.0)
    ##! this is just a safety function to make sure all values are available, and some potential scaling methods (to avoid exr-to-png precision loss as much as possible)
    return Image.fromarray((img * 255).astype(np.uint8))


def save_Image_as_exr(img, file_name):
    img = np.asarray(img).astype(np.float32) / 255.0
    img = img / (1.0 - img)
    return exr.write(img, file_name)


def normalize_quaternion(quaternion):
    """
    Normalize a quaternion to ensure its length is 1 (unit quaternion).
    """
    quaternion = np.array(quaternion)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a normalized quaternion to a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = quaternion
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def compute_camera_center(quaternion, translation_vector):
    normalized_quaternion = normalize_quaternion(quaternion)
    rotation_matrix = quaternion_to_rotation_matrix(normalized_quaternion)
    rotation_transpose = np.transpose(rotation_matrix)
    camera_center = -np.dot(rotation_transpose, translation_vector)
    return camera_center


def process(work_dir, target_pattern, resolution, crop_resolution):

    files = sorted(glob.glob(target_pattern))

    imgs = []
    print('loading images')
    for file in tqdm.tqdm(files):
        imgs.append(read_exr_as_Image(file))
        # imgs.append(0)
    N = len(imgs)
    print(f'{N} of {len(files)} files used.')


# read camera info (SIMPLE_RADIAL camera)
    camera_dict = None
    with open(os.path.join(work_dir, 'cameras.txt'), 'r') as file:
        lines = file.readlines()

        # Skip the first three lines
        lines = lines[3:]
        parts = lines[0].strip().split(' ')

        assert (len(parts) >= 4)
        # Extract the values
        camera_id = int(parts[0].strip())
        model = parts[1].strip()
        width = int(parts[2].strip())
        height = int(parts[3].strip())

        # Parse the PARAMS[] as a list
        params = [param.strip() for param in parts[4:]]

        # Create and return the dictionary
        camera_dict = {
            "CAMERA_ID": camera_id,
            "MODEL": model,
            "WIDTH": width,
            "HEIGHT": height,
            "PARAMS": params
        }
        W = width
        H = height
        params = list(map(float, params))
        focal_length = params[0]
        principal_point_x = params[1]
        principal_point_y = params[2]
        distortion = params[3]

    print("camera: ", W, H, focal_length,
          principal_point_x, principal_point_y, distortion)

# read images info
    with open(os.path.join(work_dir, 'images.txt'), 'r') as file:
        lines = file.readlines()

    # Skip the first 4 lines (headers)
    lines = lines[4:]

    image_info_list = [{'NAME': '0000.jpg'},]
    camera_center_list = []
    # Loop through the lines in pairs (two lines represent one image)
    # for i in range(0, len(lines), 2):
    for idx in tqdm.tqdm(range(len(lines)//2)):
        image_info = lines[idx*2].strip().split(' ')
        points2d_info = lines[idx*2 + 1].strip().split(' ')

        # Create a dictionary to store image information
        image_dict = {
            'IMAGE_ID': image_info[0],
            'Quaternion': list(map(float, image_info[1:5])),  # W,X,Y,Z
            'Translation': list(map(float, image_info[5:8])),  # X,Y,Z
            'CAMERA_ID': int(image_info[8]),
            'NAME': image_info[9],
            'POINTS2D': [],
        }

        # Extract POINTS2D information and append to the dictionary
        id = 0
        for j in range(0, len(points2d_info), 3):
            id += 1
            point2d = {
                'X': float(points2d_info[j]),
                'Y': float(points2d_info[j + 1]),
                'POINT3D_ID': int(points2d_info[j + 2]),
            }
            image_dict['POINTS2D'].append(point2d)

            # [yujie] test
            if False and point2d['POINT3D_ID'] == 3095 and i == 0:
                print("id: ", id)
                marker_point = np.array(
                    [[-14.493741882459741, 2.4158241901137041, 72.879177205746515]])
                rvecs = quaternion_to_rotation_matrix(image_dict['Quaternion'])
                tvecs = np.array([image_dict['Translation']])
                mtx = np.array([[focal_length, 0., principal_point_x], [
                               0., focal_length, principal_point_y], [0., 0., 1]])
                dist = np.array([[0., 0., 0., 0., 0.]])
                res, _ = cv2.projectPoints(
                    marker_point, rvec=rvecs, tvec=tvecs, cameraMatrix=mtx, distCoeffs=dist)
                print(res)

        # write camera pos
        # for idx, img_dict in enumerate(image_info_list):
        cam_center = compute_camera_center(
            image_dict['Quaternion'], image_dict['Translation'])
        image_dict['CAMERA_CENTER'] = cam_center
        # print(f"image {idx} camera pos:", cam_center)

        # Append the image dictionary to the list
        image_info_list.append(image_dict)

    # sort by name
    image_info_list.sort(key=lambda x: x['NAME'])

    # write camera pos
    with open(os.path.join(work_dir, 'camera_pos.txt'), 'a') as file:
        for img_info in image_info_list:
            if img_info['NAME'] == '0000.jpg':
                continue
            file.write(str(img_info['CAMERA_CENTER'])[1:-1] + '\n')

# read 3D points
    point3D_list = []
    x_list, y_list, z_list = [], [], []

    with open(os.path.join(work_dir, 'points3D.txt'), 'r') as file:
        # 跳过前三行
        for _ in range(3):
            next(file)

        for line in file:
            parts = line.strip().split()

            # 提取POINT3D_ID, X, Y, Z, R, G, B, ERROR部分
            point3d_id = int(parts[0])
            x, y, z, r, g, b, error = map(float, parts[1:8])

            # 提取TRACK[]部分，将每个二元组解析为元组
            track_raw = parts[8:]
            # track_dict = [(int(track_raw[i]), int(track_raw[i+1])) for i in range(0, len(track_raw), 2)]
            track_dict = {}
            for i in range(0, len(track_raw), 2):
                track_dict[int(track_raw[i])] = int(track_raw[i+1])

            # if len(track_dict) < N*0.1 or error > 1.0:
            #     continue

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            entry = {
                'POINT3D_ID': point3d_id,
                'X': x,
                'Y': y,
                'Z': z,
                'R': r,
                'G': g,
                'B': b,
                'ERROR': error,
                'TRACK_DICT': track_dict
            }

            point3D_list.append(entry)
    print("useful points num:", len(x_list))
    # print(max(x_list), min(x_list), max(y_list), min(y_list), max(z_list), min(z_list))
    max_x = max(x_list)
    min_x = min(x_list)
    max_y = max(y_list)
    min_y = min(y_list)
    max_z = max(z_list)
    min_z = min(z_list)
    mean_z = np.mean(z_list)
    rt_vec = np.array([max_x, max_y, mean_z])
    lt_vec = np.array([min_x, max_y, mean_z])
    lb_vec = np.array([min_x, min_y, mean_z])
    rb_vec = np.array([max_x, min_y, mean_z])
    marker_point_id = [-1]*5
    marker_points = np.zeros((5, 3))

    # choose marker point
    for idx, key in enumerate([64341, 44604, 47962, 7909]):
        for entry in point3D_list:
            if entry['POINT3D_ID'] == key:
                marker_points[idx, :] = np.array(
                    [entry['X'], entry['Y'], entry['Z']])
                break
        # marker_points[idx, :] = result_array[key-1,:]

    def proj2image(point: np.ndarray, imgID: int) -> np.ndarray:
        rvecs = quaternion_to_rotation_matrix(
            image_info_list[imgID]['Quaternion'])
        tvecs = np.array([image_info_list[imgID]['Translation']])
        mtx = np.array([[focal_length, 0., principal_point_x], [
                       0., focal_length, principal_point_y], [0., 0., 1]])
        dist = np.array([[0., 0., 0., 0., 0.]])
        res, _ = cv2.projectPoints(
            point, rvec=rvecs, tvec=tvecs, cameraMatrix=mtx, distCoeffs=dist)
        res = np.reshape(res, (res.shape[0], 2))
        return res

# rectifying
    print('rectifying')
    for idx in tqdm.tqdm(range(1, len(image_info_list))):
    # for idx in tqdm.tqdm(range(1,5)):
        srcPoints = []
        for i in range(4):
            proj_point = proj2image(marker_points[i, :], idx)[0]
            srcPoints.append(proj_point)

        srcPoints = np.array(srcPoints)
        # destPoints = np.array([[0.25,0.75],[0.75,0.75],[0.5,0.5],[0.25,0.25],[0.75,0.25]])
        destPoints = np.array(
            [[0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25]])
        destPoints *= resolution
        destPoints[:, 1] = resolution - destPoints[:, 1]  # reverse y

        # print(f"image{idx} :", srcPoints)
        # print(f"image{idx} :", destPoints)
        HomoMat, _ = cv2.findHomography(srcPoints, destPoints)
        img_in = np.array(imgs[idx-1])

        # plt.imshow(img_in)
        # plt.plot(imagePoints[:,0],imagePoints[:,1], 'r.')
        # plt.show()

        img_out = cv2.warpPerspective(
            img_in, HomoMat, (resolution, resolution))
        img_out = Image.fromarray(img_out)
        img_out.save(os.path.join(work_dir, 'marker_%02d.png' % idx))
        img_out = img_out.crop(((resolution-crop_resolution)/2, (resolution-crop_resolution)/2,
                               (resolution+crop_resolution)/2, (resolution+crop_resolution)/2))
        # img_out.save(os.path.join(work_dir, '%02d.png' % idx))
        save_Image_as_exr(img_out, os.path.join(
            work_dir, '%02d_rectify.exr' % idx))


if __name__ == '__main__':
    work_dir = "/home/yujie/data/mat/video_test2_square"
    target_pattern = os.path.join(work_dir, 'exr', '*.exr')
    process(work_dir=work_dir, target_pattern=target_pattern,
            resolution=2160, crop_resolution=600)
