
import sys
sys.path.append('/home/yujie/Projects/mitsuba-tensorNLB/scripts/apriltag')

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# from util import *

import tqdm
import apriltag
import cv2
import pickle


import lib.utils.exr as exr
from lib.utils.apriltag import obj_points


np.set_printoptions(precision=4, suppress=True)


def my_print(print_func):
    from datetime import datetime
    import traceback, os
    def wrap(*args, **kwargs):
        i = -2
        call = traceback.extract_stack()[i]
        while call[2] in ('log', 'show'):
            i -= 1
            call = traceback.extract_stack()[i]
        print_func(f'\x1b[0;96;40m[{datetime.now().strftime("%H:%M:%S")} {os.path.relpath(call[0])}:{call[1]}]\x1b[0;37;49m ', end='')
        print_func(*args, **kwargs)
    return wrap
pr = print
print = my_print(print)


def read_exr_as_Image(file_name):
    img = exr.read(file_name)
    img = img / (img + 1.0) ##! this tonemapping, and all possible transformation here (and corresponding in save_Image_as_exr()) won't change the generated data
    ##! this is just a safety function to make sure all values are available, and some potential scaling methods (to avoid exr-to-png precision loss as much as possible)
    return Image.fromarray((img * 255).astype(np.uint8))


def save_Image_as_exr(img, file_name):
    img = np.asarray(img).astype(np.float32) / 255.0
    img = img / (1.0 - img)
    return exr.write(img, file_name)


def detect_tags(img):
    
    img_width = img.width if img.width > img.height else img.height
    s = max(1, img_width // 1000)
    img = img.resize((round(img.width/s), round(img.height/s)), Image.LANCZOS)
    img = np.array(img.convert('L'))
    detector = apriltag.Detector()
    result = detector.detect(img)
    return result


def rotate_needed(detection):

    ''' [fjh] use the first tag to detect if needs rotation '''

    c = detection[0].corners
    c1,c2,c3,c4 = np.split(c,4, axis=0)
    c1 = c1[0];c2 = c2[0];c3 = c3[0];c4 = c4[0]
    if abs((c1[0]+c2[0])-(c3[0]+c4[0])) < abs((c1[1]+c2[1])-(c3[1]+c4[1])):
        if (c1[0]+c4[0]) < (c2[0]+c3[0]):
            rot = 0
        else:
            rot = 180
    else:
        if (c1[0]+c2[0]) < (c3[0]+c4[0]):
            rot = -90
        else:
            rot = 90

    return len(detection), rot


def get_tag_corners(img, work_dir, idx, output_detect_images=False):

    img_width = img.width if img.width > img.height else img.height
    s = max(1, img_width // 1000)
    result = detect_tags(img)

    tag_id_list = []
    center_list = []
    corners_list = []
    for tag in result:
        tag_id_list.append(tag.tag_id)
        center_list.append(tag.center.astype('float32') *s)
        corners_list.append(tag.corners.astype('float32') *s)

    if output_detect_images:
        plt.imshow(np.array(img))
        for center in center_list:
            plt.plot(center[0], center[1], 'k.')
        for corners in corners_list:
            plt.plot(corners[0,0], corners[0,1], 'r.')
            plt.plot(corners[1,0], corners[1,1], 'g.')
            plt.plot(corners[2,0], corners[2,1], 'b.')
            plt.plot(corners[3,0], corners[3,1], 'y.')
        plt.title(idx)
        # plt.show()
        plt.savefig(os.path.join(work_dir, 'detect_%02d.jpg' % idx))
        plt.close()
        # exit()

    corners_list = np.vstack(corners_list)

    return corners_list, tag_id_list


def preprocess(target_pattern):
    files = sorted(glob.glob(target_pattern))
    # print('input files:', "\n".join(files), sep='\n')

    with tqdm.tqdm(files) as pbar:
        for file in pbar:
            img = read_exr_as_Image(file)
            detection=detect_tags(img)

            detected_markers, rot = rotate_needed(detection=detection)
            img = img.rotate(rot, expand=True)
            # img.save(file.replace('.exr', '.png'))
            save_Image_as_exr(img, file)
            pbar.set_description(f'{os.path.basename(file)}, rotate: {rot}')

    return len(files)


def calibrate(imgs, work_dir, objectPoints, output_detect_images):
    N = len(imgs)
    W = imgs[0].width
    H = imgs[0].height

    objectPoints_list = []
    imagePoints_list = []
    with tqdm.tqdm(range(N)) as pbar:
        for idx in pbar:
            img = imgs[idx]
            corners, tag_list = get_tag_corners(img, work_dir, idx, output_detect_images)
            imagePoints_list.append(corners) ## detected positions in images coordinate system
            cur_objectPoints_list = [] ## actual positions in world coordinate system
            for tag in tag_list:
                cur_objectPoints_list.append(objectPoints[tag*4:(tag+1)*4, :])
            objectPoints_list.append(np.vstack(cur_objectPoints_list))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints_list, imagePoints_list, (W,H), None, None)
    
    ## render
    # ret  0.44535891719583515
    # mtx  [[1504.716     0.      749.5267]
    #       [   0.     1504.8268  999.5733]
    #       [   0.        0.        1.    ]]
    # dist [[-0.0382  0.2496 -0.0001 -0.0001 -0.565 ]]
    
    ## capture wood
    # [20:38:42 scripts/calib.py:150] 1.1829471462663745
    # [20:38:42 scripts/calib.py:151] [[1553.9084    0.      997.6953]
    #                                  [   0.     1551.9198  722.1417]
    #                                  [   0.        0.        1.    ]]
    # [20:38:42 scripts/calib.py:152] [[ 0.0734 -0.2189 -0.0043 -0.0022  0.1588]]

    # for idx in range(N):
    #     reprojectPoints, _ = cv2.projectPoints(
    #         objectPoints_list[idx], rvecs[idx], tvecs[idx], mtx, dist)
    #     reprojectPoints = np.reshape(reprojectPoints, (reprojectPoints.shape[0],2))
    #     plt.imshow(np.array(imgs[idx]))
    #     plt.plot(reprojectPoints[:,0], reprojectPoints[:,1], 'r.')
    #     plt.show()

    return mtx, dist, rvecs, tvecs, objectPoints_list, imagePoints_list


def reorder(cameraPos, imgs, objectPoints, imagePoints, rvecs, tvecs):

    def move_row_to_end(mat, idx):
        mat = np.append(mat, np.reshape(mat[idx,:], (-1,3)), axis=0)
        mat = np.delete(mat, idx, axis=0)
        return mat

    def move_element_to_end(lis, idx):
        lis.append(lis[idx])
        del lis[idx]
        return lis

    ## [fjh] find the left / right / up / down most points
    idx = np.argmax( cameraPos[:,0]+cameraPos[:,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    idx = np.argmax(-cameraPos[:-1,0]-cameraPos[:-1,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    idx = np.argmax(-cameraPos[:-2,0]+cameraPos[:-2,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    idx = np.argmax( cameraPos[:-3,0]-cameraPos[:-3,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    ## [fjh] find the max and min x / y, apart from above maximals 
    idx = np.argmax( cameraPos[:-4,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    idx = np.argmin( cameraPos[:-5,1])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    idx = np.argmin( cameraPos[:-6,0])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)
    
    idx = np.argmax( cameraPos[:-7,0])
    cameraPos       = move_row_to_end(cameraPos, idx)
    imgs            = move_element_to_end(imgs,         idx)
    objectPoints    = move_element_to_end(objectPoints, idx)
    imagePoints     = move_element_to_end(imagePoints,  idx)
    rvecs           = move_element_to_end(rvecs,        idx)
    tvecs           = move_element_to_end(tvecs,        idx)

    return cameraPos, imgs, objectPoints, imagePoints, rvecs, tvecs


def process(work_dir, board_size, res, crop, target_pattern, board_thickness_offset):

    objectPoints = obj_points.initObjPoints(board_size) # letter

    # files = []
    # for file in sorted(glob.glob(target_pattern)):
    #     numeric_part = file.split(os.sep)[-1].split('.')[0]
    #     first_four_digits = int(numeric_part[:4])
    #     if True or first_four_digits < 200:
    #         files.append(file)
    files = sorted(glob.glob(target_pattern))
    imgs = []
    print('load images')
    with tqdm.tqdm(files) as pbar:
        for file in pbar:
            img = read_exr_as_Image(file)
            detection=detect_tags(img)
            # print(len(detection))       
            if len(detection) < 16: # process input that hasn't all 16 markers
                if '_BAD.exr' not in file:
                    os.system(f"mv {file} {file.replace('.exr', '_BAD.exr')}")
                continue
            detected_markers, rot = rotate_needed(detection=detection)
            img = img.rotate(rot, expand=True)
            # img.save(file.replace('.exr', '.png'))
            # save_Image_as_exr(img, file)
            pbar.set_description(f'{os.path.basename(file)}, rotate: {rot}')
            imgs.append(img)
    N = len(imgs)
    print(f'{N} of {len(files)} files used.')

    print('calibration')
    mtx, dist, rvecs, tvecs, objectPoints_list, _ = calibrate(imgs, work_dir, objectPoints, output_detect_images=False)
    imgs_undistort = imgs
    print('intrinsics:\n', mtx)

    imgs_undistort = []
    print('reload undistorted images')
    for img in tqdm.tqdm(imgs):
        img = np.array(img)
        img = cv2.undistort(img, mtx, dist)
        imgs_undistort.append(Image.fromarray(img))

    # [yujie] save undistorted images
    # for idx, img in enumerate(imgs_undistort):
    #     file_path = os.path.join(work_dir, f"undistorted_image_{idx}.jpg")
    #     img.save(file_path, 'JPEG')
    
    print('undistorted image calibration')
    mtx, dist, rvecs, tvecs, objectPoints_list, _ = \
            calibrate(imgs_undistort, work_dir, objectPoints, output_detect_images=False)
    print('intrinsics:\n', mtx)

    cameraPos = []
    for idx in range(N):
        rmat, _ = cv2.Rodrigues(rvecs[idx])
        rmat_inv = np.linalg.inv(rmat)
        tvec = -tvecs[idx]
        cameraPos.append(np.matmul(rmat_inv,tvec))

    cameraPos = np.hstack(cameraPos).transpose()
    cameraPos[:, 1] = -cameraPos[:, 1] ## in cv2, the y axis is bottom to top. we make it consistent with exr images.
    
    # we need to make sure that in our assmbled data:
    # the origin has same xy sign with camera pos; the closer to the light source, the closer the (x, y) is to (0, 0)
    #
    #  in exr and plot: 
    #   +------> x    
    #   |             
    #   |             
    #   v y           
    
    ## [fjh] [N, 3], List: N [Image(H, W)], List: N [4*num_tags, 3], List: N [4*num_tags, 2], List: N [3, 1], List N: [3, 1]
    # cameraPos, imgs_undistort, objectPoints_list, imagePoints_list, rvecs, tvecs = \
    #     reorder(cameraPos, imgs_undistort, objectPoints_list, imagePoints_list, rvecs, tvecs)

    ## plot
    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=-90, azim=-90) ## on the screeen, x is from left to right, y is from top to bottom.
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    x, y = np.meshgrid(np.linspace(-board_size*crop/res/2,board_size*crop/res/2,10),
                        np.linspace(-board_size*crop/res/2,board_size*crop/res/2,10))
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, alpha=0.2) ## draw a image plane sample

    xy_lim = max(max(np.max(np.abs(cameraPos[:, 0])), np.max(np.abs(cameraPos[:, 1]))), board_size*crop/res)
    z_lim = np.max(np.abs(cameraPos[:, 2]))
    ax.set_xlim3d(-xy_lim, xy_lim)
    ax.set_ylim3d(-xy_lim, xy_lim)
    ax.set_zlim3d(0, z_lim)
    ax.scatter(cameraPos[:,0], cameraPos[:,1], cameraPos[:,2], color='green')

    for i, pos in enumerate(cameraPos):
        ax.text(pos[0], pos[1], pos[2], f'{i}')
    plt.savefig(os.path.join(work_dir, 'camera_pos.jpg'))
    plt.close()

    ## save pkl
    # f = open(os.path.join(work_dir,'tmp.pkl'), 'wb')
    # pickle.dump([imgs_undistort, objectPoints_list, rvecs, tvecs, mtx, dist, cameraPos, board_size], f)
    # f.close()

    # f = open(os.path.join(work_dir,'tmp.pkl'), 'rb')
    # imgs_undistort, objectPoints_list, rvecs, tvecs, mtx, dist, cameraPos, board_size = pickle.load(f)
    # f.close()
    N = len(imgs_undistort)

    print('rectifying')
    for idx in tqdm.tqdm(range(N)):

        objectPoints = objectPoints_list[idx]
        warpPoints = objectPoints[:,:2].copy()
        # print(warpPoints)
        warpPoints = (warpPoints/board_size + 0.5) * res ## [fjh] -0.5 ~ 0.5 -> 0 ~ 1
        warpPoints[:,1] = res - warpPoints[:,1] ## reverse y

        markerPoints = objectPoints.copy()
        markerPoints[:,2] = -board_thickness_offset
        imagePoints, _ = cv2.projectPoints(
            markerPoints, rvecs[idx], tvecs[idx], mtx, dist)
        imagePoints = np.reshape(imagePoints, (imagePoints.shape[0],2))

        HomoMat, _ = cv2.findHomography(imagePoints, warpPoints)
        img_in = np.array(imgs_undistort[idx])

        # plt.imshow(img_in)
        # plt.plot(imagePoints[:,0],imagePoints[:,1], 'r.')
        # plt.show()

        img_out = cv2.warpPerspective(img_in, HomoMat, (res,res))
        img_out = Image.fromarray(img_out)
        # img_out.save(os.path.join(work_dir, 'marker_%02d.png' % idx))
        img_out = img_out.crop(((res-crop)/2, (res-crop)/2, (res+crop)/2, (res+crop)/2))
        # img_out.save(os.path.join(work_dir, '%02d.png' % idx))
        save_Image_as_exr(img_out, os.path.join(work_dir, '%04d_rectify.exr' % idx))

    print('board_thickness:', board_thickness_offset)
    cameraPos[:,2] += board_thickness_offset

    # print('camera positions:\n', cameraPos)
    np.savetxt(os.path.join(work_dir, 'camera_pos.txt'), cameraPos, delimiter=',', fmt='%.4f')
    print('image size:', board_size*crop/res)


if __name__ == '__main__':
    root  = '/home/yujie/data/mat'
    mat = sys.argv[1]
    res  = int(sys.argv[2]) if len(sys.argv) > 2 else 800 
    crop = int(sys.argv[3]) if len(sys.argv) > 3 else 300 
    board_size = 10.1 # cm
    board_thickness_offset = 0.4 # cm
    if_preprocess = False if len(sys.argv) > 4 and sys.argv[4] == 'nopre' else True

    folder = os.path.join(root, mat)
    target_pattern = os.path.join(folder, 'exr', '*.exr')
    print(target_pattern)
    if if_preprocess:
        print('pre-processing')
        preprocess(target_pattern=target_pattern) 
    else:
        print('skip pre-processing')
    print('processing')
    process(folder, board_size=board_size, res=res, crop=crop, target_pattern=target_pattern, board_thickness_offset=board_thickness_offset)
