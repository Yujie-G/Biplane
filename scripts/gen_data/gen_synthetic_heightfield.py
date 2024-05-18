import tqdm
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import exr


def show_sample_points(points, save_path):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title("Random Sample Points on Hemisphere")
    plt.savefig(save_path)
    
def random_sample_points_on_sphere(center, radius, num_points):
    """
    Randomly sample points on a hemisphere in 3D space.

    :param center: Tuple (x, y, z) representing the center of the hemisphere.
    :param radius: Radius of the hemisphere.
    :param num_points: Number of points to generate on the hemisphere.
    :return: Array of points on the hemisphere.
    """
    # Randomly generate angles theta (azimuthal angle) and phi (polar angle)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi / 2, num_points)

    # Convert spherical coordinates to Cartesian coordinates
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] - radius * np.cos(phi)

    return np.column_stack((x, y, z))

def random_sample_on_plane(center, d, num_points, radius_ratio):
    """
    Randomly samples points on a plane within a square region centered at 'center' with half-width 'd'.
    A higher probability is given to sample within a central circular region.

    :param center: Center of the square (and the circle) as a tuple (x, y)
    :param d: Half-width of the square, i.e., the square extends d units in each direction from the center
    :param num_points: Number of points to sample
    :param radius_ratio: Ratio of the circle's radius to the half-width of the square
    :return: List of sampled points
    """
    # Radius of the inner circle
    radius = radius_ratio * d

    points = []
    for _ in range(num_points):
        # Decide whether to place the point inside the circle or in the square outside the circle
        if np.random.rand() < 0.6:  # Adjust this probability to change the distribution
            # Sample within the circle
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
        else:
            # Sample within the square but outside the circle
            while True:
                x = np.random.uniform(center[0] - d, center[0] + d)
                y = np.random.uniform(center[1] - d, center[1] + d)
                if np.sqrt((x - center[0])**2 + (y - center[1])**2) > radius:
                    break
        z = center[2]
        points.append((x, y, z))

    return np.array(points)


material = 'rockRoad_synthetic'
xml_path = '/root/heightfield_or_normalmap/collocated.xml'
svbrdf_path = os.path.join(f'/root/autodl-tmp/data/mat_svbrdf/{material}', material.split('_')[0])
scale_list = ['1.6', '2.4','3.2', '5', '7', '10']
for scale in scale_list:
    data_dir = f'/root/autodl-tmp/data/synthetic_data/{material}-scale{scale}/exr'
    flag = os.system(f'mkdir -p {data_dir}')
    print(data_dir)

    


    num = 50  # Number of points to sample
    radius_ratio=0.6
    points1 = random_sample_on_plane(center = (0, 0, -20), d = 10, num_points=15, radius_ratio=radius_ratio)
    points2 = random_sample_points_on_sphere(center=(0, 0, -3), radius=15, num_points = 45)
    # print(points2)
    points = np.concatenate((points1, points2))
    show_sample_points(points, save_path=f'./sample_points/{material}.jpg')

    with tqdm.tqdm(range(num)) as pbar:
        for i in pbar:
            x,y,z = points[i]
            # print(x,y,z)
            pbar.set_description(f'{x:.3f}_{y:.3f}_{z:.3f}')
            os.system(
            # print(
                f'mitsuba -q -o{os.path.join(data_dir, f"{i}.exr")} \
    -Droughness_path={svbrdf_path}_roughness.exr -Dbasecolor_path={svbrdf_path}_basecolor.exr -Dheight_path={svbrdf_path}_height.exr \
    -Dx={x:.3f} -Dy={y:.3f} -Dz={z:.3f} -Dscale={float(scale):.3f} {xml_path}'
            )
    # os.system('rm *.log')
    # exit(0)

    ### process

    os.system(f'cd /root/mitsuba-tensorNLB-master/scripts && python run_capture.py {material}-scale{scale} 2')

exit(0)



CHANNELS = ['0_wix', '1_wiy', '2_wox', '3_woy', '4_u', '5_v', '6_R', '7_G', '8_B']

cache_file_shape = 400
out_dir = f'/root/autodl-tmp/data/datagen/{material}/{cache_file_shape}x{cache_file_shape}'
print(out_dir)
os.system(f'mkdir -p {out_dir}')

file_list = sorted(glob.glob(os.path.join(data_dir, '*.exr')))
data = None
i = 0
with tqdm.tqdm(file_list) as pbar:
    for file in pbar:
        cur_data = exr.read(file, channels=CHANNELS)
        mask = np.where(cur_data[..., -1] >= 0)
        cur_data = cur_data[mask]
        # print(cur_data.shape)
        if data is None:
            data = cur_data
        else:
            data = np.concatenate([data, cur_data], 0)
        # print(data.shape[0])
        if data.shape[0] >= cache_file_shape ** 2:
            exr.write(data[:cache_file_shape**2].reshape(cache_file_shape, cache_file_shape, 9), os.path.join(out_dir, f'0_{i}.exr'), channels=CHANNELS)
            data = data[cache_file_shape**2:]
            i += 1
print(i)