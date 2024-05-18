import os, glob, time, tqdm
import shutil
import argparse


import xml.etree.ElementTree as ET


def process_xml(file_path,idx, new_pos:list, model_path:str):
    # 解析XML文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 查找并修改<lookat>元素的属性
    lookat_element = root.find(".//lookat[@origin='0, 0, -3']")
    if lookat_element is not None:
        lookat_element.set("origin", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

    for elem in root.findall(".//string[@name='checkpointPath']"):
        # 替换 value 属性
        if elem.attrib['value'] == "$model_path":
            elem.set('value', model_path)
    

    # 保存修改后的XML文件
    tmp_xmldir = './tmp'
    if not os.path.exists(tmp_xmldir):
        os.system(f'mkdir {tmp_xmldir}')
    new_file_path = file_path.replace(".xml", "_tmp.xml").split(os.sep)[-1]
    new_file_path = os.path.join(tmp_xmldir, new_file_path)
    tree.write(new_file_path)
    return new_file_path


parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', type=str, default='./scene/plane_point_persp.xml')
args = parser.parse_args()
xml_path = args.xml_path

poses = []
# for i in np.arange(-3, 3, 0.5):
#     poses.append([i, 0, -3])
# for i in np.arange(-3, 3, 0.5):
#     poses.append([0, i, -3])

# use sample-time camera pos
camera_pos_path = '/root/autodl-tmp/data/realworld/rock0302/camera_pos.txt'
with open(camera_pos_path, 'r') as file:
    for line in file:
        numbers = line.strip().split(',')
        numbers = [float(num) for num in numbers]
        numbers[-1] *= -1
        poses.append(numbers)

model_path_list = [
'/root/autodl-tmp/torch/saved_model/#compress_only-offset_depth-datagen_rock0302-tmp-epoch-30-0303_122520/epoch-90/epoch-90.pth'
]
for model_path in model_path_list:
    matname = model_path.split('/')[-3].split('-')[2]
    print('handling:',matname)

    for idx, pos in tqdm.tqdm(enumerate(poses)):
        tmp_xml_path = process_xml(xml_path,idx, pos, model_path)
        os.system(
        # print(
            f"python render_new.py {tmp_xml_path} --model_path {model_path} --output {matname}-{str(idx).zfill(4)}.exr"
            )

    os.system('rm -rf /root/pytorch-tensorNLB-master/tmp/')
    os.system('rm -rf /root/autodl-tmp/torch/render/*_tmp /root/autodl-tmp/torch/render/plane_*')
    # os.system(f'tar -czvf /root/autodl-tmp/torch/render/{matname}.tar.gz /root/autodl-tmp/torch/render/*.exr')
    os.system(f'mkdir /root/autodl-tmp/torch/render/{matname} & mv /root/autodl-tmp/torch/render/*.exr /root/autodl-tmp/torch/render/{matname}/')
    os.system(f'python exrlist_to_video.py /root/autodl-tmp/torch/render/{matname} ./{matname}-recovery.mp4 3')

