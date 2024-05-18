import os
import sys
import glob
import tqdm

material = sys.argv[1]
raw_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/data/collocated_auto_close/{material}/raw'
exr_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/data/collocated_auto_close/{material}/exr'
out_dir = f'/test/repositories/mitsuba-pytorch-tensorNLB/data/collocated_auto_close/{material}'

os.system(f'mkdir -p {exr_dir} {out_dir}')

files = glob.glob(os.path.join(raw_dir, '*.dng'))
with tqdm.tqdm(files) as pbar:
    for file in pbar:
        pbar.set_description(os.path.basename(file))
        os.system(f'dcraw -W -g 1 1 -r 2.137632 1.000000 1.974059 1.000356 -j {file}') ##! CAUTION: because of gamma=1, the final PNG output will seems different from EXR output.
        os.system(f'pfsin {file.replace(".dng", ".ppm")} | pfsout {os.path.join(exr_dir, os.path.basename(file).replace(".dng", ".exr"))}')