import sys
import os
import numpy as np


def my_print(print_func):
    from datetime import datetime
    import traceback, os
    def wrap(*args, **kwargs):
        i = -2
        call = traceback.extract_stack()[i]
        while call[2] in ('log', 'show'):
            i -= 1
            call = traceback.extract_stack()[i]
        print_func(
            f'\x1b[0;96;40m[{datetime.now().strftime("%H:%M:%S")} {os.path.relpath(call[0])}:{call[1]}]\x1b[0;37;49m ',
            end='')
        print_func(*args, **kwargs)

    return wrap

pr = print
print = my_print(print)

materials = (sys.argv[1]).split(',')
is_video = sys.argv[2] if len(sys.argv) > 2 else None
mode = int(sys.argv[3]) if len(sys.argv) > 3 else None

# USAGE: python run_capture.py steel03 [--video_in] [0|1|2]
os.chdir('Biplane')
root = ''

for mat in materials:
    print(mat)

    if mode == 0 or mode is None:
        if is_video == '--video_in':
            os.system((
                f'python lib/utils/format_convert_mp42exr.py {mat} --video_in 2 &&'
                f'echo ">>> {mat} transfer done!"'
            ))
        else:
            os.system((
                # f'python scripts/format_transfer.py {mat} && '
                f'python lib/utils/format_convert_mp42exr.py {mat}  2 &&'
                f'echo ">>> {mat} transfer done!"'
            ))

    # if mode == 1 or mode is None:
    #     os.system((
    #         f'python scripts/calib.py {mat} {480} && '
    #         f'python scripts/make_data.py {mat} {480} && '
    #         f'python scripts/light.py {mat} && '
    #         f'echo ">>> {mat} done!"'
    #     ))

    if mode == 2 or mode is None:
        light_file = os.path.join(root, mat, 'light_intensity.txt')
        intensity = float(np.loadtxt(light_file)) if os.path.exists(light_file) else 3000
        print('load intensity', intensity)
        reso = 1600
        crop = 800
        os.system((
            f'python scripts/calib.py {mat} {reso} {crop} nopre && '
            f'python scripts/make_data.py {mat} {reso} {crop} && '
            f'python scripts/gen_data/gen_collocated.py {mat} {intensity} && '
            f'echo ">>> {mat} done!"'
        ))