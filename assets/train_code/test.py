import torch
from line_profiler import LineProfiler

def main():
    for i in range(10):
        data = torch.randn(262144, 100, 9)
        data = data.to('cuda:0')
        x = data[..., 0:3]
        y = data[..., 3:6]
        z = data[..., 6:9]
        x, y, z = x - 1.0, y - 1.0, z - 1.0
        del data
    
def main2():
    for i in range(10):
        data = torch.randn(262144, 100, 9)
        x = data[..., 0:3]
        y = data[..., 3:6]
        z = data[..., 6:9]
        x, y, z = map(lambda x: x.to('cuda:0'), [x, y, z])
        x, y, z = x - 1.0, y - 1.0, z - 1.0
        del data
    
if __name__ == "__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()