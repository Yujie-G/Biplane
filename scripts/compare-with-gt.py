
from torch.utils.data import Dataset
from btf_extractor import Ubo2014
import matplotlib.pyplot as plt

from lib.utils import exr
from utils import *
from config import RepConfig


class BTFDataset(Dataset):
    def __init__(self, path, sample_size=64, train_size=2048, validation_size=512, side_len=512, shuffle=True,
                 max_level=4):
        self.shuffle = shuffle
        self.btf = Ubo2014(path)
        self.sample_size = sample_size
        # self.train_size = train_size
        self.side_len = side_len
        angles = list(self.btf.angles_set)
        self.angles = np.array(angles)
        self.train_size = train_size
        self.validation_size = validation_size
        choice = np.random.choice(len(angles), (self.train_size + self.validation_size,))
        self.train = self.angles[choice[:self.train_size]]
        self.validation = self.angles[choice[self.train_size:]]

        # rgb + uv + wi + wo
        self.data_ch = 3 + 2 + 3 + 3 + 1
        # UBO img width
        self.img_width = 400

    # def get_shuffled_sample(self, ):
    #     choice = np.random.choice(self.data.shape[0], (self.side_len ** 2,))
    #
    #     data = self.data[choice].reshape((self.side_len, self.side_len, self.data_ch))
    #     image = data[..., :3]
    #     coords = data[..., 3:5]
    #     level = data[..., 5:6]
    #     wi = data[..., 6:9]
    #     wo = data[..., 9:]
    #
    #     # convert to tensors
    #     image = torch.tensor(image, dtype=torch.float32)
    #     coords = torch.tensor(coords, dtype=torch.float32)
    #     wi = torch.tensor(wi, dtype=torch.float32)
    #     wo = torch.tensor(wo, dtype=torch.float32)
    #     level = torch.tensor(level, dtype=torch.float32)
    #
    #     return image, coords, level, wi, wo

    def get_sample(self, idx):
        a = self.angles[idx]
        t_l, p_l, t_v, p_v = a
        image = self.btf.angles_to_image(*a)
        image = image[..., ::-1].copy()

        wi = spherical2dir(t_l, p_l)
        wo = spherical2dir(t_v, p_v)
        wi = np.tile(wi, (self.img_width, self.img_width, 1))
        wo = np.tile(wo, (self.img_width, self.img_width, 1))
        level = np.tile(0., (self.img_width, self.img_width, 1))
        coords = create_uvs(self.img_width).reshape((self.img_width, self.img_width, 2))

        # convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        wi = torch.tensor(wi, dtype=torch.float32)
        wo = torch.tensor(wo, dtype=torch.float32)
        level = torch.tensor(level, dtype=torch.float32)

        return image, coords, level, wi, wo

    # def get_validation_generator(self, ):
    #     coords = utils.create_uvs(self.img_width).reshape((self.img_width, self.img_width, 2))
    #     coords = torch.tensor(coords, dtype=torch.float32)
    #     for a in self.validation:
    #         t_l, p_l, t_v, p_v = a
    #         image = self.btf.angles_to_image(*a)
    #         image = image[..., ::-1].copy()
    #
    #         wi = utils.spherical2dir(t_l, p_l)
    #         wo = utils.spherical2dir(t_v, p_v)
    #         wi = np.tile(wi, (self.img_width, self.img_width, 1))
    #         wo = np.tile(wo, (self.img_width, self.img_width, 1))
    #         level = np.tile(0., (self.img_width, self.img_width, 1))
    #
    #         level = torch.tensor(level, dtype=torch.float32)
    #         image = torch.tensor(image, dtype=torch.float32)
    #
    #         wi = torch.tensor(wi, dtype=torch.float32)
    #         wo = torch.tensor(wo, dtype=torch.float32)
    #         yield image, coords, level, wi, wo
    #
    # def get_ds_generator(self, ):
    #     for i in range(self.train_size):
    #         yield self.get_sample(i)
    #
    # def get_input(size, wo=[0., 0.], wi=[0., 0.], level=0):
    #     coords = utils.create_uvs(size)
    #     wi = np.tile(utils.spherical2dir(*wi), (size, size, 1))
    #     wo = np.tile(utils.spherical2dir(*wo), (size, size, 1))
    #     level = np.tile(level, (size, size, 1))
    #     return coords, level, wi, wo

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        if self.shuffle:
            return self.get_shuffled_sample()
        else:
            return self.get_sample(idx)


def calculate_mse(image1, image2):
    # 确保两个图像的形状相同
    if image1.shape != image2.shape:
        raise ValueError("The input images must have the same dimensions.")

    # 计算每个像素位置的差值的平方
    mse = np.mean((image1 - image2) ** 2)
    return mse


def render(
        decoder, decom, adapter,
        material_index, config: RepConfig,
        buf_path, out_path, device,
        u: torch.Tensor = None, v: torch.Tensor = None,
        view: torch.Tensor = None, light: torch.Tensor = None,
        output_brdf_value=False, multiplier=np.pi, use_brdf_sampling=False):
    uv = torch.cat([u, v], dim=-1).reshape(-1, 2)
    if view is not None:
        wi = view.clone().detach().to(device=device, dtype=torch.float32).reshape(-1, 2)
    if light is not None:
        wo = light.clone().detach().to(device=device, dtype=torch.float32).reshape(-1, 2)
    ''' render '''
    n = uv.shape[0]
    batch_size = min(n, 262144)
    with torch.no_grad():

        wi, wo = map(xy_to_xyz, [wi, wo])
        h, d = wiwo_xyz_to_hd_thetaphi(wi, wo)  ## [512, 512, 2]
        h, d = thetaphi_to_xyz(h)[:, :2], thetaphi_to_xyz(d)[:, :2]  ## [512, 512, 2]

        if __name__ == "__main__":
            print('start rendering.')
        final_output = None
        for start in range(0, n, batch_size):
            end = start + batch_size
            new_h = h[start:end]
            new_u, new_v, off = uv[start:end, 0], uv[start:end, 1], None
            latent = decom(material_index, new_h, new_u, new_v)  ## size: [bs, latent_size]
            output = decoder(torch.cat([d[start:end, ], latent], dim=-1))  ## size: [bs, 4+latent_size] -> [bs, 3]
            if adapter is not None:  ## [uv, 3, 3]
                output = adapter(output, material_index, new_u, new_v, radius=0)
            output = output * multiplier  ## the training dataset is divided by pi. so for validation, we need to *pi

            if final_output is None:
                final_output = output.reshape(-1, 3)
            else:
                final_output = torch.cat([final_output, output.reshape(-1, 3)], dim=0)

        final_output = final_output.cpu().numpy()

        res = 400
        final_output = final_output.reshape(res, res, 3)
        final_output = np.flipud(final_output)
        final_output = gamma_correction(final_output)
        return final_output
        # exr.write(final_output, out_path)
        # if __name__ == "__main__":
        #     print('saved into', out_path)



def get_sample(decoder, decom, adapter, config, ds: BTFDataset, idx: int, save_name='test'):
    with torch.no_grad():
        gt, co, l, wi, wo = ds.get_sample(idx) # [400,400,3], [400,400,2], [400,400,1], [400,400,3], [400,400,3]
        gt = gamma_correction(gt)
        co, l, wi, wo = (co.cuda(), l.cuda(), wi.cuda(), wo.cuda())
        u, v = co[..., 0], co[..., 1]
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)
        wi = wi[..., :2].reshape(-1, 2)
        wo = wo[..., :2].reshape(-1, 2)

        gt = gt.numpy()

        output = render(decoder, decom, adapter, -1, config,\
            None, None, detect_device(), u=u, v=v, view=wi, light=wo,\
            output_brdf_value=False, multiplier=1, use_brdf_sampling=False)

        # Align brightness
        output = align_brightness(gt, output)


        mse = calculate_mse(gt, output)
        # with open(f'/lizixuan/Biplane/torch/render/{save_name}_mse.txt', 'w') as f:
        #     f.write(str(mse))
        print(f'mse: {mse}')
        plt.imshow(gt)
        plt.savefig(f'/lizixuan/Biplane/torch/render/{save_name}_gt.png')
        plt.imshow(output)
        plt.savefig(f'/lizixuan/Biplane/torch/render/{save_name}.png')
        exr.write(gt, f'/lizixuan/Biplane/torch/render/{save_name}_gt.exr')
        exr.write(output, f'/lizixuan/Biplane/torch/render/{save_name}.exr')


def gt_gtResampled_compare():
    ds = BTFDataset("/lizixuan/Biplane/UBO2014BTF/wallpaper05_resampled_W400xH400_L151xV151.btf")
    ds1 = BTFDataset("/lizixuan/Biplane/UBO2014BTF/carpet01_resampled_W400xH400_L151xV151.btf")

    for i in range(5):
        # 5 different view dirs
        gt, co, l, wi, wo = ds.get_sample(i)
        gt1, co, l, wi, wo = ds1.get_sample(i)
        gt = gamma_correction(gt)
        gt1 = gamma_correction(gt1)
        exr.write(gt.numpy(), f'/lizixuan/Biplane/torch/render/carpet01_{i}_gt.exr')
        exr.write(gt1.numpy(), f'/lizixuan/Biplane/torch/render/carpet01_resampled_{i}_gt1.exr')


if __name__ == '__main__':
    mat_name = 'wallpaper05'
    ds = BTFDataset(f"/lizixuan/Biplane/UBO2014BTF/{mat_name}_resampled_W400xH400_L151xV151.btf")
    checkpoint_path = '/lizixuan/Biplane/torch/saved_model/#compress_only-offset_depth-ubo2014original_wallpaper05-Biplane-decoder1207-epoch-60-0516_181226/epoch-50/epoch-50.pth'
    checkpoint = torch.load(checkpoint_path)  ## dict
    config = checkpoint['config']
    decoder = checkpoint['decoder']
    decom = checkpoint['decom']
    adapter = checkpoint['adapter']
    print('checkpoint loaded.')
    get_sample(decoder, decom, adapter, config, ds, idx=1, save_name=mat_name)

