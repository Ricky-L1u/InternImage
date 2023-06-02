import functools
import torch
from collections import OrderedDict
from models import build_model
import torchvision.transforms as T
from PIL import Image
import tqdm
import os
import argparse
from config import get_config

# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class IntermediateLayerGetter:
    def __init__(self, model):
        """Wraps a Pytorch module to get intermediate values

        Arguments:
            model {nn.module} -- The Pytorch module to call
            return_layers {dict} -- Dictionary with the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)

        Keyword Arguments:
            keep_output {bool} -- If True model_output contains the final model's output
            in the other case model_output is None (default: {True})

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are
            stored in a list.
        """
        model.forward_features = self.__forward_features.__get__(model)
        model.head = torch.nn.Identity()
        model = model.cuda()
        self._model = model
    @staticmethod
    def __forward_features(self, x):

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        return x

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

class ImgLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.png')]
        self.transform = T.Compose([
            T.Resize(config.DATA.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(config.AUG.MEAN, config.AUG.STD)
        ])
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list) - 1



def main(args, config):
    model = build_model(config)

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    mid_getter = IntermediateLayerGetter(model)
    loader = torch.utils.data.DataLoader(ImgLoader(args.img_dir),args.batch_size,pin_memory=True)
    t = []
    with torch.no_grad():
        for image in tqdm.tqdm(loader):
            image = image.cuda()
            output = mid_getter(image)
            t.append(output.clone().cpu())
            del output
    return torch.concat(t, dim=0)

    # torch.save(mid_outputs, args.img[:-3] + '.pth')

    # for k, v in mid_outputs.items():
    #     print(k, v.shape)

    # return mid_outputs, model_output




if __name__ == '__main__':


    parser = argparse.ArgumentParser('Get Intermediate Layer Output')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
    parser.add_argument('--img_dir', type=str, required=True, metavar="FILE", help='Path to img file')
    parser.add_argument("--keys", default=None, nargs='+', help="The intermediate layer's keys you want to save.")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()
    config = get_config(args)
    model_name = args.cfg.split('/')[-1].split('.')[0]
    model_output = main(args, config)


    torch.save(model_output, args.img_dir + model_name +'.pth')