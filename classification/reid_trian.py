import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as T
import torchvision
from config import get_config
import random
from PIL import Image
import einops
from models import build_model
import tqdm

# b, 1920, 12, 22

BATCH_SIZE = 1

EMBEDDING_MODEL_PTH = "C:\\Users\\ricky\\Downloads\\internimage_g_22kto1k_512.pth"
SAVE_DIR = "C:\\Users\\ricky\\Downloads\\reid_head"


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, n_heads=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_dim)
        self.attn = nn.MultiheadAttention(n_dim, num_heads=n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(n_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(n_dim, n_dim * 4),
            nn.SiLU(),
            nn.Linear(n_dim * 4, n_dim),
        )

    def forward(self, x):
        to_attn = self.ln1(x)
        attn = self.attn(to_attn, to_attn, to_attn)[0] + x
        return self.proj_out(self.ln2(attn)) + attn


class REIDHead(nn.Module):
    def __init__(self, n_dim_in, mlp_dim, enc_layers=1, seq_max=720, max_dpts=4):
        super().__init__()
        self.proj_down = nn.Linear(n_dim_in, mlp_dim)
        self.t_head = nn.Sequential(
            *(TransformerEncoderBlock(mlp_dim) for _ in range(enc_layers))
        )
        self.proj_out = nn.Linear(mlp_dim, 1)
        self.group = seq_max // (max_dpts + 1)
        self.register_parameter('positional_embedding', nn.Parameter(torch.randn(1, 720, mlp_dim)))
        # self.positional_embedding = nn.Parameter(torch.randn(1, 720, mlp_dim))

    def forward(self, x):
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', b=BATCH_SIZE)
        n = x.size(1)
        # n_datapts negates the need for padding
        x = einops.rearrange(x, 'b n c h w -> b (n h w) c')
        # print(self.proj_down(x).shape)
        x = self.proj_down(x) + self.positional_embedding
        enc = einops.rearrange(self.t_head(x), 'b (n s) c -> b n s c', n=n)
        return self.proj_out(enc[:, :, 0]).squeeze()


class REIDHeadDense(nn.Module):
    def __init__(self, mlp_dim, enc_layers = 2, max_dpts=4):
        super().__init__()
        self.register_parameter('positional_embedding', nn.Parameter(torch.randn(1, max_dpts + 1, mlp_dim)))
        self.t_head = nn.Sequential(
            *(TransformerEncoderBlock(mlp_dim) for _ in range(enc_layers))
        )
        self.proj_out = nn.Linear(mlp_dim, 1)
        self.max_dpts = max_dpts

    def forward(self, x):

        x = einops.rearrange(x, '(b n) c -> b n c', n=self.max_dpts + 1)
        x = x + self.positional_embedding
        enc = self.t_head(x)
        # print(self.proj_out(enc).shape)
        out = self.proj_out(enc).squeeze(-1).log_softmax(-1)
        # print(out)
        return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = "C:/Users/ricky/Downloads/crop_no_dup"
        self.labels_dir = "C:/Users/ricky/Downloads/Train/YOLO_darknet"
        self.reid_crops_dir = "C:/Users/ricky/Downloads/reid_crops"
        self.dir_list = os.listdir(self.dir)
        # self.transform = T.Compose([
        #     T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
        #     T.ToTensor(),
        #     T.Normalize(config.AUG.MEAN, config.AUG.STD)
        # ])
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = preprocessing = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):
        img_dir = os.path.join(self.dir, self.dir_list[index])
        dir_len = len(os.listdir(img_dir))
        t = torch.zeros((5, 3, 224,224))
        anchor = random.randint(0, dir_len - 1)
        for i in range(dir_len):
            path = os.path.join(self.dir, self.dir_list[index], str(i) + ".png")
            # print(path)
            img = Image.open(path)
            img = self.transform(img)
            t[i + 1] = img
            if i == anchor:
                cls = Image.open(path).info['class']
                crop_path = os.path.join(self.reid_crops_dir, str(cls))
                anchor_crop = os.path.join(crop_path,
                                           os.listdir(crop_path)[random.randint(0, len(os.listdir(crop_path)) - 1)])
                anchor_crop = Image.open(anchor_crop)
                anchor_crop = self.transform(anchor_crop)
                t[0] = anchor_crop
        return t, torch.Tensor([anchor + 1])

    def __len__(self):
        return len(self.dir_list) - 1


class IntermediateLayerGetter:
    def __init__(self, model, get_dense=False):
        if not get_dense:
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


def inf_loader():
    epoch = 0
    while True:
        epoch += 1
        loader = torch.utils.data.DataLoader(Dataset("C:/Users/ricky/Downloads/crop_no_dup"), batch_size=BATCH_SIZE,
                                             shuffle=True, pin_memory=True)
        for img, y in loader:
            yield epoch, img, y


loader = inf_loader()


def main():
    import argparse
    global config

    scaler = torch.cuda.amp.GradScaler()

    parser = argparse.ArgumentParser('Get Intermediate Layer Output')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
    parser.add_argument('--resume', help='resume from checkpoint')
    args = parser.parse_args()
    config = get_config(args)

    # emb_model = build_model(config)
    # checkpoint = torch.load(config.MODEL.RESUME, map_location='cuda:0')
    # emb_model.load_state_dict(checkpoint['model'])
    # emb_model = IntermediateLayerGetter(emb_model, get_dense=True)


    emb_model = torchvision.models.resnet50(pretrained=True)
    emb_model = emb_model.cuda()
    emb_model.fc = torch.nn.Identity()
    head = REIDHeadDense(2048).cuda()
    optim = torch.optim.AdamW(head.parameters(), lr=1e-4)
    loss_fn = nn.NLLLoss()
    with tqdm.tqdm(loader) as pbar:
        t, running_avg = 0, 0
        for iter, (epoch, image, y) in enumerate(pbar):
            image, y = image.cuda(), y.cuda()
            image = einops.rearrange(image, 'b n c h w -> (b n) c h w')
            with torch.no_grad():
                output = emb_model(image)
                # print(output.shape)
            # with torch.cuda.amp.autocast():
            y_hat = head(output)
            # y = F.one_hot(y.squeeze(-1).long(), num_classes=5).float()
            y = y.squeeze(-1).long()
            loss = loss_fn(y_hat, y)
            # print(y_hat)
                # print(y_hat)
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            loss.backward()
            optim.step()
            optim.zero_grad()
            t += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'epoch': epoch, 'avg_loss': running_avg})
            if iter % 50 == 0:
                running_avg = t / 50
                t = 0
                print(y_hat.exp())
            if iter % 1000 == 0:
                torch.save({
                    'model': head.state_dict(),
                    'optim': optim.state_dict(),
                    'scaler': scaler.state_dict(),
                }, f'{SAVE_DIR}/checkpoint_{iter}.pth')


if __name__ == '__main__':
    main()
