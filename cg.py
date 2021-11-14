import PIL

import numpy as np
from torchvision.transforms import transforms

from networks import define_G
import torch

from PIL import Image


class CycleGANModel:
    # i extract transform condition from original cycle gan repo
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self):

        # i set up opt in here, i want to make this as small as possible.
        # if you setup different opt, than change here
        self.model = define_G(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG="resnet_9blocks",
            norm="instance",
            use_dropout=False,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=[]
        )
        self.load_path = 'weight/real_to_cartoon.pth'
        self.setup()

    def setup(self):
        """
            Load and print networks
        """
        self.load_networks()
        self.model.eval()
        self.print_networks(verbose=True)

    def extract_to_jit(self, jit_save_path: str = "./weight/jit_real_to_cartoon.pth"):
        """
            extract model weight to torch-jit format.
        """
        with torch.no_grad():
            m = torch.jit.script(self.model)
            m.save(jit_save_path)

    @staticmethod
    def convert_pil_to_tensor(pil_image: PIL.Image.Image) -> torch.Tensor:
        image: torch.Tensor = CycleGANModel.transform(pil_image)  # [3, 224, 224]
        return image.unsqueeze(dim=0)  # [1, 3, 224, 224]

    @staticmethod
    def convert_tensor_to_pil(tensor_image: torch.Tensor) -> PIL.Image.Image:
        # tensor_image : [1, 3, 256, 256]
        image_numpy = tensor_image[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        return Image.fromarray(image_numpy.astype(np.uint8))

    def inference(self, pil_image: PIL.Image.Image) -> PIL.Image.Image:
        tensor_image = self.convert_pil_to_tensor(pil_image)

        with torch.no_grad():
            output_tensor: torch.Tensor = self.model(tensor_image)  # [1, 3, 256, 256]

        return self.convert_tensor_to_pil(output_tensor)

    def load_networks(self):
        state_dict = torch.load(self.load_path, map_location="cpu")
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.model, key.split('.'))
        self.model.load_state_dict(state_dict)

    def print_networks(self, verbose=True):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')

        net = self.model
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
            Fix InstanceNorm checkpoints incompatibility (prior to 0.4)
            As i used InstanceNorm i saved this,
             but if you did not use instance norm, you can delete this function
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
