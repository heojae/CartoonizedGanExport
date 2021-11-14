import torch
from PIL import Image
from cg import CycleGANModel

if __name__ == '__main__':
    jit_path = "./weight/jit_real_to_cartoon.pth"
    jit_model = torch.jit.load(jit_path)

    load_path = "./sample/a.jpeg"
    output_path = "./sample/jit_cg_a.jpeg"

    pil_image = Image.open(load_path)

    tensor_image = CycleGANModel.convert_pil_to_tensor(pil_image)
    with torch.no_grad():
        output_tensor = jit_model(tensor_image)
    output_pil_image = CycleGANModel.convert_tensor_to_pil(output_tensor)

    output_pil_image.save(output_path)
