from PIL import Image

from cg import CycleGANModel

if __name__ == '__main__':
    load_path = "./sample/a.jpeg"
    output_path = "./sample/cg_a.jpeg"

    pil_image = Image.open(load_path)

    cycle_gan = CycleGANModel()
    output_image = cycle_gan.inference(pil_image)

    output_image.save(output_path)
