# main imports
import os, sys, argparse

# image processing imports
from PIL import Image
import ipfml.iqa.fr as fr

def main():

    parser = argparse.ArgumentParser(description="Compute .csv dataset file")

    parser.add_argument('--reference', type=str, help='Reference image')
    parser.add_argument('--reconstructed', type=str, help='Image to compare')
    parser.add_argument('--iqa', type=str, help='Image to compare', choices=['ssim', 'mse', 'rmse', 'mae', 'psnr'])
    args = parser.parse_args()

    param_reference = args.reference
    param_reconstructed = args.reconstructed
    param_iqa = args.iqa

    reference_image = Image.open(param_reference)
    reconstructed_image = Image.open(param_reconstructed)

    try:
        fr_iqa = getattr(fr, param_iqa)
    except AttributeError:
        raise NotImplementedError("FR IQA `{}` not implement `{}`".format(fr.__name__, param_iqa))

    print(fr_iqa(reference_image, reconstructed_image))


if __name__== "__main__":
    main()