import argparse
from util import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--img_path', type=str, default='data/302008.jpg', help='path to the image')
    parser.add_argument('--sigma', type=float, default='1.4', help='intensity of gaussian blur')
    parser.add_argument('--LTR', type=float, default=0.15, help='Low Threshold Ratio for Double Threshold Hysterisis')
    parser.add_argument('--HTR', type=float, default=0.23, help='High Threshold Ratio for Double Threshold Hysterisis')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    I = read_image(img_path=args.img_path)
    I_gauss = gaussblur(I, sigma=args.sigma)
    final_image = canny_edge(I, args.sigma, args.HTR, args.LTR)

    show_sidebyside([I, I_gauss, final_image], [f'original', f'blur(sigma={args.sigma})', 'final'])