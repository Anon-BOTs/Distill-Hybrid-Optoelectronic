import os
import torch
import argparse




def main(path):
    root = os.path.dirname(path)
    new = {
        'state_dict' : {}
    }
    data = torch.load(path)
    for k, v in data['state_dict'].items():
        if 'backbone' in k or 'neck' in k:
            new_k = 'img_' + k
        elif 'query_head' in k:
            new_k = k.replace('query_head', 'img_roi_head')
        else:
            new_k = k


        new['state_dict'][new_k] = v
    torch.save(new, os.path.join(root, 'convert_ckpt.pth'))
    print('pth has been saved to {}'.format(os.path.join(root, 'convert_ckpt.pth')))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ckpt')
    args = arg_parser.parse_args()
    main(args.ckpt)