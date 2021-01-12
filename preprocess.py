import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data')
    parser.add_argument('--out', default='./data')
    args = parser.parse_args()

    datasets = ['coco', 'f30k']
    splits = ['dev', 'test']

    for dataset in datasets:
        for split in splits:

            caps = []
            PATH = os.path.join(args.root, dataset, split+'_caps.txt')
            with open(PATH, 'r') as f:
                for i in f.readlines():
                    caps.append(i.split('\n')[0])

            json = []
            for ix, cap in enumerate(zip(*[iter(caps)] * 5)):
                json.append({'image_id': ix, 'refs': list(cap), 'test': ''})

            OUT_PATH = os.path.join
            json.dump(flickr_json, open('./data/flickr_test.json', 'w'))