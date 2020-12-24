import argparse
import tqdm
import json
import os

import numpy as np
import pandas as pd

from collections import defaultdict

class Metric:
    def __init__(self, args):
        self.args = args
        self.IMG_THRESHOLD = int(args.threshold)
        self.FUNCTION_MAP = {'hard': self.hard, 'soft': self.soft, 'softer': self.softer}
        self.TEXT_PER_IMG = 5
        self.TOP_K = 10
        self.RECALL_THRESHOLDS = [1, 5, 10]

        if args.dataset == 'coco':
            with open(os.path.join(args.dataset_path, args.dataset + '_test.json')) as fp:
                self.gt = json.load(fp)
        else:
            with open(os.path.join(args.dataset_path, args.dataset + '_dataset.json')) as fp:
                self.gt = json.load(fp)

        if args.metric == 'spice':
            metric = pd.read_csv(os.path.join(args.metric_path, args.dataset + '_' + args.metric + '.csv'), sep=',',
                                 header=None)
            metric = metric.to_numpy()
            if args.dataset == 'coco': self.metric = metric[:, :5000]
            if args.dataset == 'f30k': self.metric = metric[:, :1000]

        elif args.metric == 'cider':
            self.metric = np.load(os.path.join(args.metric_path, args.dataset+'_cider.npy'))

        filename = os.path.join(args.metric_path, 'sims_' + args.model_name + '_' + args.dataset + '_precomp.json')
        self.sims = json.load(open(filename, 'r'))

        self.get_intersection()

    def get_intersection(self):
        self.intersection = []
        for ix in tqdm.tqdm(range(len(self.sims)), leave=False):
            # GET MOST RELEVANT NON-GROUND TRUTH ITEMS
            intersection = []
            for i in range(self.TEXT_PER_IMG):
                index = self.TEXT_PER_IMG * ix + i
                idx = np.argsort(self.metric[index])[::-1]
                intersection.append({'indexes': idx[:10], 'scores': self.metric[index, idx[:10]]})

            count = defaultdict(int)
            for elm in intersection:
                for elm_ix, (spice_ix, sc) in enumerate(zip(elm['indexes'], elm['scores'])):
                    # count[spice_ix] += sc * (len(elm['indexes']) - elm_ix)
                    count[spice_ix] += sc

            new_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
            pop_ix = [i for i, j in enumerate(new_count) if j[0] == ix][0]
            if not self.args.include_anns:
                new_count.pop(pop_ix)
            self.intersection.append(new_count)

    def build_ranks(self):
        ranks = {}
        for sc in self.args.score:
            ranks[sc] = []
        return ranks

    def calculate_ranks(self, ranks, score_type, gt_ranks=None, modality='i2t'):
        ranks = np.array(ranks)
        # TODO: THERE IS A BUG; when IMG_THRESHOLD=1, 'hard', 'recall', 't2i'
        if score_type == 'hard' and args.recall_type == 'recall' and len(ranks.shape)>1:
            if modality == 'i2t':
                # This constant is the amount of relevant items
                num_relevant = self.TEXT_PER_IMG * self.IMG_THRESHOLD
            if modality == 't2i':
                num_relevant = self.IMG_THRESHOLD

            r1 = sum([sum(r[:1])/num_relevant for r in ranks])/len(ranks) * 100
            r5 = sum([sum(r[:5])/num_relevant for r in ranks])/len(ranks) * 100
            r10 = sum([sum(r[:10])/num_relevant for r in ranks])/len(ranks) * 100
            print("Hard score with Recall, R@1: {}, R@5: {}, R@10: {}".format(r1, r5, r10))

        elif score_type == 'hard':
            r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
            print("Hard score, R@1: {}, R@5: {}, R@10: {}".format(r1, r5, r10))

        elif score_type == 'soft':

            r1 = sum([sum(r[:1]) for r in ranks])/len(ranks) * 100
            r5 = sum([sum(r[:5]) for r in ranks])/len(ranks) * 100
            r10 = sum([sum(r[:10]) for r in ranks])/len(ranks) * 100

            print("Soft score with Recall, R@1: {}, R@5: {}, R@10: {}".format(r1, r5, r10))

        elif score_type == 'softer':
            r1 = 100.0 * ranks[:, :1].mean(axis=1).mean(axis=0) / (gt_ranks[:, :1].mean(axis=1).mean(axis=0))
            r5 = 100.0 * ranks[:, :5].mean(axis=1).mean(axis=0) / (gt_ranks[:, :5].mean(axis=1).mean(axis=0))
            r10 = 100.0 * ranks[:, :10].mean(axis=1).mean(axis=0) / (gt_ranks[:, :10].mean(axis=1).mean(axis=0))
            print("Softer score, R@1: {}, R@5: {}, R@10: {}".format(r1, r5, r10))

    def recall(self, ix, modality):
        if modality == 'i2t':
            relevant_items = self.intersection[ix][:self.IMG_THRESHOLD]
            relevant_indexes = []
            for item in relevant_items:
                relevant_indexes.extend(list(range(item[0] * self.TEXT_PER_IMG,
                                                   item[0] * self.TEXT_PER_IMG + self.TEXT_PER_IMG)))
        elif modality == 't2i':
            relevant_items = self.intersection[ix // self.TEXT_PER_IMG][:self.IMG_THRESHOLD]
            relevant_indexes = [item[0] for item in relevant_items]

        return relevant_indexes

    def i2t(self):
        ranks = self.build_ranks()
        gt_ranks = np.zeros((len(self.sims), 10))

        for ix, sim in enumerate(tqdm.tqdm(self.sims, leave=False)):
            inds = np.argsort(sim)[::-1]

            if not self.args.include_anns:
                # Remove the index from the similarity
                gt = list(range(self.TEXT_PER_IMG * ix, self.TEXT_PER_IMG * ix + self.TEXT_PER_IMG, 1))
                # 100x faster
                inds = inds[~np.isin(inds, gt)]
                # More readable
                # inds = np.array([i for i in inds if i not in gt])

            for sc in args.score:
                self.FUNCTION_MAP[sc](ix, inds, ranks[sc], 'i2t', gt_ranks)

        for sc in args.score:
            self.calculate_ranks(ranks[sc], sc, gt_ranks, modality='i2t')

    def t2i(self):
        sims = np.array(self.sims).T
        ranks = self.build_ranks()
        gt_ranks = np.zeros((len(sims), 10))

        for ix, sim in enumerate(tqdm.tqdm(sims, leave=False)):
            inds = np.argsort(sim)[::-1]
            if not self.args.include_anns:
                inds = inds[~np.isin(inds, [ix // self.TEXT_PER_IMG])]
                # inds = np.array([i for i in inds if i != ix // self.TEXT_PER_IMG])
            for sc in args.score:
                self.FUNCTION_MAP[sc](ix, inds, ranks[sc], 't2i', gt_ranks)

        for sc in args.score:
            self.calculate_ranks(ranks[sc], sc, gt_ranks, modality='t2i')

    def hard(self, ix, inds, ranks, modality='i2t', gt=None):
        if modality == 'i2t':
            if args.recall_type == 'vse_recall':
                rank = 1e20
                for c in self.intersection[ix][:self.IMG_THRESHOLD]:
                    for i in range(self.TEXT_PER_IMG * c[0], self.TEXT_PER_IMG * c[0] + self.TEXT_PER_IMG, 1):
                        tmp = np.where(inds == i)[0][0]
                        if tmp < rank:
                            rank = tmp

                ranks.append(rank)

            elif args.recall_type == 'recall':
                relevant_indexes = self.recall(ix, modality)
                rel = [1 if i in relevant_indexes else 0 for i in inds[:10]]
                ranks.append(rel)

        elif modality == 't2i':
            if args.recall_type == 'vse_recall' or self.IMG_THRESHOLD == 1:
                rank = 1e20
                for c in self.intersection[ix // self.TEXT_PER_IMG][:self.IMG_THRESHOLD]:
                    tmp = np.where(inds == c[0])[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks.append(rank)

            elif args.recall_type == 'recall' and self.IMG_THRESHOLD >= 2:
                relevant_indexes = self.recall(ix, modality)
                rel = [1 if i in relevant_indexes else 0 for i in inds[:10]]
                ranks.append(rel)

    def soft(self, ix, inds, ranks, modality='i2t', gt=None):

        relevant_indexes = self.recall(ix, modality)

        if modality == 'i2t':
            # TODO: Check if correct
            constant = sum(self.metric[relevant_indexes, ix]) + 1e-20
            rel = [self.metric[i, ix]/constant if i in relevant_indexes else 0 for i in inds[:10]]
            # rel = [self.metric[i, ix] if i in relevant_indexes else 0 for i in inds[:10]]
            ranks.append(rel)

        elif modality == 't2i':
            constant = sum(self.metric[ix, relevant_indexes]) + 1e-20
            rel = [self.metric[ix, i] / constant if i in relevant_indexes else 0 for i in inds[:10]]
            # rel = [self.metric[ix, i] if i in relevant_indexes else 0 for i in inds[:10]]
            ranks.append(rel)

    def softer(self, ix, inds, ranks, modality='i2t', gt_ranks=None):
        if modality == 'i2t':
            ranks.append(self.metric[inds[:10]][:, ix])
            # For normalization
            gt = list(range(self.TEXT_PER_IMG * ix, self.TEXT_PER_IMG * ix + self.TEXT_PER_IMG, 1))
            inds_metric = np.argsort(self.metric[:, ix])[::-1]
            if not self.args.include_anns:
                inds_metric = inds_metric[~np.isin(inds_metric, gt)]
                # inds_metric = np.array([i for i in inds_metric if i not in gt])
            gt_ranks[ix, :] = self.metric[inds_metric[:10]][:, ix]

        elif modality == 't2i':
            ranks.append(self.metric[:, inds[:10]][ix, :])
            # For normalization
            inds_metric = np.argsort(self.metric[ix, :])[::-1]
            if not self.args.include_anns:
                inds_metric = inds_metric[~np.isin(inds_metric, [ix // self.TEXT_PER_IMG])]
                # inds_metric = np.array([i for i in inds_metric if i !=ix//self.TEXT_PER_IMG])
            gt_ranks[ix, :] = self.metric[:, inds_metric[:10]][ix, :]

    def compute_metrics(self):
        print("\nModel name:{},\n"
              "Dataset: {},\n"
              "Recall Type: {},\n"
              "Metric:{},\n".format(self.args.model_name, self.args.dataset, self.args.recall_type, self.args.metric))
        print("####I2T#####")
        self.i2t()
        print("####T2I#####")
        self.t2i()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data', help='ground truth data path')

    parser.add_argument('--metric_path', type=str, default='./out', help='the path that has metrics and model output')

    parser.add_argument('--dataset', type=str, default='f30k', help='which dataset to use, options are: coco, f30k')

    parser.add_argument('--metric', type=str, default='spice',
                        help='which image captioning metric to use, options are: cider, spice')

    parser.add_argument('--recall_type', type=str, default='recall', help='Options are recall and vse_recall')

    parser.add_argument('--score', default=['softer'], nargs="+",
                        help='which scoring method to use, options are: hard, soft, softer')

    parser.add_argument('--model_name', type=str, default='VSRN',
                        help='which model to use, options are: VSEPP, SCAN, VSRN, CVSE')

    parser.add_argument('--threshold', type=int, default=1,
                        help='Threshold of number of relevant samples to compute metrics, options are: 1,2,3')

    parser.add_argument('--include_anns', type=bool, default=True,
                        help='Include human annotations to define relevant items, options are: True, False')


    args = parser.parse_args()
    M = Metric(args)
    print("\n ... LOADING DATA ...\n")
    M.compute_metrics()

