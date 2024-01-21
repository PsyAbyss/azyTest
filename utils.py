import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)  # 这个user无数据时重新随机选一个

        seq = np.zeros([maxlen], dtype=np.int32)  # 训练集seq除去最后一个，前面补0
        pos = np.zeros([maxlen], dtype=np.int32)  # seq向左平移一位，最后一位为训练集seq最后一位（即正样本，pos是seq对应位置之前序列的预测值）
        neg = np.zeros([maxlen], dtype=np.int32)  # seq和pos有值的位置随机取负样本
        nxt = user_train[user][-1][0]
        idx = maxlen - 1
        time_intervals = np.zeros([maxlen], dtype=np.int32)

        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_intervals[idx] = i[2]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break

        return (user, seq, time_intervals, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())  # 每个sample为size=4的tuple

        result_queue.put(zip(*one_batch))  # zip(*zipped)操作，相当于将batch_size个sample分别拆开后重组为user, seq, pos, neg四个list


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def timeSlice(time_set):  # 原时间戳-最小时间戳
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map, time_span):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()  # user和item重新编号从1开始
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])  # {1: [[3186, 978300019.0], [1270, 978300055.0], [1721, 978300055.0]], 2:[[],[]]}

    User_res = dict()  # user和item替换为新编号，且时间戳替换为（原时间戳-最小时间戳）
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]],items))  # {1: [[2758, 21596087], [1069, 21596123], [1445, 21596123]], 2:[[],[]]}

    time_max = set()
    time_intervals = list()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set() # 去除0后的时间差set，只为了计算time_scale
        time_intervals = list() # 每两两时间戳的时间差列表
        for i in range(len(time_list) - 1):
            if i == 0:
                time_intervals.append(0)
            else:
                time_intervals.append(time_list[i] - time_list[i-1])

            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        time_intervals.append(time_list[len(time_list)-1] - time_list[len(time_list)-2]) #加上最后一项
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff) # 计算最小时间间隔 shortest time interval (other than 0)
        time_min = min(time_list)  # 最小时间戳 这里每个时间戳先减去最小时间戳的做法论文中没有涉及 意义不大 以为后面还要计算时间差
        # 这里是先除以time_scale再在后面计算时间差
        User_res[user] = list(map(lambda x: [x[0],
                                             int(round((x[1] - time_min) / time_scale) + 1),
                                             min(int(round(time_intervals[items.index(x)] / time_scale)), time_span)
                                             ],items))
        # User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)],items)) # divide by the shortest time interval
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)  # 最大时间间隔

# train/val/test data generation
def data_partition(fname, time_span):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')  # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        # if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
        #     continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map, time_span)

    for user in User:  # 序列的最后一个item为测试集，倒数第二为验证集，剩余为训练集
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_interval = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        time_interval[idx] = valid[u][0][2]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_interval[idx] = i[2]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0],train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_interval], item_idx]])
        predictions = -model.predict(np.array([u]), np.array([seq]), np.array([time_interval]), args.time_interval_emb, np.array(item_idx))
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
