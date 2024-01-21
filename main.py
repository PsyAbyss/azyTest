import os
import time
import torch
import argparse

from model import NASDataTheftDetection
from tqdm import tqdm
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

# parser.add_argument('--attn', type=str, default='cifa', help='attention used in encoder, options:[cifa, full]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--factor', type=float, default=0.6, help='sampling factor of query')
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# parser.add_argument('--exp_weight', type=float, default=1.0, help='exponential weight when calculate M')

parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--time_interval_emb', action='store_true', help='whether to add time interval to embedding')

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

def main():
    dataset = data_partition(args.dataset, args.time_span)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    tt = str(time.time()).split('.')[0]
    f = open(os.path.join(args.dataset + '_' + args.train_dir, tt+'_'+'log.txt'), 'w')
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.write('\n')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = NASDataTheftDetection(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)  # Xavier初始化
        except:
            pass

    model.train()  # enable model training

    # 继续之前的训练进度
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;
            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):  # range(num_batch):
            u, seq, time_intervals, pos, neg = [np.array(x) for x in sampler.next_batch()]
            pos_logits, neg_logits = model(u, seq, time_intervals, args.time_interval_emb, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)

            if t_test[0] >= best_result[0]:
                best_result[0] = t_test[0]
                best_epoch[0] = epoch
            if t_test[1] >= best_result[1]:
                best_result[1] = t_test[1]
                best_epoch[1] = epoch
            print('epoch:%d, time: %f(s), test (%.2f, %.2f), best (%.2f, %.2f) [%d, %d]'
                  % (
                  epoch, T, t_test[0] * 100, t_test[1] * 100, best_result[0] * 100, best_result[1] * 100, best_epoch[0],
                  best_epoch[1]))
            f.write('epoch:%d, time: %f(s), test (%.2f, %.2f), best (%.2f, %.2f) [%d, %d]'
                    % (epoch, T, t_test[0] * 100, t_test[1] * 100, best_result[0] * 100, best_result[1] * 100,
                       best_epoch[0], best_epoch[1]))
            f.write('\n')
            f.flush()
            t0 = time.time()
            model.train()

        # if epoch == args.num_epochs:
        #     folder = args.dataset + '_' + args.train_dir
        #     fname = 'CIFARec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #     fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        #     torch.save(model.state_dict(), os.path.join(folder, fname))

    f.write('Best Result: \n')
    f.write('NDCG@10: %.4f\tHR@10: %.4f\tEpoch: %d, %d'
            % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]) + '\n')
    f.close()
    sampler.close()
    print("Done")


if __name__ == '__main__':
    main()
