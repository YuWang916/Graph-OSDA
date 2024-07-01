import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import itertools

from network import GNN, DomainDiscriminator, Classifier, Attention, Classifier_OSDA


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def bce_loss(output, target): #from utils.py in OSDA-BP
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def test(data, encoder, ppmi_encoder, att_model, classifier, cache_name, k_class, threshold, mask=None, prin=None):
    # for model in models:
    #     model.eval()
    encoder.eval(), ppmi_encoder.eval(), att_model.eval(), classifier.eval()

    with torch.no_grad():
        embedding = encoder(data.x, data.edge_index, cache_name)
        ppmi_embedding = ppmi_encoder(data.x, data.edge_index, cache_name)
        feats = att_model([embedding, ppmi_embedding])
        logits = classifier(feats)

        # feats = encoder(data.x, data.edge_index, cache_name)
        # logits = classifier(feats)

        if cache_name == 'source':
            logits = logits[mask]
            source_preds = logits.argmax(dim=1)
            source_labels = data.y if mask is None else data.y[mask]
            # evaluate
            source_corrects = source_preds.eq(source_labels)
            accuracy = source_corrects.float().mean()
            return accuracy
        if cache_name == 'target':
            correct, correct_close = 0.0, 0.0
            label_t = data.y if mask is None else data.y[mask]

            class_list = [i for i in range(k_class)]
            class_list.append(k_class)

            per_class_num = np.zeros((k_class + 1))
            per_class_correct = np.zeros((k_class + 1)).astype(np.float32)
            per_class_correct_cls = np.zeros((k_class + 1)).astype(np.float32)

            out_t = F.softmax(logits)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()
            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = k_class

            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                # print(' the {}-class len is {}'.format(i, len(t_ind[0])))
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                if len(t_ind[0]) == 0:
                    per_class_num[i] = 1
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))

            per_class_acc = per_class_correct / per_class_num
            # close_p = float(per_class_correct_cls.sum() / per_class_num.sum())
            if prin is not None:
                print("per_class_acc is ", per_class_acc)
            # zhu compute common_acc, unknown_acc, and h-score
            common_acc = per_class_acc[:-1]
            common_acc = common_acc.mean()
            unknown_acc = per_class_acc[-1]
            h_score = 2 * common_acc * unknown_acc / (common_acc + unknown_acc)

        return per_class_acc.mean(), common_acc, unknown_acc, h_score


# the function for training and testing process
def main():
    # the configuration
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='dblpv7', help='source domain data')
    parser.add_argument("--target", type=str, default='acmv9', help='target domain data')
    parser.add_argument("--lblToRemove1", type=int, default=0, help='first label to remove for source domain data')
    parser.add_argument("--lblToRemove2", type=int, default=1, help='second label to remove for source domain data')
    parser.add_argument("--name", type=str, default='Ours')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--encoder_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # root_path = './results/' + args.name + '/S[' + args.source + ']-T[' + args.target + ']/seed[' + str(args.seed) + ']/'
    root_path = './results/' + args.name + '/S[' + args.source + ']-T[' + args.target + ']/removedLabels[' + str(args.lblToRemove1) + str(args.lblToRemove2) + ']/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    log_path = os.path.join(root_path, 'logs.txt')
    write_log(args, log_path)

    print('device is ', device)

    # check the source and target dataset
    print("source: {}, target: {}, seed: {}, encoder_dim: {}, removed_label: {},{}"
          .format(args.source, args.target, args.seed, args.encoder_dim, args.lblToRemove1, args.lblToRemove2))

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # process the source and target dataset
    s_dataset = DomainData("data/{}".format(args.source), name=args.source, domain='source', lblToRemove1= args.lblToRemove1, lblToRemove2= args.lblToRemove2)
    s_data = s_dataset[0]
    print("Processed PyG data: ")
    print(s_data)
    print(f'Number of source labels: {s_dataset.num_classes}')
    t_dataset = DomainData("data/{}".format(args.target), name=args.target, domain='target', lblToRemove1= args.lblToRemove1, lblToRemove2= args.lblToRemove2)
    t_data = t_dataset[0]
    print("Processed PyG data: ")
    print(t_data)
    print(f'Number of target labels: {t_dataset.num_classes}')
    s_data = s_data.cuda()
    t_data = t_data.cuda()

    # set the class information
    threshold = np.log(s_dataset.num_classes) / 2  # threshold to compare with entropy in target domain
    k_class = s_dataset.num_classes  # label for unknown class, e.g., 3 classes = 0,1,2; new unknow = 3
    num_class = s_dataset.num_classes + 1  # total number of classes, including "unknown"
    class_list = [i for i in range(s_dataset.num_classes)]

    CE = nn.CrossEntropyLoss().cuda()
    BCE = nn.BCELoss().cuda()
    MSE = nn.MSELoss().cuda()

    # build the model
    classifier = Classifier(encoder_dim=args.encoder_dim, num_classes=num_class).cuda()
    encoder = GNN(feats_dim=s_data.num_features, emb_dim=args.encoder_dim, base_model=None, type="gcn", ).cuda()
    ppmi_encoder = GNN(feats_dim=s_data.num_features, emb_dim=args.encoder_dim, base_model=encoder, type="ppmi", path_len=10).cuda()
    discriminator = DomainDiscriminator(encoder_dim=args.encoder_dim).cuda()
    att_model = Attention(args.encoder_dim).cuda()

    classifier, encoder, ppmi_encoder, discriminator, att_model = classifier.cuda(), encoder.cuda(), ppmi_encoder.cuda(), discriminator.cuda(), att_model.cuda()

    # models = [encoder, cls_model, domain_model] # Original UDAGCN
    # models = [encoder, cls_bp_model] # delete domain classifier and target classifier, add L_adv
    models = [encoder, classifier, discriminator]  # keep all three classifiers in UDAGCN, add L_adv

    models.extend([ppmi_encoder, att_model])
    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(params, lr=3e-3)

    # training
    best_t_acc = 0.0
    best_t_acc_k = 0.0
    best_t_acc_unk = 0.0
    best_epoch = 0.0
    best_t_hs = 0.0

    for epoch in range(args.epochs):
        for model in models:
            model.train()
        
        # get the features
        ## for source
        s_embedding = encoder(s_data.x, s_data.edge_index, 'source')
        s_ppmi_embedding = ppmi_encoder(s_data.x, s_data.edge_index, 'source')
        s_feats = att_model([s_embedding, s_ppmi_embedding])
        s_c_output = classifier(s_feats)
        ## for target
        t_embedding = encoder(t_data.x, t_data.edge_index, 'target')
        t_ppmi_embedding = ppmi_encoder(t_data.x, t_data.edge_index, 'target')
        t_feats = att_model([t_embedding, t_ppmi_embedding])
        t_c_output = classifier(t_feats)

        # compute the consistency between local and global
        t_feats_mat = torch.matmul(t_feats, t_feats.t()) / 0.05
        mask = torch.eye(t_feats_mat.size(0), t_feats_mat.size(0)).bool().cuda()
        t_feats_mat.masked_fill_(mask, -1 / 0.05)

        loss_nc = 0.05 * entropy(torch.cat([t_c_output, t_feats_mat], 1))

        # compute loss
        ## for classifier
        loss_ce = CE(s_c_output, s_data.y)

        ## for discriminator
        s_d_output = discriminator(s_feats)
        t_d_output = discriminator(t_feats)
        all_d_output = torch.cat((s_d_output, t_d_output), dim=0)
        d_label = torch.from_numpy(np.array([[1]] * s_d_output.size()[0] + [[0]] * t_d_output.size()[0])).float().cuda()
        loss_bce = BCE(all_d_output, d_label)

        # loss = loss_ce + loss_bce   # UDAGCN
        loss = loss_ce + loss_nc  # ERM

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch:[{}], loss_ce:{:.6f}, loss_nc:{:.6f}'
              .format(epoch, loss_ce.cpu().item(), loss_nc.cpu().item()))
        log_path = os.path.join(root_path, 'loss.txt')
        f = open(log_path, mode='a')
        f.write('epoch:[{}], loss_ce:{:.6f}, loss_nc:{:.6f}\n'
                .format(epoch, loss_ce.cpu().item(), loss_nc.cpu().item()))
        f.close()

        if epoch % 20 == 0:
            prin = True
        else:
            prin = None

        # do test
        # (data, encoder, ppmi_encoder, att_model, classifier, cache_name, unk_class, threshold, mask=None):
        s_acc = test(data=s_data, encoder=encoder, ppmi_encoder=ppmi_encoder, att_model=att_model, classifier=classifier,
                     cache_name='source', k_class=k_class, threshold=threshold, mask=s_data.test_mask)
        t_acc, t_acc_k, t_acc_unk, t_hs = test(data=t_data, encoder=encoder, ppmi_encoder=ppmi_encoder,
                                               att_model=att_model, classifier=classifier, cache_name='target',
                                               k_class=k_class, threshold=threshold, prin=prin)
        if t_hs > best_t_hs:
            best_t_acc, best_t_acc_k, best_t_acc_unk, best_t_hs, best_epoch = t_acc, t_acc_k, t_acc_unk, t_hs, epoch

        if epoch % 20 == 0:
            print("==>testing")
            print('epoch:[{}], s_acc:{:.4f}, t_acc:{:.4f}, t_acc_k:{:.4f}, t_acc_unk:{:.4f}, t_hs:{:.4f}'
                  .format(epoch, s_acc, t_acc, t_acc_k, t_acc_unk, t_hs))
        log_path = os.path.join(root_path, 'acc.txt')
        f = open(log_path, mode='a')
        f.write('epoch:[{}], s_acc:{:.4f}, t_acc:{:.4f}, t_acc_k:{:.4f}, t_acc_unk:{:.4f}, t_hs:{:.4f}\n'
                .format(epoch, s_acc, t_acc, t_acc_k, t_acc_unk, t_hs))
        f.close()

    print('Best_epoch:[{}], t_acc:{:.4f}, t_acc_k:{:.4f}, t_acc_unk:{:.4f}, t_hs:{:.4f}'
          .format(best_epoch, best_t_acc, best_t_acc_k, best_t_acc_unk, best_t_hs))
    log_path = os.path.join(root_path, 'acc.txt')
    f = open(log_path, mode='a')
    f.write('Best_epoch:[{}], t_acc:{:.4f}, t_acc_k:{:.4f}, t_acc_unk:{:.4f}, t_hs:{:.4f} \n'
            .format(best_epoch, best_t_acc, best_t_acc_k, best_t_acc_unk, best_t_hs))
    f.close()


if __name__ == "__main__":
    main()





