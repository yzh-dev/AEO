from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_audio_HAC_OSTTA_epic import HACDOMAIN
import torch.nn.functional as F
from sklearn import metrics

source_all = [0, 1, 2, 3, 4, 5, 6]

def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr

def compute_all_metrics(conf, label):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)


    results = [fpr, auroc, aupr_in, aupr_out]

    return results

def normalized_prediction_entropy(logits):
    # Apply softmax to convert logits into probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Adding a small epsilon to avoid log(0)
    
    # Normalize entropy
    max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy



def validate_one_step(clip, labels, spectrogram, clip_open, labels_open, spectrogram_open):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
        clip_open = clip_open['imgs'].cuda().squeeze(1)
        clip = torch.cat((clip, clip_open), dim=0)
    labels = labels.cuda()
    labels_open = labels_open.cuda()
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
        spectrogram_open = spectrogram_open.unsqueeze(1).type(torch.FloatTensor).cuda()
        spectrogram = torch.cat((spectrogram, spectrogram_open), dim=0)

    predict1 = None
    audio_predict = None

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)  
            v_feat = (x_slow.detach(), x_fast.detach())  
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            

    if args.online_adapt:
        if args.use_video:
            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
        if args.use_audio:
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

        feat = torch.cat((v_emd, audio_emd), dim=1)

        predict = mlp_cls(feat)

        p_sum = predict.softmax(dim=-1).sum(dim=-2)
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()    

        pred_ent = normalized_prediction_entropy(predict)

        tanh_w = torch.tanh(args.tanh_k * (pred_ent - args.tanh_alpha))
        loss_ra = (-pred_ent * tanh_w).mean()

        l1_distance = torch.sum(torch.abs(nn.Softmax(dim=1)(predict1) - nn.Softmax(dim=1)(audio_predict)), dim=1)
        loss_a2d_dis = (-l1_distance * tanh_w).mean()

        pred_ent_v = normalized_prediction_entropy(predict1)
        pred_ent_f = normalized_prediction_entropy(audio_predict)
        loss_a2d_ent = ((-pred_ent_v * tanh_w).mean() + (-pred_ent_f * tanh_w).mean())/2

        loss_a2d = loss_a2d_dis + loss_a2d_ent

        loss = loss_ra + args.a2d_ratio * loss_a2d - args.marginal_ent_wei * loss_bal
            
        loss.backward()
        optim.step()
        optim.zero_grad()
    else:
        with torch.no_grad():
            if args.use_video:
                v_feat = model.module.backbone.get_predict(v_feat)
                predict1, v_emd = model.module.cls_head(v_feat)
            if args.use_audio:
                audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

            feat = torch.cat((v_emd, audio_emd), dim=1)

            predict = mlp_cls(feat)

    return predict, predict1, audio_predict

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
        
    def forward(self, feat):
        return self.enc_net(feat)

def acc(pred, label, ood_label=7):
    ind_pred = pred[label != ood_label]
    ind_label = label[label != ood_label]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='datapath')
    parser.add_argument('--datapath_open', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='datapath_open')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=32,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--use_single_pred', action='store_true')
    parser.add_argument('--online_adapt', action='store_true')
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument('--marginal_ent_wei', type=float, default=0.1,
                        help='marginal_ent_wei')
    parser.add_argument('--a2d_ratio', type=float, default=0.1,
                        help='a2d_ratio')
    parser.add_argument('--tanh_k', type=float, default=4.0,
                        help='tanh_k')
    parser.add_argument('--tanh_alpha', type=float, default=0.8,
                        help='tanh_alpha')
    parser.add_argument('--resume_file', type=str, default='/path/to/resume_file',
                        help='resume_file')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    num_class = len(source_all)
    input_dim = 0

    cfg = None
    cfg_flow = None

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log%s_TTA"%(args.source_domain)
    if args.use_video:
        log_name = log_name + '_video'
    if args.use_audio:
        log_name = log_name + '_audio'
    if args.use_single_pred:
        log_name = log_name + '_single_pred'

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)

    #resume_file = base_path_model + log_name + '.pt'
    resume_file = args.resume_file
    print("Resuming from ", resume_file)
    checkpoint = torch.load(resume_file)
    
    if args.use_video:
        model = init_recognizer(config_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, num_class).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])

        input_dim = input_dim + 2304

    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.fc = nn.Linear(512, num_class)
        audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
        audio_cls_model = audio_cls_model.cuda()

        input_dim = input_dim + 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()
    mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])

    if args.online_adapt:
        params = list(mlp_cls.parameters())
        if args.use_video:
            params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
            model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters())
        if args.use_audio:
            params = params + list(audio_cls_model.parameters()) 
        
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    test_dataset = HACDOMAIN(split='test', source=False, domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, datapath_open=args.datapath_open, use_video=args.use_video, use_audio=args.use_audio)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'test': test_dataloader}

    for epoch_i in range(args.nepochs):
        for split in ['test']:
            print("epoch_i: ", epoch_i)
            print(split)
            mlp_cls.train(args.online_adapt)
            if args.use_video:
                model.train(args.online_adapt)
            if args.use_audio:
                audio_cls_model.train(args.online_adapt)

            label_list, output_list = [], []
            with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                for (i, (clip, spectrogram, labels, clip_open, spectrogram_open, labels_open)) in enumerate(dataloaders[split]):
                    
                    for _ in range(args.steps):
                        predict1, v_predict, audio_predict = validate_one_step(clip, labels, spectrogram, clip_open, labels_open, spectrogram_open)

                    output_list.append(predict1.cpu())

                    labels = torch.cat((labels, labels_open), dim=0)
                    label_list.append(labels.cpu())
                    pbar.update()

                output_list = torch.cat(output_list)
                label_list = torch.cat(label_list)

                score = torch.softmax(output_list, dim=1)
                conf, pred = torch.max(score, dim=1)

                label_list = label_list.detach().numpy().astype(int)
                conf = conf.detach().numpy()
                pred = pred.detach().numpy().astype(int)
                    
                accuracy = acc(pred, label_list, ood_label=7)
                print("acc: ", accuracy)

                label_list[label_list==7] = -1
                ood_metrics = compute_all_metrics(conf, label_list)

                print("FPR@95: ", ood_metrics[0])
                print("AUROC: ", ood_metrics[1])