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
from dataloader_video_audio_EPIC_OSTTA_hac import EPICDOMAIN
import torch.nn.functional as F
from sklearn import metrics
from copy import deepcopy

source_all = [0, 1, 2, 3, 4, 5, 6, 7]#EPIC的8中动作类型

def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label) # treat ID as negative
    ood_indicator[label == -1] = 1  # treat OOD as positive

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in  = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out  = metrics.precision_recall_curve(ood_indicator, -conf)

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
# todo:调试TTA过程
def validate_one_step(clip, labels, spectrogram, clip_open, labels_open, spectrogram_open):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)#EPIC数据集视频数据（不区分来自train还是test） torch.Size([32, 3, 32, 224, 224])
        clip_open = clip_open['imgs'].cuda().squeeze(1)#HAC视频数据，测试模型在未知类别上的表现。torch.Size([32, 3, 32, 224, 224])
        clip = torch.cat((clip, clip_open), dim=0)#拼接->torch.Size([64, 3, 32, 224, 224])
    labels = labels.cuda()
    labels_open = labels_open.cuda()
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()#->torch.Size([32, 1, 257, 1004])
        spectrogram_open = spectrogram_open.unsqueeze(1).type(torch.FloatTensor).cuda()#->torch.Size([32, 1, 257, 1004])
        spectrogram = torch.cat((spectrogram, spectrogram_open), dim=0)#拼接->torch.Size([64, 1, 257, 1004])

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)  #获取layer3的输出特征
            v_feat = (x_slow.detach(), x_fast.detach())  
        
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)

    if args.online_adapt:# 如果启用在线适应
        if args.use_video:
            v_feat = model.module.backbone.get_predict(v_feat)#获取layer4的输出特征
            predict1, v_emd = model.module.cls_head(v_feat)#获取layer4的输出特征
        if args.use_audio:
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

        # OOD（Out-of-Distribution）检测和开放集识别中常用的设计，目的是让模型：
        # 对已知类别保持高置信度
        # 对未知类别保持低置信度（高熵）
        # 在不确定样本上允许不同模态有不同意见

        # ------------------------------------------------------------------------------------------------
        # paper: entropy loss to ensure diversity in predictions
        # loss_bal: 平衡不同类别的预测，鼓励模型对所有类别给出更均衡的预测
        # 利用自定义的mlp_cls分类器，将特征进行分类
        feat = torch.cat((v_emd, audio_emd), dim=1)#拼接视频和音频特征->torch.Size([2*batch, 2816])
        predict = mlp_cls(feat)#->torch.Size([2*batch, 8])
        p_sum = predict.softmax(dim=-1).sum(dim=-2)#对batch维度求和，得到每个类别的总概率->torch.Size([8])
        # 效果：当某些类别的预测概率过高时，loss_bal变小，总损失loss变大，鼓励模型对所有类别给出更均衡的预测，防止模型偏向于预测常见类别
        # 在开放集识别中的作用：帮助模型更好地处理类别不平衡问题，提高模型对未知类别的泛化能力，使模型不过分依赖已知类别的先验知识
        # 这种设计特别适合开放集识别任务，因为：需要平衡已知类别和未知类别的处理，防止模型过度依赖训练集中的类别分布，提高模型对新颖样本的敏感度
        loss_bal = -(p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum() 
        
        # ------------------------------------------------------------------------------------------------
        # loss_ra：控制预测的不确定性,鼓励模型对不确定的样本继续保持高熵（不确定性）
        # 当 pred_ent > args.tanh_alpha 时：tanh_w 为正
        # -pred_ent * tanh_w 为负，这使得损失减小，鼓励模型保持高熵（不确定性）
        pred_ent = normalized_prediction_entropy(predict)# 归一化确保熵值在[0,1]范围内.torch.Size([32])
        # maximizes pred_ent when pred_ent > α,此时，总损失更小 (i.e. when prediction confidence is low, indicating the sample is likely unknown) 
        # minimizes pred_ent when pred_ent < α (i.e. when prediction confidence is high, indicating the sample is likely known).
        tanh_w = torch.tanh(args.tanh_k * (pred_ent - args.tanh_alpha))#使用tanh函数生成权重。tanh_k=4.0控制权重曲线的陡峭程度，tanh_alpha=0.8是阈值，决定何时开始加权.torch.Size([32])
        loss_ra = -(pred_ent * tanh_w).mean()#鼓励模型对不确定的样本给出更确定的预测，权重tanh_w使模型更关注高熵（不确定）的样本
        
        # ------------------------------------------------------------------------------------------------
        # loss_a2d：计算跨模态一致性损失
        # loss_a2d_dis: 当模态间预测不一致时，l1_distance 更大，负号使得损失更小，这鼓励模型允许不同模态在不确定样本上有不同预测
        # 计算过程：将视频和音频预测转换为概率分布，计算L1距离（绝对差）作为不一致性度量，使用tanh_w加权，更关注不确定的样本
        l1_distance = torch.sum(torch.abs(nn.Softmax(dim=1)(predict1) - nn.Softmax(dim=1)(audio_predict)), dim=1)
        loss_a2d_dis = -(l1_distance * tanh_w).mean()

        # 预测熵协同
        # 鼓励不同模态在预测不确定性上保持一致
        pred_ent_v = normalized_prediction_entropy(predict1)#视频模态的预测熵
        pred_ent_f = normalized_prediction_entropy(audio_predict)#音频模态的预测熵
        loss_a2d_ent = -((pred_ent_v * tanh_w).mean() + (pred_ent_f * tanh_w).mean())/2#两个模态的损失取平均
        loss_a2d = loss_a2d_dis + loss_a2d_ent
        # 总损失
        loss = loss_ra + args.a2d_ratio * loss_a2d - args.marginal_ent_wei * loss_bal
        # 对每个batch进行参数更新，需要谨慎控制学习率（此项目取lr=2e-5），防止过度适应单个batch
        # 适应目标：
            # 通过loss_ra控制预测的不确定性
            # 通过loss_a2d确保多模态一致性
            # 通过loss_bal保持类别平衡
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

    # predict: mlp_cls的预测结果
    # predict1：视频模态的layer4预测结果
    # audio_predict：视频模态的预测结果
    # feat：拼接的视频layer4特征和音频特征
    return predict, predict1, audio_predict, feat

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
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
    parser.add_argument('--datapath_open', type=str, default='/path/to/HAC/',
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

    num_class = len(source_all)#8
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
        model.cls_head.fc_cls = nn.Linear(2304, num_class).cuda()#修改分类头
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

    if args.online_adapt:#进行调整参数
        params = list(mlp_cls.parameters())
        if args.use_video:#训练了layer4的参数
            params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
            model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) 
        if args.use_audio:
            params = params + list(audio_cls_model.parameters()) 
        
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    test_dataset = EPICDOMAIN(split='test', domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, datapath_open=args.datapath_open, use_video=args.use_video, use_audio=args.use_audio)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
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
                for (i, (clip, flow, labels, clip_open, flow_open, labels_open)) in enumerate(dataloaders[split]):
                    for _ in range(args.steps):
                        # predict: mlp_cls的预测结果
                        # predict1：视频模态的layer4预测结果
                        # audio_predict：视频模态的预测结果
                        # feat：拼接的视频layer4特征和音频特征
                        predict1, v_predict, audio_predict, feat = validate_one_step(clip, labels, flow, clip_open, labels_open, flow_open)
                    # 这里只用到了predict: mlp_cls的预测结果
                    output_list.append(predict1.cpu())
                    labels = torch.cat((labels, labels_open), dim=0)#HAC有7个动作类型。拼接后形状为：[2*batch,]
                    label_list.append(labels.cpu())
                    pbar.update()

                output_list = torch.cat(output_list)# 将所有批次的预测结果合并
                label_list = torch.cat(label_list)# 将所有批次的标签合并

                score = torch.softmax(output_list, dim=1)# 将输出转换为概率分布
                conf, pred = torch.max(score, dim=1)# 获取最大概率值(conf)和对应类别(pred)
                # 转换数据类型
                label_list = label_list.detach().numpy().astype(int)
                conf = conf.detach().numpy()
                pred = pred.detach().numpy().astype(int)
                # 计算已知类别的准确率，ood_label=8 表示标签为8的样本被视为未知类别，不参与准确率计算
                accuracy = acc(pred, label_list, ood_label=8)
                print("acc: ", accuracy)

                label_list[label_list==8] = -1
                ood_metrics = compute_all_metrics(conf, label_list)

                print("FPR@95: ", ood_metrics[0])# 95%召回率下的假正例率
                print("AUROC: ", ood_metrics[1])# ROC曲线下面积

                