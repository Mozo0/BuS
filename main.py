from data.BUS_data import get_data
from models.SKNet import SKNet50
import pandas as pd

# -*- coding: utf-8 -*-
"""
VAE on mnist
"""
import glob

import torch
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import os
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
from torch.utils.data import Subset
import random
import time


# 绘制sample
def plot_sample(dataset, ind, file_name, if_score=None, is_prop=False):
    img, lbl = dataset[ind]
    img = img.squeeze()

    if img.dim() == 2:  # 灰度图
        plt.imshow(img.cpu().detach().numpy(), cmap='gray')
    elif img.dim() == 3:  # RGB图像
        plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy())

    if if_score is not None:
        if is_prop:
            plt.title(f"Proponent Influencing Label: {lbl}, Score: {if_score},\n File:{file_name}")
        else:
            plt.title(f"Opponent Influencing Label: {lbl}, Score: {if_score},\n File:{file_name}")
    else:
        plt.title(f"Influenced Label: {lbl},\n File:{file_name}")

    plt.show()


# 载入模型
def checkpoints_load_func(net, path):
    weights = torch.load(path)
    net.load_state_dict(weights["model_state_dict"])
    return 1.


CHECK_POINT_PRE = "skn_rgb"  # checkPoint 命名

LOAD_CHECK_POINT = True  # to load or not
TRAIN = False  # to train or not
PLT = False  # plot the test sample and the corresponding K highest IF score train sample(opponent/proponent respectively)
k = 10  # analyze the frequency of the First K highest IF score
MAX_IF_TEST_BATCH = 128  # run MAX_IF_TEST_BATCH test samples in one time
train_influencing_subset_size = -1  # number of train samples to compute IF score, chosen randomly from train_dataset, -1 to include all
test_influenced_indices = range(0)  # number of test samples to compute IF score, chosen sequencely from test_dataset, empty to include all

# 训练参数
epochs = 20  # 训练时期
batch_size = 16  # 每步训练样本数
learning_rate = 1e-5  # 学习率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
# 文件存在
os.chdir(os.path.dirname(__file__))

# 载入模型/新建模型
modelPath = 'CheckPoints/{}_{}.pth'.format(CHECK_POINT_PRE, epochs - 1)

model = SKNet50(nums_class=2, in_channel=3).to(device)
if LOAD_CHECK_POINT:
    try:
        model.load_state_dict(torch.load(modelPath)['model_state_dict'])
        print('[INFO] Load Model complete')
    except:
        pass
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备mnist数据集

train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = get_data(device, batch_size)

if train_influencing_subset_size == -1:
    train_influencing_subset_size = len(train_dataset)
if len(test_influenced_indices) == 0:
    test_influenced_indices = range(len(test_dataset))

# 训练
if TRAIN:
    # 训练及测试
    loss_history = {'train': [], 'eval': []}
    for epoch in range(epochs):
        # 训练
        model.train()
        # 每个epoch重置损失，设置进度条
        train_loss = 0
        train_nsample = 0
        t = tqdm(train_loader, desc=f'[train]epoch:{epoch}')
        for imgs, lbls in t:
            bs = imgs.shape[0]
            # 获取数据
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            # 模型运算
            feature = model(imgs)
            pred = torch.nn.functional.softmax(feature, dim=1)
            # 计算损失
            loss = criterion(pred, lbls.squeeze())
            # 反向传播、参数优化，重置
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 计算平均损失，设置进度条
            train_loss += loss.item()
            train_nsample += bs
            t.set_postfix({'loss': train_loss / train_nsample})
        # 每个epoch记录总损失
        loss_history['train'].append(train_loss / train_nsample)

        # 存储模型
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
        },
            os.path.join('CheckPoints', '{}_{}.pth'.format(CHECK_POINT_PRE, epoch)))

        # 测试
        model.eval()
        # 每个epoch重置损失，设置进度条
        test_loss = 0
        test_n_sample = 0
        e = tqdm(test_loader, desc=f'[eval]epoch:{epoch}')
        accuracy = 0
        for imgs, labels in e:
            bs = imgs.shape[0]
            # 获取数据
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 模型运算
            feature = model(imgs)
            pred = torch.nn.functional.softmax(feature, dim=1)
            # 计算损失
            loss = criterion(pred, labels.squeeze())
            # 计算平均损失，设置进度条
            test_loss += loss.item()
            test_n_sample += bs
            e.set_postfix({'loss': test_loss / test_n_sample})
            # 计算预测的类别
            pred_class = torch.argmax(pred, dim=1)
            # 计算分类准确率
            correct = (pred_class == labels).sum().item()
            total = labels.size(0)
            accuracy += correct / total
        # 每个epoch记录总损失
        loss_history['eval'].append(test_loss / test_n_sample)
        print("Eval acc:{}".format(accuracy / len(e)))

        # 显示每个epoch的loss变化
        plt.plot(range(epoch + 1), loss_history['train'])
        plt.plot(range(epoch + 1), loss_history['eval'])
        plt.show()

# 存档点整理
checkpoints_dir = "CheckPoints"
checkpoint_paths = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
final_checkpoint = os.path.join(checkpoints_dir, '{}_{}.pth'.format(CHECK_POINT_PRE, str(epochs - 1)))
checkpoints_load_func(model, final_checkpoint)

# 从训练集子集选举支持者与反对者
train_influencing_indices = random.sample(range(len(train_dataset)), train_influencing_subset_size)
train_influencing_subset = Subset(train_dataset, train_influencing_indices)

tracin_cp_fast = TracInCPFast(
    model=model,
    final_fc_layer=list(model.children())[-1],
    train_dataset=train_influencing_subset,
    checkpoints=checkpoint_paths,
    checkpoints_load_func=checkpoints_load_func,
    loss_fn=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    vectorize=False,
)

be_prop_freq = {}
be_oppo_freq = {}
ma_prop_freq = {}
ma_oppo_freq = {}

for l in range(0, len(test_influenced_indices), MAX_IF_TEST_BATCH):
    sub_test_influenced_indices = test_influenced_indices[l: (l + MAX_IF_TEST_BATCH)]
    start_time = time.time()
    sub_test_influenced_imgs = torch.stack([test_dataset[i][0] for i in sub_test_influenced_indices]).to(device)
    sub_test_influenced_true_labels = torch.Tensor([test_dataset[i][1] for i in sub_test_influenced_indices]).long().to(device)

    proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
        (sub_test_influenced_imgs, sub_test_influenced_true_labels), k=k, proponents=True
    )
    opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
        (sub_test_influenced_imgs, sub_test_influenced_true_labels), k=k, proponents=False
    )
    end_time = time.time()
    print(
        "Computed proponents / opponents from %d train candidates over a dataset of %d to %d test examples in %d seconds"
        % (len(train_influencing_subset), l, l+MAX_IF_TEST_BATCH, (end_time - start_time))
    )
    proponents_indices = proponents_indices.squeeze()
    proponents_influence_scores = proponents_influence_scores.squeeze()
    opponents_indices = opponents_indices.squeeze()
    opponents_influence_scores = opponents_influence_scores.squeeze()

    if PLT:
        for i, test_ind in enumerate(sub_test_influenced_indices):
            plot_sample(test_dataset, test_ind, test_dataset.get_file_name(test_ind))
            is_malignant = test_dataset[test_ind][1]
            for j in range(k):
                train_dataset_ind = train_influencing_indices[proponents_indices[i][j]]
                file_name = train_dataset.get_file_name(train_dataset_ind)
                plot_sample(train_dataset, train_dataset_ind,
                            file_name,
                            if_score=proponents_influence_scores[i][j], is_prop=True,
                            )
                if is_malignant == 0:
                    if file_name in be_prop_freq:
                        be_prop_freq[file_name] += 1
                    else:
                        be_prop_freq[file_name] = 1
                elif is_malignant == 1:
                    if file_name in ma_prop_freq:
                        ma_prop_freq[file_name] += 1
                    else:
                        ma_prop_freq[file_name] = 1
            for j in range(k):
                train_dataset_ind = train_influencing_indices[opponents_indices[i][j]]
                file_name = train_dataset.get_file_name(train_dataset_ind)
                plot_sample(train_dataset, train_dataset_ind,
                            file_name,
                            if_score=opponents_influence_scores[i][j], is_prop=False,
                            )
                if is_malignant == 0:
                    if file_name in be_oppo_freq:
                        be_oppo_freq[file_name] += 1
                    else:
                        be_oppo_freq[file_name] = 1
                elif is_malignant == 1:
                    if file_name in ma_oppo_freq:
                        ma_oppo_freq[file_name] += 1
                    else:
                        ma_oppo_freq[file_name] = 1
print("benign proponent:{}".format(be_prop_freq))
print("benign opponent:{}".format(be_oppo_freq))
print("malignant proponent:{}".format(ma_prop_freq))
print("malignant opponent:{}".format(ma_oppo_freq))
df = pd.DataFrame(list(be_prop_freq.items()), columns=["file_name", "frequency"])
df.to_excel("benign_proponent_frequency.xlsx", index=False)
df = pd.DataFrame(list(be_oppo_freq.items()), columns=["file_name", "frequency"])
df.to_excel("benign_opponent_frequency.xlsx", index=False)
df = pd.DataFrame(list(ma_prop_freq.items()), columns=["file_name", "frequency"])
df.to_excel("malignant_proponent_frequency.xlsx", index=False)
df = pd.DataFrame(list(ma_oppo_freq.items()), columns=["file_name", "frequency"])
df.to_excel("malignant_opponent_frequency.xlsx", index=False)
