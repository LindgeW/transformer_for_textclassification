from dataloader.Dataloader import batch_iter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time


# 分类器
class Classifier(object):
    def __init__(self, model, args, vocab):
        assert isinstance(model, nn.Module)
        self._model = model
        self._args = args
        self._vocab = vocab

    # 打印模型结构
    def summary(self):
        print(self._model)

    # 训练（学习）
    def train(self, train_data, dev_data):

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                               lr=self._args.lr,
                               betas=(0.9, 0.98),
                               eps=1e-9)

        # 自定义学习速率调整策略
        # 注：epoch从0开始算起
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
        #                                                     factor=0.1,
        #                                                     patience=3,
        #                                                     verbose=True,
        #                                                     min_lr=1e-5)

        # 每经过T_max个epoch之后重新调整lr(先下降，最小到eta_min，后上升)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-4)
        # 每经过5个epoch，学习速率变为原来的gamma倍
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        # 在epoch为8或15轮时，学习速率变为原来的gamma倍
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 15], gamma=0.2)

        # optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self._model.parameters()),
        #                            lr=1.0,
        #                            weight_decay=self._args.weight_decay)

        # optimizer_sgd = optim.SGD(filter(lambda p: p.requires_grad, self._model.parameters()),
        #                           lr=self._args.lr,
        #                           momentum=0.9,
        #                           weight_decay=self._args.weight_decay,
        #                           nesterov=True
        #                           )

        # 训练多轮
        train_loss_lst, train_acc_lst = [], []
        dev_loss_lst, dev_acc_lst = [], []
        for ep in range(self._args.epoch):
            start = time.time()
            train_acc, train_loss = 0, 0
            self._model.train()  # self.training置成True

            lr_scheduler.step()

            # 按照batch进行训练
            for batch in batch_iter(train_data, self._args.batch_size, self._vocab, device=self._args.device):
                # 梯度初始化（置0）
                self._model.zero_grad()
                # 前向传播（数据喂给模型）
                pred = self._model(batch.src)
                # 反向传播BP，计算误差
                loss = self._calc_loss(pred, batch.target)
                train_loss += loss.data.item()
                train_acc += self._calc_acc(pred, batch.target)
                loss.backward()
                # 用梯度去更新模型参数
                optimizer.step()

            print('lr: ', lr_scheduler.get_lr())

            # 开发集调参
            dev_loss, dev_acc = self._validate(dev_data)
            end = time.time()

            train_acc /= len(train_data)
            train_acc_lst.append(train_acc)
            train_loss_lst.append(train_loss)
            dev_acc_lst.append(dev_acc)
            dev_loss_lst.append(dev_loss)

            print('[Epoch %d] train loss: %.3f, train acc: %.3f' % (ep, train_loss, train_acc))
            print('dev loss: %.3f, dev acc: %.3f' % (dev_loss, dev_acc))
            print('time: %.3f s' % (end-start))

            # Epoch每经过patience轮，如果开发集acc没有上升或
            # 开发集Loss没有下降，则停止训练
            if (ep + 1) % self._args.patience == 0 and dev_acc_lst[ep] < dev_acc_lst[ep - self._args.patience + 1]:
                break

    # 验证（调参）
    def _validate(self, dev_data):
        dev_acc, dev_loss = 0, 0
        self._model.eval()  # self.training置成False
        with torch.no_grad():
            for batch in batch_iter(dev_data, self._args.batch_size, self._vocab, device=self._args.device):
                pred = self._model(batch.src)
                loss = self._calc_loss(pred, batch.target)
                dev_loss += loss.data.item()
                dev_acc += self._calc_acc(pred, batch.target)

            dev_acc /= len(dev_data)
        return dev_loss, dev_acc

    # 评估
    def evaluate(self, test_data):
        test_acc, test_loss = 0, 0
        self._model.eval()  # self.training置成False
        for batch in batch_iter(test_data, self._args.batch_size, self._vocab, device=self._args.device):
            pred = self._model(batch.src)
            loss = self._calc_loss(pred, batch.target)
            test_loss += loss.data.item()
            test_acc += self._calc_acc(pred, batch.target)

        test_acc /= len(test_data)
        print('------- test loss:%.3f test acc:%.3f -------' % (test_loss, test_acc))
        return test_loss, test_acc

    # 计算准确率
    def _calc_acc(self, pred, target):
        '''
        :param pred: [batch_size, label_size]
        :param target: [batch, ]
        :return: 预测正确的item的数目
        '''
        return torch.eq(torch.argmax(pred, dim=1), target).sum().item()

    # 计算误差
    def _calc_loss(self, pred, target, label_smoothing=True, eps=0.1):
        '''
        :param pred: 网络输出
        :param target: 标签值（离散值 0, 1, 2 ,3, ..., nb_class-1）
        :param label_smoothing: 标签平滑
        :param eps: label smoothing value
        :return:
        '''
        if label_smoothing:
            nb_class = pred.size(-1)
            # 将索引标签转换成one-hot形式
            one_hot_tgt = torch.zeros_like(pred).scatter_(1, target.unsqueeze(-1), 1)  # dim index value
            # 平滑标签：1, 0 -> 1-eps, eps
            smooth_target = (1 - eps) * one_hot_tgt + eps / (1 - nb_class)
            ce = -1 * smooth_target * F.log_softmax(pred, dim=1)
            return torch.mean(ce, 0).sum()

        else:
            # CrossEntropyLoss = LogSoftmax + NLLLoss
            return F.cross_entropy(pred, target)

    # 保存模型
    def save(self, path):
        with open(path, 'wb') as fw:
            torch.save(self._model, fw)
            # torch.load("", map_location='cpu')
