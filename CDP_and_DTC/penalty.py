import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparateAngleLoss(nn.Module):
    def __init__(self, model, args):
        super(SeparateAngleLoss, self).__init__()
        self.last_w = []
        self.args = args
        self.model = model

    def forward(self, output, target):
        _cost = F.cross_entropy(output, target)
        pi = 3.1415926535
        _var_reg = 0
        _var_reg2 = 0
        _l2_reg = 0
        group_lasso = 0
        _prop_reg = 0

        if self.args.var != 0.0:
            cos1 = 0
            for weight_name, weight_data in self.model.named_parameters():
                if weight_data.ndim == 4:
                    # 这里先求出其范数值，然后按范数值大小得到索引，取前x%
                    filter_weight = weight_data

                    # 得到性质与weight一样的均值向量tensor
                    one_weight = torch.ones_like(filter_weight)
                    filter_mean = torch.mean(filter_weight, dim=[0, 2, 3])
                    filter_mean = filter_mean.unsqueeze(0)
                    filter_mean = filter_mean.unsqueeze(2)
                    filter_mean = filter_mean.unsqueeze(3)
                    filter_mean = one_weight * filter_mean
                    # 重新计算范数值，也就是模的大小
                    filter_norm = torch.sqrt(torch.sum(torch.square(filter_weight), dim=[1, 2, 3]))
                    filter_mean_norm = torch.sqrt(torch.sum(torch.square(filter_mean), dim=[1, 2, 3]))

                    # 向量与均值向量点积
                    filter_dot_mean = torch.sum(filter_weight * filter_mean, dim=[1, 2, 3])  # 3D
                    one_dim_mul = filter_norm * filter_mean_norm
                    # norm_mul = torch.ones_like(one_dim_mul)
                    # stripe_dot_mean_v2 = torch.ones_like(stripe_dot_mean_v1)
                    # for i in range(one_dim_mul.shape[0]):  # 由于可能在分母中模的乘积会出现0的可能导致loss为nan这里需要除零
                    # if one_dim_mul[i] > 0:
                    # norm_mul[i] = one_dim_mul[i]
                    # stripe_dot_mean_v2[i] = stripe_dot_mean_v1[i]

                    # 求出向量到均值向量的角余弦值
                    cos_filter = torch.div(filter_dot_mean, one_dim_mul + 0.00000001)  # 3D

                    # 余弦相似度 取值范围为[0, 1]
                    similarity_stripe = 1 - torch.div(torch.acos(cos_filter), pi)

                    cos1 += torch.sum(similarity_stripe) * self.args.var
            _var_reg = cos1

        if self.args.var2 != 0:

            cos2 = 0
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    FS_weight = weight_data
                    one_weight = torch.ones_like(FS_weight)
                    # FS_mean = torch.mean(FS_weight, dim=[0, 1, 2])
                    FS_mean = torch.mean(FS_weight, dim=0)  # 3*3
                    FS_mean = one_weight * FS_mean.unsqueeze(0)

                    # 计算范数值，也就是模的大小
                    FS_norm = torch.sqrt(torch.sum(torch.square(FS_weight), dim=[1, 2]))
                    FS_mean_norm = torch.sqrt(torch.sum(torch.square(FS_mean), dim=[1, 2]))

                    # 向量与均值向量点积
                    FS_dot_mean_v1 = torch.sum(FS_weight * FS_mean, dim=[1, 2])  # 1D
                    # print(FS_dot_mean_v1)
                    # 求出向量到均值向量的角余弦值
                    cos_FS = torch.div(FS_dot_mean_v1, FS_norm * FS_mean_norm + 0.0000001)  # 1D

                    # 余弦相似度 取值范围为[0, 1]
                    similarity_stripe = 1 - torch.div(torch.acos(cos_FS), pi)

                    cos2 += torch.sum(similarity_stripe) * self.args.var2
            _var_reg2 = cos2
            # print(_var_reg2)

        if self.args.l1_value != 0.0:
            loss_regularization = []
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    print(weight_data)
                    loss_regularization.append(torch.norm(weight_data, p=1))
            _l1_reg = self.args.l1_value * torch.sum(torch.stack(loss_regularization))

        if self.args.l2_value != 0.0:
            loss_regularization = []
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    loss_regularization.append(torch.norm(weight_data))
                    # loss_regularization.append(tf.math.reduce_sum(tf.math.square(p)))
            _l2_reg = self.args.l2_value * torch.sum(torch.stack(loss_regularization))

        if self.args.gl_a != 0.0:
            loss_gl = []
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    gg = torch.sum(torch.abs(weight_data), dim=[1, 2])
                    gg = torch.square(gg)
                    _group = torch.sum(torch.sqrt(torch.sum(gg, dim=0)))
                    loss_gl.append(_group * self.args.gl_a)
            group_lasso = torch.sum(torch.stack(loss_gl))

        if self.args.prop != 0.0:
            loss_gl = []
            _later_cnt = 0
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    _now_g = torch.sum(torch.sqrt(torch.square(torch.sum(torch.abs(weight_data), dim=[1, 2]))))
                    _now_l1 = torch.norm(weight_data, p=1)
                    FS_weight = weight_data
                    one_weight = torch.ones_like(FS_weight)
                    # FS_mean = torch.mean(FS_weight, dim=[0, 1, 2])
                    FS_mean = torch.mean(FS_weight, dim=0)  # 3*3
                    FS_mean = one_weight * FS_mean.unsqueeze(0)

                    # 计算范数值，也就是模的大小
                    FS_norm = torch.sqrt(torch.sum(torch.square(FS_weight), dim=[1, 2]))
                    FS_mean_norm = torch.sqrt(torch.sum(torch.square(FS_mean), dim=[1, 2]))

                    # 向量与均值向量点积
                    FS_dot_mean_v1 = torch.sum(FS_weight * FS_mean, dim=[1, 2])  # 1D
                    # print(FS_dot_mean_v1)
                    # 求出向量到均值向量的角余弦值
                    cos_FS = torch.div(FS_dot_mean_v1, FS_norm * FS_mean_norm + 0.0000001)  # 1D
                    # 余弦相似度 取值范围为[0, 1]
                    similarity_stripe = 1 - torch.div(torch.acos(cos_FS), pi)
                    similarity_stripe = torch.sum(similarity_stripe)
                    # 计算惩罚力度
                    if len(self.last_w):
                        _last_w = self.last_w[_later_cnt]
                        FS_weight = _last_w
                        one_weight = torch.ones_like(FS_weight)
                        # FS_mean = torch.mean(FS_weight, dim=[0, 1, 2])
                        FS_mean = torch.mean(FS_weight, dim=0)  # 3*3
                        FS_mean = one_weight * FS_mean.unsqueeze(0)

                        # 计算范数值，也就是模的大小
                        FS_norm = torch.sqrt(torch.sum(torch.square(FS_weight), dim=[1, 2]))
                        FS_mean_norm = torch.sqrt(torch.sum(torch.square(FS_mean), dim=[1, 2]))

                        # 向量与均值向量点积
                        FS_dot_mean_v1 = torch.sum(FS_weight * FS_mean, dim=[1, 2])  # 1D
                        # print(FS_dot_mean_v1)
                        # 求出向量到均值向量的角余弦值
                        cos_FS = torch.div(FS_dot_mean_v1, FS_norm * FS_mean_norm + 0.0000001)  # 1D

                        # 余弦相似度 取值范围为[0, 1]
                        similarity_stripe_last = 1 - torch.div(torch.acos(cos_FS), pi)
                        similarity_stripe_last = torch.sum(similarity_stripe_last)
                        a = (torch.div(similarity_stripe_last, similarity_stripe)) * self.args.prop
                        b = (torch.div(similarity_stripe, similarity_stripe_last)) * self.args.prop

                        loss_gl.append(b * _now_g + a * _now_l1)
                        # 第一次不做惩罚力度控制
                    else:
                        loss_gl.append(self.args.prop * _now_g + self.args.prop * _now_l1)

                    # 层计数
                    _later_cnt += 1

                # 保存新的权重
            self.last_w.clear()
            for weight_name, weight_data in self.model.named_parameters():
                if 'FilterSkeleton' in weight_name:
                    self.last_w.append(weight_data)

            _prop_reg = torch.sum(torch.stack(loss_gl))
        # print(_var_reg)

        return _cost + _var_reg + _var_reg2 + _l2_reg + group_lasso + _prop_reg
