import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def compute_reweighing_weights(df, label_col="labels", sensitive_col="age_group", min_w=0.1, max_w=5.0):
    joint_counts = df.groupby([label_col, sensitive_col]).size().add(1)
    p_obs = joint_counts / len(df)
    p_y = df[label_col].value_counts(normalize=True)
    p_s = df[sensitive_col].value_counts(normalize=True)

    p_exp = {
        (y, s): p_y[y] * p_s[s]
        for y in p_y.index
        for s in p_s.index
    }

    weights = {
        (y, s): min(max(p_exp[(y, s)] / p_obs.get((y, s), 1e-4), min_w), max_w)
        for (y, s) in p_exp
    }
    return weights

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets, weights):
        loss = self.criterion(outputs, targets)
        return torch.mean(loss * weights)
    
def get_reweighted_training_setup(model, train_df, lr=0.01, wd=1e-3):
    weights_dict = compute_reweighing_weights(train_df, label_col="labels", sensitive_col="age_group")
    criterion = WeightedCrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    return criterion, optimizer, scheduler, weights_dict
    
class EqualizedOddsLoss(nn.Module):
    def __init__(self, num_classes, lambda_fairness=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_fairness = lambda_fairness
        self.num_classes = num_classes

    def forward(self, outputs, targets, sensitive_groups):
        ce_loss = self.ce(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        group_ids = torch.unique(sensitive_groups)

        tpr_per_group = {g.item(): [] for g in group_ids}
        fpr_per_group = {g.item(): [] for g in group_ids}

        for class_id in range(self.num_classes):
            for g in group_ids:
                g_mask = (sensitive_groups == g)
                y_true = (targets == class_id)[g_mask]
                y_pred = (preds == class_id)[g_mask]

                tpr = (y_pred & y_true).sum().float() / y_true.sum() if y_true.sum() > 0 else torch.tensor(0.0, device=outputs.device)
                fpr = (y_pred & ~y_true).sum().float() / (~y_true).sum() if (~y_true).sum() > 0 else torch.tensor(0.0, device=outputs.device)

                tpr_per_group[g.item()].append(tpr)
                fpr_per_group[g.item()].append(fpr)

        tpr_matrix = torch.stack([torch.stack(tpr_per_group[g.item()]) for g in group_ids], dim=0)
        fpr_matrix = torch.stack([torch.stack(fpr_per_group[g.item()]) for g in group_ids], dim=0)

        def calc_group_mse(metric_matrix):
            mean = metric_matrix.mean(dim=0, keepdim=True)
            return F.mse_loss(metric_matrix, mean.expand_as(metric_matrix))

        fairness_penalty = calc_group_mse(tpr_matrix) + calc_group_mse(fpr_matrix)
        return ce_loss + self.lambda_fairness * fairness_penalty

def get_equalized_odds_training_setup(model, lr=0.01, wd=1e-3, lambda_fairness=0.1):
    criterion = EqualizedOddsLoss(num_classes=7, lambda_fairness=lambda_fairness)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    return criterion, optimizer, scheduler

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
    

class AdversarialDebiasingModel(nn.Module):
    def __init__(self, num_classes=7, num_sensitive=5, lambda_adv=1.0):
        super(AdversarialDebiasingModel, self).__init__()
        self.lambda_adv = lambda_adv

        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.task_head = nn.Linear(num_ftrs, num_classes)

        self.adv_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, num_sensitive)
        )

    def forward(self, x):
        features = self.backbone(x)
        task_output = self.task_head(features)
        rev_features = GradientReversalFunction.apply(features, self.lambda_adv)
        adv_output = self.adv_head(rev_features)
        return task_output, adv_output
