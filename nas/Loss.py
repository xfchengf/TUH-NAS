import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, d, e, f, ignore_lb=-1, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.d = d
        self.e = e
        self.f = f
        self.ignore_lb = ignore_lb
        self.focal_gamma = focal_gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def _prepare_logits_labels(self, logits, labels):
        n, c, h, w = logits.size()
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, c)
        labels = labels.view(-1).clone()  # Ensure labels are on the same device
        valid_mask = labels != self.ignore_lb
        return logits[valid_mask], labels[valid_mask]

    def dice_loss(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        num_classes = probs.size(1)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().permute(0, 1).unsqueeze(2).unsqueeze(3)
        probs = probs.unsqueeze(2).unsqueeze(3)

        intersection = (probs * labels_one_hot).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + labels_one_hot.sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()

    def focal_loss(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
        ce_loss = -labels_one_hot * torch.log(probs + 1e-5)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forward(self, logits, labels):
        logits, labels = self._prepare_logits_labels(logits, labels)
        ce_loss = self.ce_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        combined_loss = self.d * ce_loss + self.e * dice_loss + self.f * focal_loss
        return combined_loss
