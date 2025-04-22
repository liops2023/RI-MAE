import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyAnchorLoss(nn.Module):
    """
    일반적인 Proxy Anchor Loss
    (Partial FC 없이 단독으로 사용할 수도 있음)
    """
    def __init__(self, nb_classes, sz_embed, margin=0.1, alpha=32):
        """
        Args:
            nb_classes (int): 클래스 수
            sz_embed (int): 임베딩 차원
            margin (float)
            alpha (float)
        """
        super(ProxyAnchorLoss, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.margin = margin
        self.alpha = alpha
        # 학습 가능한 Proxy 파라미터
        self.proxies = nn.Parameter(torch.Tensor(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): (B, sz_embed)
            labels (torch.Tensor): (B,)
        Returns:
            loss (torch.Tensor)
        """
        # Normalize embeddings & proxies
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)

        # cos similarity
        cos_sim = F.linear(embeddings_norm, proxies_norm)  # (B, nb_classes)

        # one-hot
        labels_one_hot = F.one_hot(labels, num_classes=self.nb_classes).float()
        pos_mask = labels_one_hot.bool()  # (B, nb_classes)
        neg_mask = ~pos_mask

        # pos_sim: (B,) -> 각 샘플마다 정답 클래스의 similarity
        pos_sim = cos_sim[pos_mask]

        # neg_sim: (B, nb_classes-1) -> 정답을 제외한 모든 클래스
        neg_sim = cos_sim[neg_mask].view(embeddings.size(0), -1)

        # hardest negative 안정적 계산
        with torch.no_grad():
            max_neg_sim, _ = torch.max(neg_sim, dim=1, keepdim=True)
            neg_exp_stable = torch.exp(self.alpha * (neg_sim - max_neg_sim))

        # hinge-style positive component
        pos_loss = F.relu(1 + self.margin - pos_sim)

        # negative loss
        neg_term_stable = torch.log(torch.sum(neg_exp_stable, dim=1)) / self.alpha + max_neg_sim.squeeze()
        neg_loss = F.relu(neg_term_stable + self.margin)

        num_pos = torch.sum(pos_mask).float()
        if num_pos == 0:
            pos_loss_sum = torch.tensor(0.0, device=embeddings.device)
        else:
            pos_loss_sum = torch.sum(self.alpha * pos_loss) / num_pos

        neg_loss_sum = torch.sum(self.alpha * neg_loss) / embeddings.size(0)
        loss = pos_loss_sum + neg_loss_sum
        return loss

