import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

"""
PartialFC_V2 예시
- 분산 환경에서, Sample Rate < 1 이면 일부 클래스의 Proxy만 골라서 업데이트
- Proxy Anchor류의 손실과 함께 사용 가능
"""

class DistCrossEntropyFunc(torch.autograd.Function):
    """
    분산 환경에서 Cross Entropy를 계산하기 위한 예시 함수
    (필요 시 Proxy Anchor가 아니라 일반 Softmax를 사용하려면)
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        # 분산환경에서 모든 rank에서 최대값 동기화
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        dist.all_reduce(max_logits, op=dist.ReduceOp.MAX)
        logits = logits - max_logits  # for numerical stability
        
        # exp
        logits_exp = torch.exp(logits)
        sum_logits_exp = torch.sum(logits_exp, dim=1, keepdim=True)
        dist.all_reduce(sum_logits_exp, op=dist.ReduceOp.SUM)
        
        probs = logits_exp / sum_logits_exp
        # valid index
        valid_idx = (label != -1).nonzero(as_tuple=True)[0]
        
        # gather prob of target
        log_preds = torch.zeros_like(label, dtype=logits.dtype, device=logits.device)
        log_preds[valid_idx] = torch.log(probs[valid_idx, label[valid_idx]])
        
        # loss = -mean of log_preds
        total_loss = -torch.sum(log_preds)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        num_valid = valid_idx.numel()
        dist.all_reduce(num_valid, op=dist.ReduceOp.SUM)
        
        ctx.save_for_backward(probs, label, torch.tensor(num_valid, device=probs.device))
        return total_loss / (num_valid + 1e-6)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: scalar
        probs, label, num_valid = ctx.saved_tensors
        batch_size = probs.size(0)
        
        grad_logits = probs.clone()  # (B, C)
        valid_idx = (label != -1).nonzero(as_tuple=True)[0]
        
        # one-hot sub
        grad_logits[valid_idx, label[valid_idx]] -= 1.0
        grad_logits /= (num_valid.item() + 1e-6)
        
        grad_logits *= grad_output
        return grad_logits, None


class DistCrossEntropy(nn.Module):
    """
    Partial FC 환경에서 임시로 cross entropy를 계산할 때 사용
    (Proxy Anchor를 안 쓰고 일반적인 CE로 학습할 경우 사용)
    """
    def forward(self, logits, label):
        return DistCrossEntropyFunc.apply(logits, label)


class AllGatherFunc(torch.autograd.Function):
    """
    분산환경에서 all_gather를 수행하면서 backprop까지 연결
    """
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        dist.all_gather(list(gather_list), tensor)
        return tuple(gather_list)
    
    @staticmethod
    def backward(ctx, *grads):
        # 역전파를 각 rank로 합쳐주는 reduce
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        grad_out = grads[rank]
        
        # 나머지 grad를 각자 랭크로 reduceSum
        grad_list = list(grads)
        for r in range(world_size):
            if r == rank:
                dist.reduce(grad_out, r, op=dist.ReduceOp.SUM)
            else:
                dist.reduce(grad_list[r], r, op=dist.ReduceOp.SUM)
        
        grad_out = grad_out * world_size
        return (grad_out,) + (None,)*(world_size)

AllGather = AllGatherFunc.apply


class PartialFC_V2(nn.Module):
    """
    분산환경에서 클래스가 매우 클 때, proxy(=class centers)를 부분적으로만 업데이트
    Proxy Anchor와도 함께 사용할 수 있다.
    
    margin_softmax에 Proxy Anchor를 넣을 경우, 내부적으로 logits -> proxy anchor... 
    혹은, logits 반환 후, 바깥에서 proxy anchor를 계산하도록 구성할 수도 있음.
    본 예시에서는 Embedding과 Sub-weight를 반환 후, 외부에서 ProxyAnchor 형태로 사용하도록 구성
    """
    def __init__(self, 
                 margin_softmax=None,  # 예: ProxyAnchorLoss 등의 함수형
                 embedding_size=128, 
                 num_classes=1000,
                 sample_rate=1.0,
                 fp16=False):
        super().__init__()
        assert dist.is_initialized(), "Distributed mode must be enabled before using PartialFC_V2."
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self.margin_softmax = margin_softmax  # 
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.fp16 = fp16
        
        # local number of classes
        self.num_local = num_classes // self.world_size + int(self.rank < num_classes % self.world_size)
        # 시작 인덱스
        self.class_start = num_classes // self.world_size * self.rank + min(self.rank, num_classes % self.world_size)
        
        # 실제 파라미터 weight
        self.weight = nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))
        self.register_buffer('weight_index', torch.arange(self.num_local, dtype=torch.long))

        # sample number
        self.num_sample = int(self.sample_rate * self.num_local)
        
        # cross entropy
        self.dist_ce = DistCrossEntropy()

    @torch.no_grad()
    def sample_indices(self, labels: torch.Tensor, index_positive):
        """
        labels 중 현재 rank 소유 클래스 범위에 속하는 것들만 골라서
        그 중 positive 클래스 + 추가 negative 샘플 등
        """
        unique_local_labels = torch.unique(labels[index_positive], sorted=True)
        # unique_local_labels: e.g., [classA, classB, ...] in [0..num_local-1]
        
        if self.num_sample <= unique_local_labels.size(0):
            index_sampled = unique_local_labels
        else:
            # negative 샘플을 추가적으로 뽑기
            perm = torch.rand(self.num_local, device=labels.device)
            # positive는 무조건 포함
            perm[unique_local_labels] = 2.0  # large
            # top num_sample
            index_sampled = torch.topk(perm, k=self.num_sample)[1].sort()[0]

        return index_sampled

    def forward(self, local_embeddings: torch.Tensor, local_labels: torch.Tensor):
        """
        1) embeddings, labels를 all_gather
        2) label -> this rank 소유 클래스만 남김, 아닌 건 -1
        3) sample -> sub_weight
        4) logits 계산 -> (optionally margin_softmax)
        5) loss or logits 반환
        """
        local_labels = local_labels.long().view(-1)
        batch_size = local_embeddings.size(0)
        
        # all_gather
        gather_embed_list = [
            torch.zeros_like(local_embeddings) for _ in range(self.world_size)
        ]
        gather_label_list = [
            torch.zeros_like(local_labels) for _ in range(self.world_size)
        ]
        # 실행
        embed_all = AllGather(local_embeddings, *gather_embed_list)
        label_all = AllGather(local_labels, *gather_label_list)

        # concat
        embed_all = torch.cat(embed_all, dim=0)
        label_all = torch.cat(label_all, dim=0)

        # label_all 의 값을 local index로 매핑
        # 예) [class_start, class_start+1, ...]
        index_positive = (label_all >= self.class_start) & (label_all < self.class_start + self.num_local)
        label_all[~index_positive] = -1  # not belongs to current rank
        label_all[index_positive] -= self.class_start  # => [0..num_local-1]

        # sampling
        if self.sample_rate < 1.0:
            # 실제로 sub_weight 인덱스를 고름
            index_sampled = self.sample_indices(label_all, index_positive)
        else:
            index_sampled = torch.arange(self.num_local, device=embed_all.device)

        # sub_label
        sub_label_valid_mask = index_positive & (label_all >= 0)
        # label_all[sub_label_valid_mask] in index_sampled 안에서의 위치
        # searchsorted
        # 이 mapping이 -1 아닌 것
        new_label = label_all.clone()
        new_label[sub_label_valid_mask] = torch.searchsorted(index_sampled, label_all[sub_label_valid_mask])
        
        # sub_weight
        sub_weight = F.normalize(self.weight[index_sampled], dim=1)

        # compute logits
        norm_embed_all = F.normalize(embed_all, dim=1)
        logits = norm_embed_all.mm(sub_weight.t())  # (B_all, #sampled)
        logits = torch.clamp(logits, min=-1.0, max=1.0)

        # margin softmax 있으면 적용
        # 여기서는 Proxy Anchor를 내부에서 직접 계산하기보다는
        # "logits, new_label, index_sampled"등을 반환해서
        # 바깥에서 proxy anchor를 계산하거나
        # 혹은 crossentropy를 쓴다면 dist CE를 써도 됨
        # => 아래는 일반 CE 예시
        if self.margin_softmax is None:
            loss = self.dist_ce(logits, new_label)
            return loss
        else:
            # Proxy Anchor를 사용할 경우:
            # 1) margin_softmax가 callable이라 가정(예: 함수 or nn.Module)
            # 2) embeddings(=norm_embed_all), sub_weight, new_label 이용
            #    => label -1 제외
            valid_idx = (new_label != -1).nonzero(as_tuple=True)[0]
            # gather only valid embeddings, labels
            valid_embed = norm_embed_all[valid_idx]
            valid_labels = new_label[valid_idx]
            # partial fc에서 sub_weight의 인덱스 -> 실제 global class -> alpha...
            # 하지만 Proxy Anchor는 사실 class-wise proxy를 직접 갖고 있어야 하는데
            # 이 예시에선 sub_weight가 local. => 아래처럼 근사
            # -> 또는 margin_softmax.forward(valid_embed, valid_labels)로 proxy anchor 계산
            #    (proxy anchor 안에 self.proxies가 있을 수도 있음 -> 충돌 주의)
            # 보통 partial fc 시나리오에서는, proxy anchor도 partial fc가 proxy를 관리
            # => 아래는 예시로 'margin_softmax(logits, new_label)' 형태로 하거나
            #    'margin_softmax(valid_embed, valid_labels, sub_weight, ...)' 등
            # 여기서는 logits과 label로는 Proxy Anchor가 불완전하므로,
            # 임시로 Cross Ent와 동일 로직 or dummy return
            # ------------------------------------------
            # "권장" -> Proxy Anchor를 이 시점에서 custom하게 계산:
            #   loss = self.margin_softmax(valid_embed, valid_labels, sub_weight, #sampled)
            # 실 구현은 아래처럼 래핑
            loss = self.margin_softmax(valid_embed, valid_labels) 
            return loss
