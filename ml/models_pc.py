"""포인트클라우드 백본. PointNet / DGCNN.

⚠ T-Net(입력 3x3 변환) 없음. 이유:
   (1) 데이터가 이미 정준 정렬돼 있다 (지면 z=0, 대칭축 y=0, x 앞→뒤).
   (2) 학습된 3x3 변환은 **미터 스케일을 되돌릴 수 있다**. 절대 높이가 Cd의 최강 신호라
       스케일을 없애는 어떤 연산도 넣으면 안 된다.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _head(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )


class PointNet(nn.Module):
    """per-point MLP -> max pool. 국소 이웃 정보를 전혀 안 본다."""

    def __init__(self, n_dims: int = 0, emb: int = 1024, attn: str = "none"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, emb, 1), nn.BatchNorm1d(emb), nn.ReLU(inplace=True),
        )
        self.attn = attn
        self.att = {"none": None, "se": _SE, "cbam": _PointCBAM,
                    "pool": _AttnPool, "sa": _SelfAttn}[attn]
        if self.att is not None:
            self.att = self.att(emb)
        self.head = _head(emb * (2 if attn == "pool" else 1) + n_dims)

    def forward(self, x, dims=None):          # x: (B,N,3)
        f = self.mlp(x.transpose(1, 2))       # (B,emb,N)
        if self.attn in ("se", "cbam", "sa"):
            f = self.att(f)
        g = torch.cat([f.amax(-1), self.att(f)], 1) if self.attn == "pool" else f.amax(-1)
        if dims is not None:
            g = torch.cat([g, dims], 1)
        return self.head(g).squeeze(-1)


# ----------------------------------------------------------------------------
# 어텐션 모듈 (--attn se|cbam|pool|sa). 전부 풀링 직전 (B,emb,N) 피처에 붙는다.
class _SE(nn.Module):
    """채널 어텐션 (Squeeze-Excitation)."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):                      # (B,C,N)
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class _PointCBAM(nn.Module):
    """CBAM의 포인트클라우드 판: 채널 게이트(avg+max 공유 MLP) -> 점 게이트."""

    def __init__(self, c, r=4):
        super().__init__()
        self.cfc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c))
        self.pconv = nn.Conv1d(2, 1, 1)

    def forward(self, x):                      # (B,C,N)
        x = x * torch.sigmoid(self.cfc(x.mean(-1)) + self.cfc(x.amax(-1))).unsqueeze(-1)
        pw = torch.sigmoid(self.pconv(torch.cat([x.mean(1, keepdim=True),
                                                 x.amax(1, keepdim=True)], 1)))
        return x * pw


class _AttnPool(nn.Module):
    """어텐션 풀링: 점별 score softmax 가중합. max-pool과 병행(concat) 사용."""

    def __init__(self, c):
        super().__init__()
        self.score = nn.Conv1d(c, 1, 1)

    def forward(self, x):                      # (B,C,N) -> (B,C)
        return (x * torch.softmax(self.score(x), -1)).sum(-1)


class _SelfAttn(nn.Module):
    """셀프어텐션 1블록 (pre-LN MHSA + residual). N점을 토큰으로 — O(N²) 주의."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):                      # (B,C,N)
        t = x.transpose(1, 2)
        h = self.ln(t)
        return (t + self.mha(h, h, h, need_weights=False)[0]).transpose(1, 2)


# ----------------------------------------------------------------------------
def knn(x, k):                                 # x: (B,C,N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def graph_feature(x, k, idx=None):             # -> (B,2C,N,k)
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + base).view(-1)
    xt = x.transpose(2, 1).contiguous()
    feat = xt.view(B * N, C)[idx].view(B, N, k, C)
    xr = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([feat - xr, xr], dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    """EdgeConv. 국소 이웃의 상대 좌표를 본다 -> 곡률/후미 형상을 잡을 수 있다."""

    def __init__(self, n_dims: int = 0, k: int = 20, emb: int = 1024):
        super().__init__()
        self.k = k
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.c1, self.c2, self.c3, self.c4 = blk(6, 64), blk(128, 64), blk(128, 128), blk(256, 256)
        self.c5 = nn.Sequential(nn.Conv1d(512, emb, 1, bias=False), nn.BatchNorm1d(emb),
                                nn.LeakyReLU(0.2, inplace=True))
        self.head = _head(emb * 2 + n_dims)

    def forward(self, x, dims=None):           # x: (B,N,3)
        x = x.transpose(1, 2)                  # (B,3,N)
        x1 = self.c1(graph_feature(x, self.k)).amax(-1)
        x2 = self.c2(graph_feature(x1, self.k)).amax(-1)
        x3 = self.c3(graph_feature(x2, self.k)).amax(-1)
        x4 = self.c4(graph_feature(x3, self.k)).amax(-1)
        g = self.c5(torch.cat([x1, x2, x3, x4], 1))
        g = torch.cat([g.amax(-1), g.mean(-1)], 1)
        if dims is not None:
            g = torch.cat([g, dims], 1)
        return self.head(g).squeeze(-1)


class TriplaneCNN(nn.Module):
    """TripNet-스타일 triplane 라이트 재구현 (TripNet 공식 코드 비공개라 표현 아이디어만).

    점 -> 3개 직교평면(xy/xz/yz)에 점유·최대수직좌표 2채널 래스터 -> 공유 2D CNN -> concat.
    래스터 범위는 **전역 고정 bbox(미터)** — 차가 그리드에서 차지하는 크기로 절대 스케일이
    인코딩된다 (불변식 1 유지). per-cloud bbox로 바꾸면 스케일 신호가 사라지니 금지.
    """

    def __init__(self, n_dims: int = 0, R: int = 64, ch: int = 32):
        super().__init__()
        self.R = R
        # 전역 bbox: DATA_SUMMARY §3 실측 min/max + 소폭 여유
        self.register_buffer("lo", torch.tensor([-1.20, -1.25, -0.05]))
        self.register_buffer("hi", torch.tensor([4.30, 1.25, 1.80]))

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(blk(2, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                 blk(ch * 4, ch * 8), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = _head(ch * 8 * 3 + n_dims)

    def _plane(self, u, v, w, B):
        """u,v: (B,N) 그리드 정수좌표, w: (B,N) 수직좌표 [0,1] -> (B,2,R,R)"""
        R = self.R
        flat = (torch.arange(B, device=u.device).unsqueeze(1) * R * R + u * R + v).view(-1)
        occ = torch.zeros(B * R * R, device=u.device).index_put_(
            (flat,), torch.ones_like(flat, dtype=torch.float32), accumulate=True)
        hmax = torch.zeros(B * R * R, device=u.device).scatter_reduce_(
            0, flat, w.reshape(-1), reduce="amax", include_self=True)
        occ, hmax = occ.view(B, 1, R, R), hmax.view(B, 1, R, R)
        occ = occ / (occ.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.cat([occ, hmax], 1)

    def forward(self, x, dims=None):           # x: (B,N,3) meters
        B = x.shape[0]
        t = ((x - self.lo) / (self.hi - self.lo)).clamp(0, 1 - 1e-6)
        g = (t * self.R).long()
        planes = [self._plane(g[..., a], g[..., b], t[..., c], B)
                  for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0))]   # 평면⊥수직축
        f = torch.cat([self.cnn(p) for p in planes], 1)
        if dims is not None:
            f = torch.cat([f, dims], 1)
        return self.head(f).squeeze(-1)


class RegDGCNNWrap(nn.Module):
    """DrivAerNet 저자 RegDGCNN(k40, emb512, dropout0.4)을 공정 비교 인터페이스로 래핑."""

    def __init__(self, n_dims: int = 0, **_):
        super().__init__()
        assert n_dims == 0, "regdgcnn 래퍼는 --dims 0만 지원"
        import sys as _sys
        p = "/home/kwy00/qi/external/DrivAerNet/DeepSurrogates"
        if p not in _sys.path:
            _sys.path.insert(0, p)
        from DeepSurrogate_models import RegDGCNN
        self.net = RegDGCNN({"k": 40, "emb_dims": 512, "dropout": 0.4})

    def forward(self, x, dims=None):           # (B,N,3)
        return self.net(x.transpose(1, 2)).squeeze(-1)


class DimsMLP(nn.Module):
    """대조군(R1): 포인트클라우드를 아예 안 본다. 로버스트 치수 6개만. R0의 비선형 상한."""

    def __init__(self, n_dims: int = 6, **_):
        super().__init__()
        assert n_dims == 6, "DimsMLP는 --dims 1 필요"
        self.head = _head(n_dims)

    def forward(self, x, dims=None):
        return self.head(dims).squeeze(-1)


BACKBONES = {"pointnet": PointNet, "dgcnn": DGCNN, "mlp": DimsMLP,
             "regdgcnn": RegDGCNNWrap, "triplane": TriplaneCNN}
