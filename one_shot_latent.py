"""
One-image-per-class learning dans un espace latent (sans CNN).

Idée :
  1 image par classe  ->  augmentations FORTES (couleur + déformation géométrique
  de la forme, sans découpage)  ->  encodeur ViT from scratch  ->  espace latent
  -> perte contrastive supervisée (SupCon) : toutes les variantes d'une même
  image/classe sont rapprochées, les classes sont éloignées.
  Test : classification par prototype (similarité cosinus) sur de nouvelles images.

Arborescence attendue :
  data/train/cat.jpg        (le nom du fichier = nom de la classe)
  data/train/dog.jpg        (un PNG RGBA avec fond transparent )
  data/test/cat/*.jpg
  data/test/dog/*.jpg

Lancer :
  python one_shot_latent.py --train_dir data/train --test_dir data/test
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
def load_rgba(path, size):
    """Charge une image en tenseur (4,H,W) dans [0,1]. Alpha=1 si pas de transparence."""
    im = Image.open(path).convert("RGBA").resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(im).copy()).float().div(255).permute(2, 0, 1)
    return t


def rgba_to_rgb_gray_bg(t):
    """Composite sur fond gris (pour les images de test)."""
    rgb, a = t[:3], t[3:]
    return rgb * a + 0.5 * (1 - a)


def normalize(x):
    return (x - 0.5) / 0.5


def load_train(train_dir, size):
    files = sorted(p for p in Path(train_dir).iterdir() if p.suffix.lower() in EXTS)
    assert len(files) >= 2, f"Il faut au moins 2 images dans {train_dir}"
    names = [p.stem for p in files]
    imgs = torch.stack([load_rgba(p, size) for p in files])  # (C,4,H,W)
    return names, imgs


def load_test(test_dir, names, size):
    xs, ys = [], []
    for ci, n in enumerate(names):
        d = Path(test_dir) / n
        if not d.is_dir():
            print(f"[warn] dossier de test absent: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                xs.append(normalize(rgba_to_rgb_gray_bg(load_rgba(p, size))))
                ys.append(ci)
    if not xs:
        return None, None
    return torch.stack(xs), torch.tensor(ys)


# ----------------------------------------------------------------------------
# Augmentations : couleur + déformation géométrique (batchées, sur GPU/CPU)
# ----------------------------------------------------------------------------
_T_YIQ = torch.tensor([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.322],
                       [0.211, -0.523, 0.312]])
_T_RGB = torch.linalg.inv(_T_YIQ)


def U(B, lo, hi, device):
    return torch.rand(B, device=device) * (hi - lo) + lo


class Augmenter:
    """
    Pipeline :  couleur (sur l'objet) -> warp géométrique (affine + élastique,
    l'objet est déformé, jamais découpé) -> fond aléatoire -> bruit/occlusion.
    """

    def __init__(self, size, device, max_rot=35, elastic_sigma=0.12, grid=4):
        self.size, self.device = size, device
        self.max_rot = math.radians(max_rot)
        self.sigma, self.grid = elastic_sigma, grid
        self.T_yiq, self.T_rgb = _T_YIQ.to(device), _T_RGB.to(device)

    # ---- couleur -----------------------------------------------------------
    def color(self, rgb):
        B, _, H, W = rgb.shape
        d = rgb.device
        # 1) rotation de teinte dans l'espace YIQ (chat roux -> bleu -> vert ...)
        th = U(B, -math.pi, math.pi, d)
        R = torch.zeros(B, 3, 3, device=d)
        R[:, 0, 0] = 1
        R[:, 1, 1], R[:, 1, 2] = torch.cos(th), -torch.sin(th)
        R[:, 2, 1], R[:, 2, 2] = torch.sin(th), torch.cos(th)
        M = self.T_rgb @ R @ self.T_yiq
        x = torch.einsum("bij,bjhw->bihw", M, rgb)
        # 2) saturation
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        s = U(B, 0.2, 1.8, d).view(B, 1, 1, 1)
        x = gray + s * (x - gray)
        # 3) permutation aléatoire des canaux (p=0.3)
        idx = torch.stack([torch.randperm(3, device=d) for _ in range(B)])
        xp = torch.gather(x, 1, idx[:, :, None, None].expand(-1, -1, H, W))
        use = (torch.rand(B, 1, 1, 1, device=d) < 0.3)
        x = torch.where(use, xp, x)
        # 4) contraste / luminosité
        c = U(B, 0.6, 1.4, d).view(B, 1, 1, 1)
        b = U(B, 0.7, 1.3, d).view(B, 1, 1, 1)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
        # 5) niveaux de gris (p=0.15)
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        use = (torch.rand(B, 1, 1, 1, device=d) < 0.15)
        x = torch.where(use, g.expand(-1, 3, -1, -1), x)
        return x.clamp(0, 1)

    # ---- géométrie ---------------------------------------------------------
    def geometry(self, x):
        """x: (B,C,H,W). Affine aléatoire + champ de déformation élastique lisse."""
        B, C, H, W = x.shape
        d = x.device
        ang = U(B, -self.max_rot, self.max_rot, d)
        sx = U(B, 0.75, 1.15, d)
        sy = sx * U(B, 0.85, 1.15, d)
        shear = U(B, -0.2, 0.2, d)
        tx, ty = U(B, -0.15, 0.15, d), U(B, -0.15, 0.15, d)
        flip = torch.where(torch.rand(B, device=d) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([
            torch.stack([cos * flip / sx, -sin / sx + shear, tx], 1),
            torch.stack([sin * flip / sy, cos / sy, ty], 1)], 1)  # (B,2,3)
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        # déformation élastique : points de contrôle grid x grid, interpolés (bicubic)
        coarse = torch.randn(B, 2, self.grid, self.grid, device=d) * self.sigma
        flow = F.interpolate(coarse, size=(H, W), mode="bicubic", align_corners=True)
        grid = grid + flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)

    # ---- fond / bruit / occlusion -----------------------------------------
    def background(self, B):
        k = random.choice([1, 2, 4, 8])
        low = torch.rand(B, 3, k, k, device=self.device)
        return F.interpolate(low, size=(self.size, self.size), mode="bilinear",
                             align_corners=False)

    def noise_occlusion(self, x):
        B, _, H, W = x.shape
        d = x.device
        x = x + torch.randn_like(x) * U(B, 0.0, 0.06, d).view(B, 1, 1, 1)
        # occlusion rectangulaire (p=0.5)
        h = (U(B, 0.1, 0.4, d) * H).long().view(B, 1, 1)
        w = (U(B, 0.1, 0.4, d) * W).long().view(B, 1, 1)
        y0 = (torch.rand(B, device=d) * (H - h.view(B))).long().view(B, 1, 1)
        x0 = (torch.rand(B, device=d) * (W - w.view(B))).long().view(B, 1, 1)
        ys = torch.arange(H, device=d).view(1, H, 1)
        xs = torch.arange(W, device=d).view(1, 1, W)
        m = (ys >= y0) & (ys < y0 + h) & (xs >= x0) & (xs < x0 + w)
        m = m & (torch.rand(B, 1, 1, device=d) < 0.5)
        fill = torch.rand(B, 3, 1, 1, device=d)
        x = torch.where(m.unsqueeze(1), fill.expand_as(x), x)
        # léger flou (p=0.3 par batch)
        if random.random() < 0.3:
            x = F.avg_pool2d(x, 3, stride=1, padding=1)
        return x.clamp(0, 1)

    @torch.no_grad()
    def __call__(self, rgba):
        rgba = rgba.to(self.device)
        rgb, a = rgba[:, :3], rgba[:, 3:]
        rgb = self.color(rgb)
        pm = torch.cat([rgb * a, a], 1)            # alpha pré-multiplié
        pm = self.geometry(pm)
        rgb_pm, a = pm[:, :3], pm[:, 3:]
        x = rgb_pm + (1 - a) * self.background(rgba.shape[0])
        x = self.noise_occlusion(x)
        return normalize(x)


# ----------------------------------------------------------------------------
# Encodeur : ViT from scratch (aucune convolution)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, size=64, patch=8, dim=128, depth=4, heads=4, proj_dim=64):
        super().__init__()
        assert size % patch == 0
        self.patch = patch
        n = (size // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, dim)   # pas de Conv2d
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proj_dim))

    def patchify(self, x):
        p = self.patch
        B, C, H, W = x.shape
        x = x.unfold(2, p, p).unfold(3, p, p)            # B,C,nh,nw,p,p
        return x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)

    def forward(self, x):
        t = self.embed(self.patchify(x))
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t = b(t)
        h = self.norm(t)[:, 0]                           # représentation (backbone)
        z = F.normalize(self.proj(h), dim=-1)            # espace latent (projeté)
        return F.normalize(h, dim=-1), z


# ----------------------------------------------------------------------------
# Perte : Supervised Contrastive
# ----------------------------------------------------------------------------
def supcon_loss(z, y, temp=0.1):
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    return -((log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


# ----------------------------------------------------------------------------
# Évaluation : classification par prototype
# ----------------------------------------------------------------------------
@torch.no_grad()
def embed(model, x, device, bs=256):
    model.eval()
    hs, zs = [], []
    for i in range(0, len(x), bs):
        h, z = model(x[i:i + bs].to(device))
        hs.append(h), zs.append(z)
    return torch.cat(hs), torch.cat(zs)


@torch.no_grad()
def evaluate(model, base, aug, Xte, Yte, names, device, n_proto=256):
    C = len(base)
    views = aug(base.repeat_interleave(n_proto, 0))
    h, z = embed(model, views, device)
    ph = F.normalize(h.view(C, n_proto, -1).mean(1), dim=-1)
    pz = F.normalize(z.view(C, n_proto, -1).mean(1), dim=-1)
    th, tz = embed(model, Xte, device)
    Yte = Yte.to(device)
    res = {}
    for tag, t, p in (("backbone", th, ph), ("latent", tz, pz)):
        pred = (t @ p.T).argmax(1)
        acc = (pred == Yte).float().mean().item()
        per = {names[c]: (pred[Yte == c] == c).float().mean().item()
               for c in range(C) if (Yte == c).any()}
        res[tag] = (acc, per)
    model.train()
    return res


def save_preview(aug, base, path, n=10):
    C = len(base)
    rows = []
    for c in range(C):
        orig = normalize(rgba_to_rgb_gray_bg(base[c])).unsqueeze(0).to(aug.device)
        v = aug(base[c:c + 1].repeat(n - 1, 1, 1, 1))
        rows.append(torch.cat([orig, v], 0))
    g = torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)
    g = ((g * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(g).save(path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--test_dir", default="data/test")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--views", type=int, default=64, help="variantes par classe et par pas")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names, base = load_train(args.train_dir, args.size)
    C = len(names)
    print(f"device={device} | classes={names}")
    Xte, Yte = (None, None)
    if Path(args.test_dir).exists():
        Xte, Yte = load_test(args.test_dir, names, args.size)
        if Xte is not None:
            print(f"test: {len(Xte)} images")

    aug = Augmenter(args.size, device)
    save_preview(aug, base, out / "aug_preview.png")
    print(f"aperçu des augmentations -> {out/'aug_preview.png'}")

    model = ViT(args.size, args.patch, args.dim, args.depth, args.heads).to(device)
    print(f"paramètres: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    labels = torch.arange(C).repeat_interleave(args.views).to(device)
    src = base.repeat_interleave(args.views, 0)

    def report(step):
        if Xte is None:
            return
        r = evaluate(model, base, aug, Xte, Yte, names, device)
        s = " | ".join(f"{k}: acc={a*100:.1f}% " + str({n: round(v, 2) for n, v in per.items()})
                       for k, (a, per) in r.items())
        print(f"  [eval @ {step}] {s}")

    print("--- baseline (réseau non entraîné) ---")
    report(0)

    model.train()
    for step in range(1, args.steps + 1):
        x = aug(src)
        _, z = model(x)
        loss = supcon_loss(z, labels, args.temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            report(step)

    torch.save({"model": model.state_dict(), "names": names, "args": vars(args)},
               out / "model.pt")
    print(f"modèle sauvegardé -> {out/'model.pt'}")


if __name__ == "__main__":
    main()