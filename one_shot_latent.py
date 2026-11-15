"""
one-shot-latent v2  (ViT from scratch, sans CNN)

Changements par rapport à la v1 :
  * Ablations : --no_color --no_geo --no_bg --no_local --no_swap
  * Vues LOCALES (zoom sur une partie de l'objet) -> force des features de parties
  * Échange de palette entre classes (stats couleur du chien appliquées au chat...)
    -> la couleur ne peut plus servir d'indice de classe
  * Padding par RÉFLEXION pour les images opaques (JPG) : plus de bordures
    colorées jamais vues au test
  * Augmentations photométriques plus douces + une fraction de vues propres
  * Prototypes = moyenne (vues augmentées + image propre)
  * Plusieurs graines : --seeds 0 1 2 3 4  -> moyenne ± écart-type

Arborescence : voir README (data/train/<classe>.png, data/test/<classe>/*.jpg)
Astuce : détourer les fonds avec remove_bg.py (PNG RGBA) AVANT d'entraîner.
"""
import argparse
import json
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
    im = Image.open(path).convert("RGBA").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).float().div(255).permute(2, 0, 1)


def rgba_to_rgb_gray_bg(t):
    """Composite sur fond gris. Accepte (4,H,W) ou (B,4,H,W)."""
    rgb, a = t[..., :3, :, :], t[..., 3:, :, :]
    return rgb * a + 0.5 * (1 - a)


def normalize(x):
    return (x - 0.5) / 0.5


def load_train(train_dir, size):
    files = sorted(p for p in Path(train_dir).iterdir() if p.suffix.lower() in EXTS)
    assert len(files) >= 2, f"Il faut au moins 2 images dans {train_dir}"
    return [p.stem for p in files], torch.stack([load_rgba(p, size) for p in files])


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
# Augmentations
# ----------------------------------------------------------------------------
_T_YIQ = torch.tensor([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.322],
                       [0.211, -0.523, 0.312]])
_T_RGB = torch.linalg.inv(_T_YIQ)


def U(B, lo, hi, device):
    return torch.rand(B, device=device) * (hi - lo) + lo


class Augmenter:
    def __init__(self, size, device, base, use_color=True, use_geo=True,
                 use_bg=True, use_local=True, use_swap=True,
                 max_rot=30, elastic_sigma=0.10, grid=4, p_local=0.4):
        self.size, self.device = size, device
        self.use_color, self.use_geo, self.use_bg = use_color, use_geo, use_bg
        self.use_local, self.use_swap = use_local, use_swap
        self.max_rot = math.radians(max_rot)
        self.sigma, self.grid, self.p_local = elastic_sigma, grid, p_local
        self.T_yiq, self.T_rgb = _T_YIQ.to(device), _T_RGB.to(device)
        # statistiques couleur de chaque classe (pondérées par alpha)
        base = base.to(device)
        self.C = len(base)
        rgb, a = base[:, :3], base[:, 3:]
        w = a / a.sum((1, 2, 3), keepdim=True).clamp(min=1e-6)
        mu = (rgb * w).sum((2, 3))
        var = (((rgb - mu[:, :, None, None]) ** 2) * w).sum((2, 3))
        self.mu, self.sd = mu, var.sqrt().clamp(min=0.05)

    # ---- échange de palette entre classes ---------------------------------
    def swap(self, rgb, cls):
        B = rgb.shape[0]
        d = rgb.device
        tgt = torch.randint(0, self.C, (B,), device=d)
        tgt = torch.where(torch.rand(B, device=d) < 0.7, tgt, cls)  # 30%: identité
        s = lambda t: t[:, :, None, None]
        x = (rgb - s(self.mu[cls])) / s(self.sd[cls]) * s(self.sd[tgt]) + s(self.mu[tgt])
        return x.clamp(0, 1)

    # ---- couleur -----------------------------------------------------------
    def color(self, rgb):
        B, _, H, W = rgb.shape
        d = rgb.device
        th = U(B, -math.pi, math.pi, d)                    # rotation de teinte (YIQ)
        R = torch.zeros(B, 3, 3, device=d)
        R[:, 0, 0] = 1
        R[:, 1, 1], R[:, 1, 2] = torch.cos(th), -torch.sin(th)
        R[:, 2, 1], R[:, 2, 2] = torch.sin(th), torch.cos(th)
        x = torch.einsum("bij,bjhw->bihw", self.T_rgb @ R @ self.T_yiq, rgb)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = gray + U(B, 0.4, 1.6, d).view(B, 1, 1, 1) * (x - gray)   # saturation
        idx = torch.stack([torch.randperm(3, device=d) for _ in range(B)])
        xp = torch.gather(x, 1, idx[:, :, None, None].expand(-1, -1, H, W))
        x = torch.where(torch.rand(B, 1, 1, 1, device=d) < 0.3, xp, x)  # permut. canaux
        c = U(B, 0.75, 1.25, d).view(B, 1, 1, 1)
        b = U(B, 0.8, 1.2, d).view(B, 1, 1, 1)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = torch.where(torch.rand(B, 1, 1, 1, device=d) < 0.05, g.expand(-1, 3, -1, -1), x)
        return x.clamp(0, 1)

    # ---- géométrie ---------------------------------------------------------
    def geometry(self, pm, a0, opaque):
        """pm: (B,4,H,W) alpha pré-multiplié. a0: alpha d'origine (B,1,H,W)."""
        B, C, H, W = pm.shape
        d = pm.device
        g = 1.0 if self.use_geo else 0.0
        ang = U(B, -self.max_rot, self.max_rot, d) * g
        z = 1 + (U(B, 0.85, 1.25, d) - 1) * g
        asp = 1 + (U(B, 0.85, 1.15, d) - 1) * g
        shear = U(B, -0.2, 0.2, d) * g
        flip = torch.where(torch.rand(B, device=d) < 0.5 * g, -1.0, 1.0)

        # vues locales : zoom centré sur un pixel de l'objet (tiré selon alpha)
        local = (torch.rand(B, device=d) < self.p_local) if self.use_local \
            else torch.zeros(B, dtype=torch.bool, device=d)
        zl = U(B, 1.5, 2.6, d)
        sx = z * torch.where(local, zl, torch.ones_like(zl))
        sy = z * asp * torch.where(local, zl, torch.ones_like(zl))
        pix = torch.multinomial(a0.flatten(1) + 1e-4, 1).squeeze(1)
        cy = ((pix // W).float() + 0.5) / H * 2 - 1
        cx = ((pix % W).float() + 0.5) / W * 2 - 1
        lx, ly = (1 - 1 / sx).clamp(min=0), (1 - 1 / sy).clamp(min=0)
        tx_loc = torch.minimum(torch.maximum(cx, -lx), lx)
        ty_loc = torch.minimum(torch.maximum(cy, -ly), ly)
        tx = torch.where(local, tx_loc, U(B, -0.15, 0.15, d) * g)
        ty = torch.where(local, ty_loc, U(B, -0.15, 0.15, d) * g)

        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([
            torch.stack([cos * flip / sx, -sin / sx + shear, tx], 1),
            torch.stack([sin * flip / sy, cos / sy, ty], 1)], 1)
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        coarse = torch.randn(B, 2, self.grid, self.grid, device=d) * self.sigma * g
        flow = F.interpolate(coarse, size=(H, W), mode="bicubic", align_corners=True)
        grid = grid + flow.permute(0, 2, 3, 1)
        out_zero = F.grid_sample(pm, grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=False)
        out_refl = F.grid_sample(pm, grid, mode="bilinear", padding_mode="reflection",
                                 align_corners=False)
        # image opaque (JPG) -> réflexion ; image avec transparence -> zéros (fond aléatoire)
        return torch.where(opaque.view(B, 1, 1, 1), out_refl, out_zero)

    # ---- fond / bruit / occlusion -----------------------------------------
    def background(self, B):
        if not self.use_bg:
            return torch.full((B, 3, self.size, self.size), 0.5, device=self.device)
        k = random.choice([1, 2, 4, 8])
        low = torch.rand(B, 3, k, k, device=self.device)
        return F.interpolate(low, size=(self.size, self.size), mode="bilinear",
                             align_corners=False)

    def noise_occlusion(self, x):
        B, _, H, W = x.shape
        d = x.device
        x = x + torch.randn_like(x) * U(B, 0.0, 0.03, d).view(B, 1, 1, 1)
        h = (U(B, 0.1, 0.3, d) * H).long().view(B, 1, 1)
        w = (U(B, 0.1, 0.3, d) * W).long().view(B, 1, 1)
        y0 = (torch.rand(B, device=d) * (H - h.view(B))).long().view(B, 1, 1)
        x0 = (torch.rand(B, device=d) * (W - w.view(B))).long().view(B, 1, 1)
        ys = torch.arange(H, device=d).view(1, H, 1)
        xs = torch.arange(W, device=d).view(1, 1, W)
        m = (ys >= y0) & (ys < y0 + h) & (xs >= x0) & (xs < x0 + w)
        m = m & (torch.rand(B, 1, 1, device=d) < 0.3)
        x = torch.where(m.unsqueeze(1), torch.rand(B, 3, 1, 1, device=d).expand_as(x), x)
        if random.random() < 0.15:
            x = F.avg_pool2d(x, 3, stride=1, padding=1)
        return x.clamp(0, 1)

    @torch.no_grad()
    def __call__(self, rgba, cls):
        rgba, cls = rgba.to(self.device), cls.to(self.device)
        rgb, a = rgba[:, :3], rgba[:, 3:]
        opaque = a.amin((1, 2, 3)) > 0.99
        if self.use_swap:
            rgb = self.swap(rgb, cls)
        if self.use_color:
            rgb = self.color(rgb)
        pm = self.geometry(torch.cat([rgb * a, a], 1), a, opaque)
        rgb_pm, a2 = pm[:, :3], pm[:, 3:]
        x = rgb_pm + (1 - a2) * self.background(rgba.shape[0])
        return normalize(self.noise_occlusion(x))


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
        self.embed = nn.Linear(3 * patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proj_dim))

    def patchify(self, x):
        p = self.patch
        B, C, H, W = x.shape
        x = x.unfold(2, p, p).unfold(3, p, p)
        return x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)

    def forward(self, x):
        t = self.embed(self.patchify(x))
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t = b(t)
        h = self.norm(t)[:, 0]
        return F.normalize(h, dim=-1), F.normalize(self.proj(h), dim=-1)


def supcon_loss(z, y, temp=0.1):
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    return -((log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


# ----------------------------------------------------------------------------
# Évaluation
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
    """Prototype = normalisation(moyenne des vues augmentées + image propre)."""
    C = len(base)
    cls = torch.arange(C).repeat_interleave(n_proto)
    h, z = embed(model, aug(base.repeat_interleave(n_proto, 0), cls), device)
    hc, zc = embed(model, normalize(rgba_to_rgb_gray_bg(base)), device)
    ph = F.normalize(F.normalize(h.view(C, n_proto, -1).mean(1), dim=-1) + hc, dim=-1)
    pz = F.normalize(F.normalize(z.view(C, n_proto, -1).mean(1), dim=-1) + zc, dim=-1)
    th, tz = embed(model, Xte, device)
    Yte = Yte.to(device)
    res = {}
    for tag, t, p in (("backbone", th, ph), ("latent", tz, pz)):
        pred = (t @ p.T).argmax(1)
        res[tag] = {"acc": (pred == Yte).float().mean().item(),
                    "per_class": {names[c]: (pred[Yte == c] == c).float().mean().item()
                                  for c in range(C) if (Yte == c).any()}}
    model.train()
    return res


def fmt(res):
    return " | ".join(f"{k}: {v['acc']*100:.1f}% "
                      + str({n: round(a, 2) for n, a in v['per_class'].items()})
                      for k, v in res.items())


def save_preview(aug, base, path, n=10):
    rows = []
    for c in range(len(base)):
        orig = normalize(rgba_to_rgb_gray_bg(base[c])).unsqueeze(0).to(aug.device)
        v = aug(base[c:c + 1].repeat(n - 1, 1, 1, 1), torch.full((n - 1,), c))
        rows.append(torch.cat([orig, v], 0))
    g = torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)
    g = ((g * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(g).save(path)


# ----------------------------------------------------------------------------
# Entraînement d'une graine
# ----------------------------------------------------------------------------
def run(args, seed, names, base, Xte, Yte, device, out):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    C = len(names)
    aug = Augmenter(args.size, device, base,
                    use_color=not args.no_color, use_geo=not args.no_geo,
                    use_bg=not args.no_bg, use_local=not args.no_local,
                    use_swap=not args.no_swap,
                    elastic_sigma=args.elastic_sigma, p_local=args.p_local)
    save_preview(aug, base, out / "aug_preview.png")

    model = ViT(args.size, args.patch, args.dim, args.depth, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    labels = torch.arange(C).repeat_interleave(args.views).to(device)
    src = base.repeat_interleave(args.views, 0)
    src_clean = normalize(rgba_to_rgb_gray_bg(src)).to(device)

    baseline = evaluate(model, base, aug, Xte, Yte, names, device)
    print(f"[seed {seed}] baseline non entraînée -> {fmt(baseline)}")
    print(f"[seed {seed}] plancher théorique de la loss ~ ln({args.views - 1}) "
          f"= {math.log(args.views - 1):.3f}")

    model.train()
    last = baseline
    for step in range(1, args.steps + 1):
        x = aug(src, labels)
        if args.clean_frac > 0:
            m = torch.rand(len(x), device=device) < args.clean_frac
            x = torch.where(m.view(-1, 1, 1, 1), src_clean, x)
        _, z = model(x)
        loss = supcon_loss(z, labels, args.temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"[seed {seed}] step {step:5d}/{args.steps} | loss {loss.item():.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            last = evaluate(model, base, aug, Xte, Yte, names, device)
            print(f"[seed {seed}]   eval @ {step}: {fmt(last)}")
    torch.save({"model": model.state_dict(), "names": names, "args": vars(args)},
               out / "model.pt")
    return {"baseline": baseline, "final": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--test_dir", default="data/test")
    ap.add_argument("--out", default="runs_v2")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--clean_frac", type=float, default=0.1)
    ap.add_argument("--elastic_sigma", type=float, default=0.10)
    ap.add_argument("--p_local", type=float, default=0.4)
    for f in ("color", "geo", "bg", "local", "swap"):
        ap.add_argument(f"--no_{f}", action="store_true", help=f"désactive l'augmentation {f}")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names, base = load_train(args.train_dir, args.size)
    Xte, Yte = load_test(args.test_dir, names, args.size)
    assert Xte is not None, f"Aucune image de test trouvée dans {args.test_dir}"
    print(f"device={device} | classes={names} | test={len(Xte)} images")
    flags = {f: not getattr(args, f"no_{f}") for f in ("color", "geo", "bg", "local", "swap")}
    print(f"augmentations actives: {flags}")

    out = Path(args.out)
    results = {}
    for s in args.seeds:
        results[s] = run(args, s, names, base, Xte, Yte, device, out / f"seed{s}")

    print("\n===== RÉSUMÉ =====")
    summary = {"flags": flags, "seeds": args.seeds, "n_test": len(Xte)}
    for tag in ("backbone", "latent"):
        for stage in ("baseline", "final"):
            v = np.array([results[s][stage][tag]["acc"] for s in args.seeds]) * 100
            summary[f"{stage}_{tag}"] = {"mean": float(v.mean()), "std": float(v.std())}
            print(f"{stage:9s} {tag:9s}: {v.mean():.1f} ± {v.std():.1f} %")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out/'summary.json'}")


if __name__ == "__main__":
    main()