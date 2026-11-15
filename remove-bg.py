"""
Détoure le fond des images (rembg) et recadre l'objet dans un carré transparent.
Produit des PNG RGBA utilisables par one_shot_latent_v2.py.

Installation :  pip install "rembg[cpu]" pillow
Exemples :
  python remove_bg.py --inp data_raw/train --out data/train
  python remove_bg.py --inp data_raw/test  --out data/test     # (protocole « objet seul »)
"""
import argparse
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def square_crop(im, margin=0.08):
    box = im.split()[3].getbbox()          # boîte englobante de l'alpha
    if box:
        im = im.crop(box)
    w, h = im.size
    s = int(max(w, h) * (1 + 2 * margin))
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2), im)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="u2net")
    args = ap.parse_args()

    session = new_session(args.model)
    inp, out = Path(args.inp), Path(args.out)
    files = [p for p in inp.rglob("*") if p.suffix.lower() in EXTS]
    print(f"{len(files)} images à traiter")
    for p in files:
        dst = out / p.relative_to(inp).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cut = remove(Image.open(p).convert("RGB"), session=session).convert("RGBA")
        square_crop(cut).save(dst)
        print(f"  {p} -> {dst}")


if __name__ == "__main__":
    main()