# one-shot-latent

> Apprentissage avec **une seule image par classe** : un ViT sans CNN, entraîné *from scratch*, avec de fortes augmentations de couleur et de déformation géométrique dans un espace latent contrastif.

![statut](https://img.shields.io/badge/statut-recherche%20exploratoire-orange)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-red)
![licence](https://img.shields.io/badge/licence-MIT-green)

## Motivation

Un enfant n'a pas besoin de milliers d'images pour reconnaître un chat : il sait qu'un chat reste un chat qu'il soit noir, roux ou gris, et quelle que soit sa pose.

**Hypothèse testée ici :** si un réseau apprend, à partir d'une seule image par classe, à être **invariant à la couleur** et **à la déformation de la forme**, peut-il généraliser à de nouvelles images de la même classe ?

Le projet explore cette question avec un encodeur **sans convolution** (Vision Transformer écrit à la main) entraîné de zéro, sans pré-entraînement.

## Méthode

```
1 image / classe
      │
      ├─► transformation de couleur   (teinte, saturation, canaux, contraste…)
      ├─► déformation géométrique     (affine + champ élastique, sans découpage)
      ├─► fond aléatoire, bruit, occlusion
      ▼
  Encodeur ViT (from scratch)
      ▼
  Espace latent z  ──►  perte contrastive supervisée (SupCon)
      ▼
  Test : similarité cosinus aux prototypes de classe
```

### Composants

| Composant | Détail |
|---|---|
| **Encodeur** | ViT minimal : patchs via `unfold` + couche `Linear` (aucun `Conv2d`), 4 blocs, dimension 128 (≈ 0,6 M de paramètres) |
| **Couleur** | Rotation de teinte (espace YIQ), saturation, permutation des canaux, contraste, luminosité, niveaux de gris |
| **Géométrie** | Transformation affine aléatoire + champ de déformation élastique lisse (grille 4×4 interpolée en bicubique). La forme est déformée, jamais découpée |
| **Fond** | Remplacement par des fonds aléatoires (si l'image d'entraînement est un PNG RGBA avec transparence) |
| **Perte** | *Supervised Contrastive Loss* : les variantes d'une même classe sont rapprochées, les classes distinctes sont éloignées |
| **Évaluation** | Prototype de classe = moyenne des embeddings de variantes augmentées ; classification par similarité cosinus |

Le script évalue deux couches : la représentation du **backbone** et le vecteur **latent** projeté.

## Installation

```bash
git clone https://github.com/gbelsalvador/one-shot-latent.git
cd one-shot-latent
pip install torch numpy pillow
```

## Utilisation

### 1. Préparer les données

```
data/
├── train/
│   ├── cat.png        # le nom du fichier = nom de la classe
│   └── dog.png
└── test/
    ├── cat/           # images de test de la classe « cat »
    │   ├── 01.jpg
    │   └── ...
    └── dog/
        ├── 01.jpg
        └── ...
```

> **Conseil :** utilise des PNG avec **fond transparent** (RGBA) pour l'entraînement. Le code remplace alors le fond par des fonds aléatoires, ce qui évite que le modèle apprenne l'arrière-plan au lieu de l'animal. Avec un JPG, le fond d'origine reste collé à l'objet.

### 2. Lancer l'entraînement

```bash
python one_shot_latent.py --train_dir data/train --test_dir data/test
```

### Options principales

| Option | Défaut | Description |
|---|---|---|
| `--size` | 64 | Taille des images (carrées) |
| `--patch` | 8 | Taille des patchs du ViT |
| `--dim` / `--depth` / `--heads` | 128 / 4 / 4 | Taille du ViT |
| `--steps` | 1500 | Nombre de pas d'entraînement |
| `--views` | 64 | Variantes générées par classe et par pas |
| `--temp` | 0.1 | Température de la perte contrastive |
| `--lr` | 3e-4 | Taux d'apprentissage |
| `--eval_every` | 250 | Fréquence d'évaluation |
| `--seed` | 0 | Graine aléatoire |
| `--out` | `runs` | Dossier de sortie |

### Sorties

- `runs/aug_preview.png` : aperçu des augmentations (à regarder avant tout entraînement)
- `runs/model.pt` : poids du modèle
- Console : perte et accuracy (backbone et latent), avec la baseline d'un réseau non entraîné au pas 0

## Résultats

> ⚠️ **À compléter.** Remplir avec tes mesures. Répéter chaque expérience sur plusieurs graines (par exemple 5) et reporter la moyenne ± l'écart-type : avec peu d'images de test, un seul run n'est pas significatif.

| Expérience | Accuracy backbone | Accuracy latent |
|---|---|---|
| Baseline (réseau non entraîné) | – | – |
| Modèle complet | – | – |
| Sans augmentation de couleur | – | – |
| Sans déformation géométrique | – | – |
| Sans fond aléatoire | – | – |

**Protocole de test :** 1 image d'entraînement par classe, N images de test par classe , images redimensionnées en 64×64.

## Limites

- Testé uniquement sur un problème binaire (chat / chien), avec un petit nombre d'images de test.
- Les augmentations apprennent l'invariance à la couleur et à la déformation, mais **pas la variabilité sémantique** (races, poses, point de vue) que seule une image ne contient pas.
- La déformation est appliquée à l'image entière : la forme n'est pas extraite explicitement (pas de segmentation).
- Une déformation élastique trop forte peut replier l'objet et produire des images non réalistes.
- Aucune comparaison à des baselines pré-entraînées (CLIP, DINOv2) pour l'instant.

## Pistes de recherche

- [ ] Ablations systématiques (couleur, géométrie, fond, taille du ViT)
- [ ] Extraction explicite de la forme (superpixels, segmentation) et **réseau de graphes** (GNN)
- [ ] Capsule Networks et réseaux équivariants
- [ ] Plus de classes et d'autres jeux de données (CIFAR, Oxford-IIIT Pets)
- [ ] Comparaison avec un encodeur pré-entraîné (CLIP, DINOv2)
- [ ] Augmentations apprises (déformations générées par un réseau)

## Travaux liés

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR)
- Khosla et al., *Supervised Contrastive Learning*
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT)
- Sabour et al., *Dynamic Routing Between Capsules*

## Citation

```bibtex
@misc{one_shot_latent,
  title  = {one-shot-latent: One-image-per-class learning with a CNN-free ViT},
  author = {gedeon banyi bantu},
  year   = {2026},
  url    = {https://github.com/<ton-utilisateur>/one-shot-latent}
}
```

## Licence

MIT. Voir le fichier `LICENSE`.