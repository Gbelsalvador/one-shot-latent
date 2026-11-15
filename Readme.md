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
1 image / classe (PNG détouré, RGBA)
      │
      ├─► échange de palette entre classes   (le chat reçoit les couleurs du chien…)
      ├─► transformation de couleur          (teinte, saturation, canaux, contraste…)
      ├─► déformation géométrique            (affine + champ élastique, sans découpage)
      ├─► vues locales                       (zoom sur une partie de l'objet)
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
| **Encodeur** | ViT minimal : patchs via `unfold` + couche `Linear` (aucun `Conv2d`), 4 blocs, dimension 128 (≈ 0,59 M de paramètres) |
| **Couleur** | Rotation de teinte (espace YIQ), saturation, permutation des canaux, contraste, luminosité, niveaux de gris (rare) |
| **Échange de palette** | Les statistiques couleur (moyenne, écart-type par canal) d'une classe sont appliquées à l'autre : la couleur ne peut plus servir d'indice de classe |
| **Géométrie** | Transformation affine aléatoire + champ de déformation élastique lisse (grille 4×4 interpolée en bicubique). La forme est déformée, jamais découpée |
| **Vues locales** | Zoom ×1,5 à ×2,6 centré sur un pixel de l'objet (tiré selon le masque alpha) : force l'apprentissage de features de parties (oreille, pelage, museau) |
| **Fond** | Fonds aléatoires pour les PNG transparents ; padding par **réflexion** pour les images opaques (évite des bordures artificielles jamais vues au test) |
| **Vues propres** | 10 % des échantillons d'entraînement ne sont pas augmentés, pour réduire l'écart train/test |
| **Perte** | *Supervised Contrastive Loss* : les variantes d'une même classe sont rapprochées, les classes distinctes sont éloignées |
| **Évaluation** | Prototype de classe = moyenne des embeddings de vues augmentées + embedding de l'image propre ; classification par similarité cosinus |

Le script évalue deux couches : la représentation du **backbone** et le vecteur **latent** projeté.

## Installation

```bash
git clone https://github.com/gbelsalvador/one-shot-latent.git
cd one-shot-latent
pip install torch numpy pillow
pip install "rembg[cpu]"      # uniquement pour détourer les images (remove_bg.py)
```

## Utilisation

### 1. Préparer les données

Place tes images d'origine dans `data_raw/` :

```
data_raw/
├── train/
│   ├── chat.jpg           # le nom du fichier = nom de la classe
│   └── chien.jpg
└── test/
    ├── chat/              # images de test de la classe « chat »
    │   ├── 01.jpg ...
    └── chien/
        ├── 01.jpg ...
```

### 2. Détourer les fonds

Avec 2 images d'entraînement, le fond est un raccourci très tentant pour le modèle (voir [Constats](#constats-préliminaires-v1)). `remove_bg.py` détoure l'animal avec `rembg` et le recadre dans un carré transparent. Il conserve l'arborescence (`chat/`, `chien/`).

```bash
python remove_bg.py --inp data_raw/train --out data/train
python remove_bg.py --inp data_raw/test  --out data/test
```

> Au premier lancement, `rembg` télécharge un modèle d'environ 176 Mo (la barre de progression peut afficher une vitesse absurde : c'est un défaut d'affichage). Vérifie visuellement quelques PNG générés : l'animal doit être entier, sur fond transparent.

### 3. Entraîner

```bash
# Test rapide + aperçu des augmentations
python one_shot_latent.py --train_dir data/train --test_dir data/test --steps 100 --seeds 0

# Expérience complète (5 graines, moyenne ± écart-type)
python one_shot_latent.py --train_dir data/train --test_dir data/test --seeds 0 1 2 3 4
```

**Regarde toujours `runs_v2/seed0/aug_preview.png` avant de lancer une longue expérience** : l'animal doit apparaître sur des fonds variés, avec des vues locales, sans artefacts évidents.

### Protocoles d'évaluation

| Protocole | Train | Test | Commande |
|---|---|---|---|
| **A : objet seul** | détouré | détouré | `--test_dir data/test` |
| **B : scène réelle** | détouré | images originales avec fond | `--test_dir data/test_raw` (copie de `data_raw/test`) |

L'écart entre A et B mesure à quel point le fond gêne encore le modèle. Rapporte les deux.

### Ablations

Chaque augmentation peut être désactivée individuellement :

```bash
python one_shot_latent.py --no_swap  --seeds 0 1 2 3 4 --out runs_noswap
python one_shot_latent.py --no_local --seeds 0 1 2 3 4 --out runs_nolocal
```

### Options

| Option | Défaut | Description |
|---|---|---|
| `--seeds` | `0` | Liste de graines ; le résumé donne moyenne ± écart-type |
| `--no_color` / `--no_geo` / `--no_bg` / `--no_local` / `--no_swap` | – | Désactive une augmentation (ablation) |
| `--p_local` | 0.4 | Probabilité d'une vue locale |
| `--elastic_sigma` | 0.10 | Amplitude de la déformation élastique |
| `--clean_frac` | 0.1 | Fraction de vues non augmentées |
| `--size` / `--patch` | 64 / 8 | Taille des images et des patchs |
| `--dim` / `--depth` / `--heads` | 128 / 4 / 4 | Taille du ViT |
| `--steps` | 1500 | Pas d'entraînement |
| `--views` | 64 | Variantes par classe et par pas |
| `--temp` / `--lr` | 0.1 / 3e-4 | Température SupCon / taux d'apprentissage |
| `--eval_every` | 500 | Fréquence d'évaluation |
| `--out` | `runs_v2` | Dossier de sortie |

### Sorties

- `runs_v2/seed<N>/aug_preview.png` : aperçu des augmentations
- `runs_v2/seed<N>/model.pt` : poids du modèle
- `runs_v2/summary.json` : baseline et résultats finaux, moyenne ± écart-type
- Console : perte, accuracy (backbone et latent) et baseline d'un réseau non entraîné

## Constats préliminaires (v1)

La première version (images JPG avec fond, augmentations plus agressives, une seule graine, 23 images de test) a donné ceci :

| Étape | Accuracy backbone | Accuracy latent |
|---|---|---|
| Réseau non entraîné | 60,9 % | 65,2 % |
| Après 1500 pas | 52,2 % | 52,2 % |

Deux observations :

1. **La loss atteint son minimum théorique** (≈ 4,143 = ln(63) pour 64 vues par classe) dès quelques centaines de pas, puis stagne : avec deux images, la tâche contrastive est résolue par des indices triviaux (fond, silhouette exacte) qui ne se transfèrent pas à de nouvelles images.
2. **L'accuracy est au niveau du hasard** (52 %). Avec 23 images de test, l'incertitude est d'environ ±10 points : la baseline à 60-65 % n'est pas significativement différente.

L'aperçu des augmentations a aussi révélé que le fond d'origine restait collé à l'objet, que certaines vues étaient quasi détruites (sombres, floues, bruitées) et que des bordures colorées apparaissaient à l'entraînement sans jamais exister au test. La v2 corrige ces points (détourage, augmentations plus douces, padding par réflexion, vues propres) et ajoute les vues locales et l'échange de palette pour rendre la tâche moins triviale.

## Résultats (v2)


| Expérience | Protocole | Accuracy backbone | Accuracy latent |
|---|---|---|---|
| Baseline (réseau non entraîné) | A | – | – |
| Modèle complet | A | – | – |
| Modèle complet | B | – | – |
| `--no_color` | A | – | – |
| `--no_geo` | A | – | – |
| `--no_bg` | A | – | – |
| `--no_local` | A | – | – |
| `--no_swap` | A | – | – |

**Protocole de test :** 1 image d'entraînement par classe, N images de test par classe (à préciser), images redimensionnées en 64×64.

## Limites

- Testé uniquement sur un problème binaire (chat / chien), avec un petit nombre d'images de test.
- Les augmentations apprennent l'invariance à la couleur et à la déformation, mais **pas la variabilité sémantique** (races, poses, point de vue) que deux images ne contiennent pas.
- La qualité du détourage (`rembg`) influence directement les résultats ; les erreurs de segmentation ne sont pas corrigées.
- La déformation est appliquée à l'image entière : la forme n'est pas extraite explicitement comme primitive géométrique.
- Une déformation élastique trop forte peut replier l'objet et produire des images non réalistes.
- Aucune comparaison à des baselines pré-entraînées (CLIP, DINOv2) pour l'instant.

## Pistes de recherche

- [x] Ablations par augmentation (`--no_*`) et évaluation multi-graines
- [ ] Étudier la taille minimale de test nécessaire pour détecter un effet significatif
- [ ] Extraction explicite de la forme (superpixels, segmentation) et **réseau de graphes** (GNN)
- [ ] Capsule Networks et réseaux équivariants
- [ ] Plus de classes et d'autres jeux de données (CIFAR, Oxford-IIIT Pets)
- [ ] Comparaison avec un encodeur pré-entraîné (CLIP, DINOv2) comme borne haute
- [ ] Augmentations apprises (déformations générées par un réseau)

## Structure du dépôt

```
one-shot-latent/
├── one_shot_latent.py   # entraînement + évaluation (v2)
├── remove_bg.py         # détourage des fonds (rembg)
├── data_raw/            # images originales (non versionnées)
├── data/                # images détourées (non versionnées)
├── runs_v2/             # sorties
└── README.md
```



## Travaux liés

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR)
- Khosla et al., *Supervised Contrastive Learning*
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT)
- Sabour et al., *Dynamic Routing Between Capsules*
- Gatis, *rembg* (détourage de fond)

## Citation

```bibtex
@misc{one_shot_latent,
  title  = {one-shot-latent: One-image-per-class learning with a CNN-free ViT},
  author = {gedeon banyi bantu},
  year   = {2026},
  url    = {https://github.com/gbelsalvador/one-shot-latent}
}
```

## Licence

MIT. Voir le fichier `LICENSE`.