
import requests
import time
from PIL import Image
from io import BytesIO
from pathlib import Path

BASE_DIR = Path("data/image")
NOMBRE_IMAGES = 13
TAILLE = 224

API_URL = "https://commons.wikimedia.org/w/api.php"

session = requests.Session()
session.headers.update({
    "User-Agent": "AnimalDatasetCollector/1.0 (educational project)"
})


def requete(url, params=None, max_retries=4):
    """Requête HTTP avec gestion des limitations 429."""

    for tentative in range(max_retries):
        try:
            response = session.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                attente = 5 * (2 ** tentative)
                print(f"[429] Pause de {attente}s...")
                time.sleep(attente)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as erreur:
            if tentative == max_retries - 1:
                print(f"[ERREUR] {erreur}")
                return None

            time.sleep(3 * (tentative + 1))

    return None


def scraper_images(animal, recherche):

    dossier = BASE_DIR / animal
    dossier.mkdir(parents=True, exist_ok=True)

    # Conserver les images déjà téléchargées
    compteur = len(list(dossier.glob("*.png")))

    if compteur >= NOMBRE_IMAGES:
        print(f"{animal}: Dataset déjà complet.")
        return

    print(f"\nRecherche : {recherche}")
    print(f"Images existantes : {compteur}")

    continuation = {}
    images_traitees = set()
    erreurs_consecutives = 0

    while compteur < NOMBRE_IMAGES:

        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {recherche}",
            "gsrnamespace": 6,
            "gsrlimit": 30,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 1024,
            **continuation
        }

        response = requete(API_URL, params)

        if response is None:
            print("Impossible de contacter l'API.")
            break

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        if not pages:
            print("Plus aucun résultat.")
            break

        for page in pages.values():

            if compteur >= NOMBRE_IMAGES:
                break

            infos = page.get("imageinfo", [])

            if not infos:
                continue

            # Utiliser la miniature plutôt que l'original
            image_url = infos[0].get("thumburl")

            if not image_url:
                image_url = infos[0].get("url")

            if not image_url or image_url in images_traitees:
                continue

            images_traitees.add(image_url)

            img_response = requete(image_url)

            if img_response is None:
                erreurs_consecutives += 1

                if erreurs_consecutives >= 4:
                    print("Trop d'erreurs. Arrêt temporaire.")
                    return

                continue

            try:
                image = Image.open(
                    BytesIO(img_response.content)
                )

                image.load()

                if image.width < 100 or image.height < 100:
                    continue

                image = image.convert("RGB")
                image = image.resize(
                    (TAILLE, TAILLE),
                    Image.Resampling.LANCZOS
                )

                compteur += 1

                nom = f"{animal[:-1]}_{compteur:02d}.png"
                chemin = dossier / nom

                image.save(chemin, format="PNG")

                erreurs_consecutives = 0

                print(f"[OK] {nom} - 224x224")

            except Exception as erreur:
                print(f"[IGNORÉ] {erreur}")

            # Réduire la pression sur Wikimedia
            time.sleep(2.5)

        continuation = data.get("continue", {})

        if not continuation:
            break

    print(f"\nRésultat : {compteur}/{NOMBRE_IMAGES}")


def main():

    BASE_DIR.mkdir(exist_ok=True)

    scraper_images("chats", "domestic cat")
    scraper_images("chiens", "domestic dog")

    print("\nCollecte terminée !")


if __name__ == "__main__":
    main()
