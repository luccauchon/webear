import random
import time
import requests
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="",
        description=""
    )
    parser.add_argument("--production-setup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-file",type=str,default="sp500_companies.html",help="")
    return parser.parse_args()


def entry(args):
    # L'URL de la page Wikipédia à télécharger
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Configuration des en-têtes pour imiter un navigateur réel (Chrome sur Windows)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    # Simulation d'un temps de réflexion humain (entre 2 et 5 secondes) avant de lancer la requête
    delay = random.uniform(2.0, 5.0)
    print(f"Navigation vers le site... (attente simulée de {delay:.2f} secondes)")
    time.sleep(delay)

    try:
        # Envoi de la requête GET avec les en-têtes configurés
        response = requests.get(url, headers=headers, timeout=10)

        # Vérification que le téléchargement a réussi (Statut 200)
        if response.status_code == 200:
            print("Page téléchargée avec succès !")
            # Sauvegarde du contenu HTML dans un fichier local
            with open(args.output_file, "w", encoding="utf-8") as file:
                file.write(response.text)
            print("Le fichier a été enregistré sous le nom 'sp500_companies.html'.")
        else:
            print(
                f"Échec du téléchargement. Code d'erreur du serveur : {response.status_code}"
            )

    except requests.exceptions.RequestException as e:
        print(f"Une erreur est survenue lors de la connexion : {e}")


if __name__ == "__main__":
    args = parse_args()
    entry(args)