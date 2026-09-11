import time
from datetime import datetime, timedelta


def afficher_ecart_16h():
    print("Calcul de l'écart avec 16h00 en cours... (Ctrl+C pour quitter)\n")

    while True:
        # 1. Obtenir l'heure actuelle
        maintenant = datetime.now()

        # 2. Définir la cible à 16h00 pour aujourd'hui
        cible_16h = maintenant.replace(hour=16, minute=0, second=0, microsecond=0)

        # 3. Calculer la différence absolue
        if maintenant > cible_16h:
            difference = maintenant - cible_16h
            prefixe = "Passé de"
        else:
            difference = cible_16h - maintenant
            prefixe = "Restant"

        # 4. Extraire les minutes et secondes totales
        secondes_totales = int(difference.total_seconds())
        minutes = secondes_totales // 60
        secondes = secondes_totales % 60

        # 5. Formater un affichage élégant
        # \r remet le curseur au début de la ligne, end="" évite le saut de ligne
        print(f"\r🕒 [{maintenant.strftime('%H:%M:%S')}] -> {prefixe} : {minutes}min {secondes:02d}s", end="", flush=True)

        # 6. Pause de 5 secondes
        time.sleep(5)


if __name__ == "__main__":
    try:
        afficher_ecart_16h()
    except KeyboardInterrupt:
        print("\n\n👋 Script arrêté avec succès.")
