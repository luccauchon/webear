#!/bin/bash

# Activer la gestion de Conda dans le script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Boucle de 2 à 25 incluse
for i in {2..25}
 eagle-eyedi
do
    echo "Lancement de DAY ATR $i"
    
    # Active l'environnement et lance le script en arrière-plan
    (
        conda activate PY312_HT && \
        cd ../../../src/runners && \
        python ./atr_backtesting.py --dataset-id day --step-back-range 3600 --atr-window "$i" --tightness-weight 0.66 --n-trials 1500 --n-split 0.8 --use-close-for-range
    ) &

    # Pause de 5 secondes entre chaque lancement
    sleep 5
done

# Attend que tous les processus en arrière-plan se terminent
wait
