2026.07.23: tripler le temps de Optuna --> ça a causer de l'overfitting
atr est meilleur que atr+vix+momentum sur le spectre du tightness mais atr+vix+momentum est meilleur pour le tightness 0.~0.33. On peut dire que jusqu'à tightness=0.66,
les deux sont à peut près égale, très léger meilleur score pour atr+vix+momentum.

Recommendations: utiliser ATR et atr+vix+momentum pour des très petits tightness