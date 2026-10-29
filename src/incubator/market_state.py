try:
    from version import sys__name, sys__version
except Exception:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent

    # Add the parent directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import sys
import numpy as np
import pandas as pd
from fetchers.data_factory import factory_load_data
import argparse
from argparse import Namespace


# ==========================================================
# Normalisation commune des regime_id / direction_id
# ==========================================================
# Objectif :
# - garantir que regime_id et direction_id ont la même signification
#   pour les 4 indicateurs.
#
# regime_id :
# - 0 : range / marché sans tendance nette
# - 1 : transition / régime intermédiaire / tendance modérée
# - 2 : tendance établie / tendance forte
#
# direction_id :
# - -1 : baissier
# -  0 : neutre / pas de direction exploitable
# -  1 : haussier
# ==========================================================
MARKET_STATE__REGIME_RANGE_ID = 0
MARKET_STATE__REGIME_TRANSITION_ID = 1
MARKET_STATE__REGIME_TREND_ID = 2

MARKET_STATE__DIRECTION_BEARISH_ID = -1
MARKET_STATE__DIRECTION_NEUTRAL_ID = 0
MARKET_STATE__DIRECTION_BULLISH_ID = 1


# ==========================================================
# Seuils d'interprétation
# ==========================================================
# Ces seuils sont indicatifs et peuvent être adaptés selon
# l'actif, l'unité de temps et la stratégie utilisée.
#
# ADX :
# - < 20  : range / tendance faible
# - 20-25 : transition / début ou fin possible de tendance
# - 25-50 : tendance établie
# - > 50  : tendance très forte, parfois excessive
#
# Choppiness Index :
# - < 38.2 : marché plutôt tendance
# - > 61.8 : marché plutôt range / haché
# - entre les deux : zone incertaine / transition
#
# Directional ATR Ratio :
# indicateur custom.
# - proche de 0 : pas de direction nette
# - > 0 : pression haussière
# - < 0 : pression baissière
#
# KAMA :
# - prix > KAMA et KAMA croissant : biais haussier
# - prix < KAMA et KAMA décroissant : biais baissier
# - KAMA plat : marché plus probable en range
# ==========================================================
MARKET_STATE__ADX_RANGE_MAX = 20.0
MARKET_STATE__ADX_TRANSITION_MAX = 25.0
MARKET_STATE__ADX_VERY_STRONG_MIN = 50.0

# Seuil arbitraire pour considérer que la pente du KAMA est "plate".
# Exprimé en % du prix de clôture.
MARKET_STATE__KAMA_FLAT_PCT = 0.05

MARKET_STATE__CHOPPINESS_TREND_MAX = 38.2
MARKET_STATE__CHOPPINESS_RANGE_MIN = 61.8

# Seuils arbitraires pour le Directional ATR Ratio.
MARKET_STATE__DIR_ATR_RANGE_MAX = 0.15
MARKET_STATE__DIR_ATR_STRONG_MIN = 0.30

# Seuils pour les conditions de trading
MARKET_STATE__CHOPPINESS_STRONG_DROP_POINTS = 5.0
MARKET_STATE__DIR_ATR_IMPROVEMENT_THRESHOLD = 0.10


def compute_indicators(df, verbose):
    """
    Calcule 4 indicateurs de tendance / régime de marché :
    1) ADX / +DI / -DI
    2) KAMA
    3) Choppiness Index
    4) Directional ATR Ratio
    """

    # ==========================================================
    # Nettoyage des colonnes MultiIndex si présentes
    # ==========================================================
    # Certaines sources de données renvoient des colonnes MultiIndex
    # du type ('Close', 'SPX'). On ne garde que le premier niveau.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Trier par date/index pour garantir un calcul chronologique correct
    df = df.sort_index()

    # ==========================================================
    # 1) Directional Movement : ADX, +DI, -DI
    # ==========================================================
    # Objectif :
    # - +DI et -DI mesurent la pression directionnelle haussière/baissière.
    # - ADX mesure la force de la tendance, sans sa direction.
    #
    # Lecture classique :
    # - ADX faible  -> marché en range ou tendance molle
    # - ADX élevé   -> marché en tendance
    # - +DI > -DI   -> dominante haussière
    # - -DI > +DI   -> dominante baissière
    # ==========================================================
    def compute_DM(df: pd.DataFrame) -> pd.DataFrame:
        high = df['High']
        low = df['Low']
        close = df['Close']

        period = 14

        # ==================================================
        # True Range
        # ==================================================
        # Le True Range mesure la volatilité réelle d'une période
        # en tenant compte des gaps.
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ==================================================
        # Directional Movement
        # ==================================================
        # up_move : mouvement haussier potentiel
        # down_move : mouvement baissier potentiel
        up_move = high.diff()
        down_move = low.shift(1) - low

        # +DM : uniquement si le mouvement haussier est positif
        # et supérieur au mouvement baissier.
        plus_dm = up_move.where(
            (up_move > down_move) & (up_move > 0),
            0.0
        )

        # -DM : uniquement si le mouvement baissier est positif
        # et supérieur au mouvement haussier.
        minus_dm = down_move.where(
            (down_move > up_move) & (down_move > 0),
            0.0
        )

        # ==================================================
        # Lissage Wilder
        # ==================================================
        # Le lissage Wilder est équivalent à un EMA avec alpha = 1 / n.
        def wilder_smooth(series, n, min_periods=1):
            if not isinstance(series, pd.Series):
                series = pd.Series(series, index=df.index)

            return series.ewm(
                alpha=1 / n,
                adjust=False,
                min_periods=min_periods
            ).mean()

        smoothed_tr = wilder_smooth(tr, period, min_periods=1)
        smoothed_plus_dm = wilder_smooth(plus_dm, period, min_periods=1)
        smoothed_minus_dm = wilder_smooth(minus_dm, period, min_periods=1)

        # ==================================================
        # Plus DI / Minus DI
        # ==================================================
        # Sécurité contre division par zéro
        smoothed_tr_safe = smoothed_tr.replace(0, np.nan)

        plus_di = 100 * (smoothed_plus_dm / smoothed_tr_safe)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr_safe)

        plus_di = plus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        minus_di = minus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ==================================================
        # DX puis ADX
        # ==================================================
        # DX mesure la différence relative entre +DI et -DI.
        # ADX est ensuite un lissage du DX.
        di_sum = plus_di + minus_di

        dx = 100 * ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan))

        # Quand Plus_DI + Minus_DI == 0, DX est indéterminé.
        # On le met à 0 pour éviter les NaN.
        dx = dx.fillna(0.0)
        dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        adx = wilder_smooth(dx, period, min_periods=1)

        # ==================================================
        # Stockage dans df
        # ==================================================
        df['ADX'] = adx
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        return df

    # ==========================================================
    # 2) KAMA : Kaufman Adaptive Moving Average
    # ==========================================================
    # Objectif :
    # - Le KAMA s'adapte à la tendance.
    # - Il devient plus réactif quand le marché est directionnel.
    # - Il devient plus lisse quand le marché est bruité / range.
    #
    # Lecture :
    # - Prix au-dessus d'un KAMA croissant : biais haussier.
    # - Prix en-dessous d'un KAMA décroissant : biais baissier.
    # - KAMA plat : marché probablement en range ou sans direction claire.
    # ==========================================================
    def compute_KAMA(df: pd.DataFrame) -> pd.DataFrame:
        close = df['Close']

        period = 14

        # Efficiency Ratio :
        # mesure l'efficacité du mouvement.
        # Plus ER est proche de 1, plus le mouvement est directionnel.
        # Plus ER est proche de 0, plus le mouvement est bruité.
        change = (close - close.shift(period)).abs()
        volatility = (close - close.shift(1)).abs().rolling(window=period).sum()

        # Sécurité division par zéro
        er = change / volatility.replace(0, np.nan)
        er = er.fillna(0.0)

        # Constantes de lissage
        # Fast = 2, Slow = 30
        fast_sc = 2.0 / (2.0 + 1.0)
        slow_sc = 2.0 / (30.0 + 1.0)

        # Smoothing constant finale
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Calcul récursif du KAMA
        close_values = close.to_numpy(dtype=float)
        sc_values = sc.to_numpy(dtype=float)
        kama = np.full(len(close_values), np.nan, dtype=float)

        # Initialisation après la première période complète
        first_valid_idx = period

        if len(close_values) > first_valid_idx:
            kama[first_valid_idx] = close_values[first_valid_idx]

            for i in range(first_valid_idx + 1, len(close_values)):
                # Si la valeur précédente est NaN, on réinitialise
                if np.isnan(kama[i - 1]):
                    kama[i] = close_values[i]
                    continue

                price = close_values[i]

                # Si prix manquant, on reconduit le KAMA précédent
                if np.isnan(price):
                    kama[i] = kama[i - 1]
                    continue

                smoothing = sc_values[i]

                if np.isnan(smoothing):
                    smoothing = 0.0

                kama[i] = kama[i - 1] + smoothing * (price - kama[i - 1])

        df['KAMA'] = kama

        return df

    # ==========================================================
    # 3) Choppiness Index
    # ==========================================================
    # Objectif :
    # - Mesurer si le marché est "tendance" ou "haché / range".
    # - C'est un indicateur de régime, pas un indicateur directionnel.
    #
    # Lecture classique :
    # - valeurs basses  : marché tendance
    # - valeurs hautes  : marché range / chaotique
    # - seuils souvent utilisés : 38.2 et 61.8
    # ==========================================================
    def compute_CHOPPINESS(df: pd.DataFrame) -> pd.DataFrame:
        high = df['High']
        low = df['Low']
        close = df['Close']

        period = 14

        # ==================================================
        # True Range
        # ==================================================
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Somme du True Range sur N périodes
        sum_tr = tr.rolling(window=period).sum()

        # Plus haut et plus bas sur N périodes
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # Amplitude globale high-low sur la période
        hl_range = (highest_high - lowest_low).replace(0, np.nan)

        # Ratio entre volatilité cumulée et amplitude directionnelle
        ratio = sum_tr / hl_range
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        # Le log10 n'est défini que pour des valeurs strictement positives
        ratio = ratio.where(ratio > 0, np.nan)

        # Formule du Choppiness Index
        choppiness = (
            100.0
            * np.log10(ratio)
            / np.log10(period)
        )

        # Nettoyage :
        # - bornes classiques entre 0 et 100
        # - valeur neutre 50 si indéterminé
        df['Choppiness'] = (
            choppiness
            .replace([np.inf, -np.inf], np.nan)
            .clip(0.0, 100.0)
            .fillna(50.0)
        )

        return df

    # ==========================================================
    # 4) Directional ATR Ratio
    # ==========================================================
    # Indicateur custom.
    #
    # Objectif :
    # - comparer le mouvement net du prix sur une période
    #   à la volatilité cumulée sur cette même période.
    #
    # Lecture :
    # - ratio proche de 0 : mouvement net faible, marché probablement en range
    # - ratio positif : le mouvement net est haussier
    # - ratio négatif : le mouvement net est baissier
    # - plus la valeur absolue est élevée, plus le mouvement est directionnel
    # ==========================================================
    def compute_DIRECTIONAL_ATR_RATIO(df: pd.DataFrame) -> pd.DataFrame:
        high = df['High']
        low = df['Low']
        close = df['Close']

        period = 14

        # ==================================================
        # True Range
        # ==================================================
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR simple : moyenne du True Range sur la période
        atr = tr.rolling(window=period).mean()

        # Mouvement net du prix sur la période
        net_move = close - close.shift(period)

        # Volatilité cumulée approximative sur la période
        denominator = (atr * period).replace(0, np.nan)

        # Ratio directionnel
        directional_atr_ratio = net_move / denominator

        df['Directional_ATR_Ratio'] = (
            directional_atr_ratio
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        return df

    # ==========================================================
    # Calcul final des indicateurs
    # ==========================================================
    df = compute_DM(df=df)
    df = compute_KAMA(df=df)
    df = compute_CHOPPINESS(df=df)
    df = compute_DIRECTIONAL_ATR_RATIO(df=df)

    return df


# ==========================================================
# Helpers pour l'affichage
# ==========================================================

def _format_value(value, ndigits: int = 2) -> str:
    """
    Formate proprement une valeur numérique pour les prints.
    """
    if value is None:
        return "NaN"

    try:
        if pd.isna(value):
            return "NaN"
        return f"{value:.{ndigits}f}"
    except Exception:
        return str(value)


def _bias_to_direction_id(bias: str) -> int:
    """
    Convertit un biais textuel en direction normalisée.

    - "haussier"  ->  1
    - "baissier"  -> -1
    - autre       ->  0
    """
    if bias == "haussier":
        return MARKET_STATE__DIRECTION_BULLISH_ID
    if bias == "baissier":
        return MARKET_STATE__DIRECTION_BEARISH_ID
    return MARKET_STATE__DIRECTION_NEUTRAL_ID


# ==========================================================
# Print général : à quoi s'attendre pour les valeurs
# ==========================================================

def print_indicator_expectations():
    """
    Affiche une explication générale des valeurs attendues
    pour chaque indicateur.
    """
    print("=" * 80)
    print("INTERPRÉTATION DES 4 INDICATEURS DE TENDANCE")
    print("=" * 80)

    print("\nNormalisation commune utilisée pour les scripts externes :")
    print(" regime_id :")
    print(f"   {MARKET_STATE__REGIME_RANGE_ID} = range / pas de tendance nette")
    print(f"   {MARKET_STATE__REGIME_TRANSITION_ID} = transition / régime intermédiaire / tendance modérée")
    print(f"   {MARKET_STATE__REGIME_TREND_ID} = tendance établie / tendance forte")
    print(" direction_id :")
    print(f"   {MARKET_STATE__DIRECTION_BEARISH_ID} = baissier")
    print(f"   {MARKET_STATE__DIRECTION_NEUTRAL_ID} = neutre / pas de direction exploitable")
    print(f"   {MARKET_STATE__DIRECTION_BULLISH_ID} = haussier")

    print("\n1) ADX / +DI / -DI")
    print("   - ADX < 20       : marché probablement en range ou tendance très faible.")
    print("   - ADX 20 à 25    : phase de transition, début ou fin possible de tendance.")
    print("   - ADX 25 à 50    : tendance établie.")
    print("   - ADX > 50       : tendance très forte, parfois extension excessive.")
    print("   - +DI > -DI      : pression haussière.")
    print("   - -DI > +DI      : pression baissière.")

    print("\n2) KAMA")
    print("   - Prix > KAMA et KAMA croissant : biais haussier.")
    print("   - Prix < KAMA et KAMA décroissant : biais baissier.")
    print("   - KAMA plat : marché probablement en range ou sans direction claire.")
    print("   - Le KAMA s'accélère en tendance et devient plus lisse en range.")

    print("\n3) Choppiness Index")
    print("   - Valeur < 38.2  : marché plutôt en tendance.")
    print("   - Valeur > 61.8  : marché plutôt en range / haché.")
    print("   - Entre 38.2 et 61.8 : zone intermédiaire / transition.")
    print("   - Cet indicateur ne donne pas la direction, seulement le régime.")
    print("   - direction_id est donc normalisé à 0 pour cet indicateur.")

    print("\n4) Directional ATR Ratio")
    print("   - Ratio proche de 0 : mouvement directionnel faible, range probable.")
    print("   - Ratio > 0 : pression haussière.")
    print("   - Ratio < 0 : pression baissière.")
    print(f"   - Seuil de range utilisé : valeur absolue < {MARKET_STATE__DIR_ATR_RANGE_MAX:.2f}.")
    print(f"   - Seuil de tendance forte utilisé : valeur absolue >= {MARKET_STATE__DIR_ATR_STRONG_MIN:.2f}.")

    print("\nRemarque : les seuils sont indicatifs et doivent être adaptés")
    print("à l'actif, à l'unité de temps et à la stratégie.")

    print("\nConditions de trading :")
    print("   Chaque indicateur retourné par compute_trading_conditions contient")
    print("   désormais une valeur 'direction' normalisée :")
    print(f"   {MARKET_STATE__DIRECTION_BEARISH_ID} = baisse")
    print(f"   {MARKET_STATE__DIRECTION_NEUTRAL_ID} = range / neutre / à confirmer")
    print(f"   {MARKET_STATE__DIRECTION_BULLISH_ID} = hausse")
    print("-" * 80)


def compute_adx_diagnosis(df: pd.DataFrame, verbose):
    """
    Diagnose la tendance actuelle à partir de ADX, +DI et -DI.

    Retourne :
    - regime_id : 0 range, 1 transition, 2 tendance
    - direction_id : -1 baissier, 0 neutre, 1 haussier
    """
    required_columns = ['ADX', 'Plus_DI', 'Minus_DI']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        if verbose:
            print(f"[ADX/DMI] Colonnes manquantes : {missing_columns}")
        return {}

    data = df.dropna(subset=required_columns)

    if data.empty:
        if verbose:
            print("[ADX/DMI] Données insuffisantes pour calculer l'ADX/DMI.")
        return {}

    date = data.index[-1]
    adx = data['ADX'].iloc[-1]
    plus_di = data['Plus_DI'].iloc[-1]
    minus_di = data['Minus_DI'].iloc[-1]

    # Force de tendance via ADX
    if adx < MARKET_STATE__ADX_RANGE_MAX:
        regime = "marché probablement en range / tendance très faible"
        regime_id = MARKET_STATE__REGIME_RANGE_ID
    elif adx < MARKET_STATE__ADX_TRANSITION_MAX:
        regime = "phase de transition : début ou fin possible de tendance"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
    elif adx < MARKET_STATE__ADX_VERY_STRONG_MIN:
        regime = "tendance établie"
        regime_id = MARKET_STATE__REGIME_TREND_ID
    else:
        regime = "tendance très forte, attention aux excès possibles"
        regime_id = MARKET_STATE__REGIME_TREND_ID

    # Direction via +DI / -DI
    if plus_di > minus_di:
        direction = "pression haussière (+DI > -DI)"
        direction_id = MARKET_STATE__DIRECTION_BULLISH_ID
    elif minus_di > plus_di:
        direction = "pression baissière (-DI > +DI)"
        direction_id = MARKET_STATE__DIRECTION_BEARISH_ID
    else:
        direction = "pression directionnelle neutre (+DI ≈ -DI)"
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    if verbose:
        print("\n[ADX/DMI]")
        print(
            f"Date={date}, "
            f"ADX={_format_value(adx)}, "
            f"+DI={_format_value(plus_di)}, "
            f"-DI={_format_value(minus_di)}"
        )
        print(f" -> Lecture : {regime}, avec {direction}.")
        print(f" -> regime_id={regime_id}, direction_id={direction_id}")

    return {
        "adx": {
            "regime": regime_id,
            "direction": direction_id,
            "date": date
        }
    }


def compute_kama_diagnosis(df: pd.DataFrame, verbose):
    """
    Diagnose la tendance actuelle à partir du KAMA et de sa pente.

    Retourne :
    - regime_id : 0 range, 1 transition, 2 tendance
    - direction_id : -1 baissier, 0 neutre, 1 haussier
    """
    required_columns = ['Close', 'KAMA']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        if verbose:
            print(f"[KAMA] Colonnes manquantes : {missing_columns}")
        return {}

    data = df.dropna(subset=required_columns)

    if data.empty:
        if verbose:
            print("[KAMA] Données insuffisantes pour calculer le KAMA.")
        return {}

    date = data.index[-1]
    close = data['Close'].iloc[-1]
    kama = data['KAMA'].iloc[-1]

    # Calcul de la pente du KAMA
    if len(data) >= 2:
        previous_kama = data['KAMA'].iloc[-2]
        slope = kama - previous_kama

        if close != 0:
            slope_pct = (slope / close) * 100.0
        else:
            slope_pct = 0.0
    else:
        slope = np.nan
        slope_pct = 0.0

    # Interprétation
    if abs(slope_pct) <= MARKET_STATE__KAMA_FLAT_PCT:
        regime = "range / marché sans direction claire (KAMA presque plat)"
        regime_id = MARKET_STATE__REGIME_RANGE_ID
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    elif close > kama and slope_pct > 0:
        regime = "tendance haussière (prix au-dessus du KAMA et KAMA croissant)"
        regime_id = MARKET_STATE__REGIME_TREND_ID
        direction_id = MARKET_STATE__DIRECTION_BULLISH_ID

    elif close < kama and slope_pct < 0:
        regime = "tendance baissière (prix en-dessous du KAMA et KAMA décroissant)"
        regime_id = MARKET_STATE__REGIME_TREND_ID
        direction_id = MARKET_STATE__DIRECTION_BEARISH_ID

    elif close > kama and slope_pct < 0:
        regime = "phase de transition / pullback (prix au-dessus d'un KAMA décroissant)"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
        direction_id = MARKET_STATE__DIRECTION_BEARISH_ID

    elif close < kama and slope_pct > 0:
        regime = "phase de transition / rebond technique (prix en-dessous d'un KAMA croissant)"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
        direction_id = MARKET_STATE__DIRECTION_BULLISH_ID

    else:
        regime = "indécis / en transition"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    if verbose:
        print("\n[KAMA]")
        print(
            f"Date={date}, "
            f"Close={_format_value(close)}, "
            f"KAMA={_format_value(kama)}, "
            f"Pente KAMA={_format_value(slope_pct)}%"
        )
        print(f" -> Lecture : {regime}.")
        print(f" -> regime_id={regime_id}, direction_id={direction_id}")

    return {
        "kama": {
            "regime": regime_id,
            "direction": direction_id,
            "date": date
        }
    }


def compute_choppiness_diagnosis(df: pd.DataFrame, verbose):
    """
    Diagnose le régime de marché à partir du Choppiness Index.

    Retourne :
    - regime_id : 0 range, 1 transition, 2 tendance
    - direction_id : toujours 0, car cet indicateur ne donne pas la direction
    """
    required_columns = ['Choppiness']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        if verbose:
            print(f"[Choppiness Index] Colonnes manquantes : {missing_columns}")
        return {}

    data = df.dropna(subset=required_columns)

    if data.empty:
        if verbose:
            print("[Choppiness Index] Données insuffisantes pour calculer le Choppiness Index.")
        return {}

    date = data.index[-1]
    choppiness = data['Choppiness'].iloc[-1]

    if choppiness < MARKET_STATE__CHOPPINESS_TREND_MAX:
        regime = "marché probablement en tendance (valeur basse)"
        regime_id = MARKET_STATE__REGIME_TREND_ID
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    elif choppiness > MARKET_STATE__CHOPPINESS_RANGE_MIN:
        regime = "marché probablement en range / haché (valeur élevée)"
        regime_id = MARKET_STATE__REGIME_RANGE_ID
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    else:
        regime = "zone intermédiaire : ni tendance claire ni range clair"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    if verbose:
        print("\n[Choppiness Index]")
        print(
            f"Date={date}, "
            f"Choppiness={_format_value(choppiness)}"
        )
        print(f" -> Lecture : {regime}.")
        print(" -> Rappel : cet indicateur ne donne pas la direction, seulement le régime.")
        print(f" -> regime_id={regime_id}, direction_id={direction_id}")

    return {
        "choppiness": {
            "regime": regime_id,
            "direction": direction_id,
            "date": date
        }
    }


def compute_directional_atr_ratio_diagnosis(df: pd.DataFrame, verbose):
    """
    Diagnose la tendance actuelle à partir du Directional ATR Ratio.

    Retourne :
    - regime_id : 0 range, 1 transition, 2 tendance
    - direction_id : -1 baissier, 0 neutre, 1 haussier
    """
    required_columns = ['Directional_ATR_Ratio']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        if verbose:
            print(f"[Directional ATR Ratio] Colonnes manquantes : {missing_columns}")
        return {}

    data = df.dropna(subset=required_columns)

    if data.empty:
        if verbose:
            print("[Directional ATR Ratio] Données insuffisantes pour calculer le ratio.")
        return {}

    date = data.index[-1]
    ratio = data['Directional_ATR_Ratio'].iloc[-1]
    abs_ratio = abs(ratio)

    # Direction du mouvement net
    if ratio > 0:
        direction = "haussière"
        direction_id = MARKET_STATE__DIRECTION_BULLISH_ID
    elif ratio < 0:
        direction = "baissière"
        direction_id = MARKET_STATE__DIRECTION_BEARISH_ID
    else:
        direction = "neutre"
        direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

    # Force du mouvement directionnel
    if abs_ratio < MARKET_STATE__DIR_ATR_RANGE_MAX:
        regime = "range / mouvement directionnel faible"
        regime_id = MARKET_STATE__REGIME_RANGE_ID
    elif abs_ratio < MARKET_STATE__DIR_ATR_STRONG_MIN:
        regime = "tendance modérée"
        regime_id = MARKET_STATE__REGIME_TRANSITION_ID
    else:
        regime = "tendance forte"
        regime_id = MARKET_STATE__REGIME_TREND_ID

    if verbose:
        print("\n[Directional ATR Ratio]")
        print(
            f"Date={date}, "
            f"Ratio={_format_value(ratio, 4)}"
        )
        print(
            f" -> Lecture : direction {direction}, régime : {regime}."
        )
        print(f" -> regime_id={regime_id}, direction_id={direction_id}")

    return {
        "direction_atr_ratio": {
            "direction": direction_id,
            "regime": regime_id,
            "date": date
        }
    }


# ==========================================================
# Conditions de trading
# ==========================================================

def evaluate_trading_conditions(df: pd.DataFrame) -> dict:
    """
    Détermine des conditions de trading simples à partir des dernières bougies.

    Les conditions sont volontairement simples et pédagogiques :
    - ADX qui sort d'une zone basse -> possibilité de début de tendance.
    - ADX qui sort d'une zone haute -> possible essoufflement / fin de tendance excessive.
    - Prix qui passe au-dessus/d'en-dessous d'un KAMA croissant/décroissant.
    - Choppiness qui baisse fortement -> sortie possible de range.
    - Directional ATR Ratio qui devient fortement positif/négatif.

    Retourne un dictionnaire contenant les conditions par indicateur.

    Chaque indicateur contient désormais une clé 'direction' :
    - -1 : baisse
    -  0 : range / neutre / à confirmer
    -  1 : hausse
    """

    signals = {
        'ADX': {
            'label': 'ADX / +DI / -DI',
            'date': None,
            'triggered': False,
            'bias': 'neutre',
            'direction': MARKET_STATE__DIRECTION_NEUTRAL_ID,
            'messages': []
        },
        'KAMA': {
            'label': 'KAMA',
            'date': None,
            'triggered': False,
            'bias': 'neutre',
            'direction': MARKET_STATE__DIRECTION_NEUTRAL_ID,
            'messages': []
        },
        'CHOPPINESS': {
            'label': 'Choppiness Index',
            'date': None,
            'triggered': False,
            'bias': 'à confirmer',
            'direction': MARKET_STATE__DIRECTION_NEUTRAL_ID,
            'messages': []
        },
        'DIRECTIONAL_ATR_RATIO': {
            'label': 'Directional ATR Ratio',
            'date': None,
            'triggered': False,
            'bias': 'neutre',
            'direction': MARKET_STATE__DIRECTION_NEUTRAL_ID,
            'messages': []
        }
    }

    # ------------------------------------------------------
    # Conditions ADX
    # ------------------------------------------------------
    required = ['ADX', 'Plus_DI', 'Minus_DI']
    missing = [col for col in required if col not in df.columns]

    if missing:
        signals['ADX']['messages'].append(f"Colonnes manquantes : {missing}")
    else:
        data = df[required].dropna()

        if len(data) < 2:
            signals['ADX']['messages'].append(
                "Pas assez de bougies valides pour évaluer les conditions ADX."
            )
        else:
            prev = data.iloc[-2]
            curr = data.iloc[-1]
            date = data.index[-1]

            signals['ADX']['date'] = date

            prev_adx = float(prev['ADX'])
            curr_adx = float(curr['ADX'])

            prev_plus_di = float(prev['Plus_DI'])
            prev_minus_di = float(prev['Minus_DI'])

            curr_plus_di = float(curr['Plus_DI'])
            curr_minus_di = float(curr['Minus_DI'])

            signals['ADX']['messages'].append(
                f"ADX précédent : {_format_value(prev_adx)} | "
                f"ADX actuel : {_format_value(curr_adx)}"
            )

            signals['ADX']['messages'].append(
                f"+DI actuel : {_format_value(curr_plus_di)} | "
                f"-DI actuel : {_format_value(curr_minus_di)}"
            )

            # Direction actuelle via DI
            if curr_plus_di > curr_minus_di:
                di_direction = "pression haussière (+DI > -DI)"
                di_bias = "haussier"
                di_direction_id = MARKET_STATE__DIRECTION_BULLISH_ID
            elif curr_minus_di > curr_plus_di:
                di_direction = "pression baissière (-DI > +DI)"
                di_bias = "baissier"
                di_direction_id = MARKET_STATE__DIRECTION_BEARISH_ID
            else:
                di_direction = "pression neutre (+DI ≈ -DI)"
                di_bias = "neutre"
                di_direction_id = MARKET_STATE__DIRECTION_NEUTRAL_ID

            signals['ADX']['messages'].append(f"Direction DI actuelle : {di_direction}.")
            signals['ADX']['messages'].append(f"Direction DI numérique : {di_direction_id}.")

            # Cross DI éventuel
            di_bull_cross = (prev_plus_di <= prev_minus_di) and (curr_plus_di > curr_minus_di)
            di_bear_cross = (prev_plus_di >= prev_minus_di) and (curr_minus_di > curr_plus_di)

            if di_bull_cross:
                signals['ADX']['messages'].append("Cross haussier +DI/-DI détecté.")

            if di_bear_cross:
                signals['ADX']['messages'].append("Cross baissier -DI/+DI détecté.")

            adx_low_condition = False
            adx_high_condition = False

            # Condition principale : ADX sort d'une zone basse
            adx_exited_low_zone = (
                prev_adx < MARKET_STATE__ADX_RANGE_MAX and
                curr_adx >= MARKET_STATE__ADX_RANGE_MAX
            )

            adx_rising = curr_adx > prev_adx

            if adx_exited_low_zone and adx_rising:
                adx_low_condition = True
                signals['ADX']['triggered'] = True
                signals['ADX']['bias'] = di_bias
                signals['ADX']['direction'] = di_direction_id
                signals['ADX']['messages'].append(
                    "Condition remplie : ADX sort d'une zone basse et augmente "
                    "-> possibilité de début de tendance."
                )

            # Condition complémentaire : ADX sort d'une zone haute
            adx_exited_high_zone = (
                prev_adx > MARKET_STATE__ADX_VERY_STRONG_MIN and
                curr_adx <= MARKET_STATE__ADX_VERY_STRONG_MIN
            )

            adx_falling = curr_adx < prev_adx

            if adx_exited_high_zone and adx_falling:
                adx_high_condition = True
                signals['ADX']['triggered'] = True
                signals['ADX']['bias'] = "à confirmer"
                signals['ADX']['direction'] = MARKET_STATE__DIRECTION_NEUTRAL_ID
                signals['ADX']['messages'].append(
                    "Condition remplie : ADX sort d'une zone haute et diminue "
                    "-> possible essoufflement ou fin de tendance excessive."
                )

            if not adx_low_condition and not adx_high_condition:
                signals['ADX']['messages'].append(
                    "Condition non remplie : ADX ne sort pas d'une zone basse "
                    "et ne sort pas non plus d'une zone haute dans les conditions attendues."
                )

    # ------------------------------------------------------
    # Conditions KAMA
    # ------------------------------------------------------
    required = ['Close', 'KAMA']
    missing = [col for col in required if col not in df.columns]

    if missing:
        signals['KAMA']['messages'].append(f"Colonnes manquantes : {missing}")
    else:
        data = df[required].dropna()

        if len(data) < 2:
            signals['KAMA']['messages'].append(
                "Pas assez de bougies valides pour évaluer les conditions KAMA."
            )
        else:
            prev = data.iloc[-2]
            curr = data.iloc[-1]
            date = data.index[-1]

            signals['KAMA']['date'] = date

            prev_close = float(prev['Close'])
            curr_close = float(curr['Close'])

            prev_kama = float(prev['KAMA'])
            curr_kama = float(curr['KAMA'])

            kama_slope = curr_kama - prev_kama

            if curr_close != 0:
                kama_slope_pct = (kama_slope / curr_close) * 100.0
            else:
                kama_slope_pct = 0.0

            signals['KAMA']['messages'].append(
                f"Close précédent : {_format_value(prev_close)} | "
                f"Close actuel : {_format_value(curr_close)}"
            )

            signals['KAMA']['messages'].append(
                f"KAMA précédent : {_format_value(prev_kama)} | "
                f"KAMA actuel : {_format_value(curr_kama)} | "
                f"Pente : {_format_value(kama_slope_pct)}%"
            )

            crossed_above = (prev_close <= prev_kama) and (curr_close > curr_kama)
            crossed_below = (prev_close >= prev_kama) and (curr_close < curr_kama)

            kama_rising = curr_kama > prev_kama
            kama_falling = curr_kama < prev_kama

            if crossed_above and kama_rising:
                signals['KAMA']['triggered'] = True
                signals['KAMA']['bias'] = "haussier"
                signals['KAMA']['direction'] = MARKET_STATE__DIRECTION_BULLISH_ID
                signals['KAMA']['messages'].append(
                    "Condition remplie : le prix vient de passer au-dessus d'un KAMA croissant "
                    "-> biais potentiellement haussier."
                )
            elif crossed_below and kama_falling:
                signals['KAMA']['triggered'] = True
                signals['KAMA']['bias'] = "baissier"
                signals['KAMA']['direction'] = MARKET_STATE__DIRECTION_BEARISH_ID
                signals['KAMA']['messages'].append(
                    "Condition remplie : le prix vient de passer en-dessous d'un KAMA décroissant "
                    "-> biais potentiellement baissier."
                )
            else:
                signals['KAMA']['messages'].append(
                    "Condition non remplie : pas de franchissement du KAMA "
                    "dans le sens de la pente sur les dernières bougies."
                )

    # ------------------------------------------------------
    # Conditions Choppiness Index
    # ------------------------------------------------------
    required = ['Choppiness']
    missing = [col for col in required if col not in df.columns]

    if missing:
        signals['CHOPPINESS']['messages'].append(f"Colonnes manquantes : {missing}")
    else:
        data = df['Choppiness'].dropna()

        if len(data) < 2:
            signals['CHOPPINESS']['messages'].append(
                "Pas assez de bougies valides pour évaluer les conditions Choppiness."
            )
        else:
            prev_chop = float(data.iloc[-2])
            curr_chop = float(data.iloc[-1])
            date = data.index[-1]

            signals['CHOPPINESS']['date'] = date

            change = curr_chop - prev_chop

            signals['CHOPPINESS']['messages'].append(
                f"Choppiness précédent : {_format_value(prev_chop)} | "
                f"Choppiness actuel : {_format_value(curr_chop)} | "
                f"Variation : {_format_value(change)}"
            )

            strong_drop = change <= -MARKET_STATE__CHOPPINESS_STRONG_DROP_POINTS

            exited_high_range_zone = (
                prev_chop >= MARKET_STATE__CHOPPINESS_RANGE_MIN and
                curr_chop < MARKET_STATE__CHOPPINESS_RANGE_MIN
            )

            entered_trend_zone = (
                prev_chop >= MARKET_STATE__CHOPPINESS_TREND_MAX and
                curr_chop < MARKET_STATE__CHOPPINESS_TREND_MAX
            )

            if strong_drop:
                signals['CHOPPINESS']['triggered'] = True
                signals['CHOPPINESS']['messages'].append(
                    "Condition remplie : le Choppiness baisse fortement "
                    "-> sortie possible de range."
                )

            if exited_high_range_zone:
                signals['CHOPPINESS']['triggered'] = True
                signals['CHOPPINESS']['messages'].append(
                    "Condition remplie : le Choppiness sort de la zone haute "
                    "-> réduction probable du régime range."
                )

            if entered_trend_zone:
                signals['CHOPPINESS']['triggered'] = True
                signals['CHOPPINESS']['messages'].append(
                    "Condition remplie : le Choppiness passe sous le seuil de tendance "
                    "-> marché potentiellement plus directionnel."
                )

            if signals['CHOPPINESS']['triggered']:
                signals['CHOPPINESS']['bias'] = "à confirmer"
                signals['CHOPPINESS']['direction'] = MARKET_STATE__DIRECTION_NEUTRAL_ID
                signals['CHOPPINESS']['messages'].append(
                    "Direction à confirmer avec ADX, KAMA ou Directional ATR Ratio."
                )
            else:
                signals['CHOPPINESS']['messages'].append(
                    "Condition non remplie : pas de baisse significative du Choppiness."
                )

    # ------------------------------------------------------
    # Conditions Directional ATR Ratio
    # ------------------------------------------------------
    required = ['Directional_ATR_Ratio']
    missing = [col for col in required if col not in df.columns]

    if missing:
        signals['DIRECTIONAL_ATR_RATIO']['messages'].append(f"Colonnes manquantes : {missing}")
    else:
        data = df['Directional_ATR_Ratio'].dropna()

        if len(data) < 2:
            signals['DIRECTIONAL_ATR_RATIO']['messages'].append(
                "Pas assez de bougies valides pour évaluer les conditions Directional ATR Ratio."
            )
        else:
            prev_ratio = float(data.iloc[-2])
            curr_ratio = float(data.iloc[-1])
            date = data.index[-1]

            signals['DIRECTIONAL_ATR_RATIO']['date'] = date

            delta = curr_ratio - prev_ratio

            signals['DIRECTIONAL_ATR_RATIO']['messages'].append(
                f"Ratio précédent : {_format_value(prev_ratio, 4)} | "
                f"Ratio actuel : {_format_value(curr_ratio, 4)} | "
                f"Variation : {_format_value(delta, 4)}"
            )

            turned_strong_positive = (
                prev_ratio <= MARKET_STATE__DIR_ATR_STRONG_MIN and
                curr_ratio >= MARKET_STATE__DIR_ATR_STRONG_MIN
            )

            turned_strong_negative = (
                prev_ratio >= -MARKET_STATE__DIR_ATR_STRONG_MIN and
                curr_ratio <= -MARKET_STATE__DIR_ATR_STRONG_MIN
            )

            strong_positive_improvement = (
                curr_ratio >= MARKET_STATE__DIR_ATR_STRONG_MIN and
                delta >= MARKET_STATE__DIR_ATR_IMPROVEMENT_THRESHOLD
            )

            strong_negative_deterioration = (
                curr_ratio <= -MARKET_STATE__DIR_ATR_STRONG_MIN and
                delta <= -MARKET_STATE__DIR_ATR_IMPROVEMENT_THRESHOLD
            )

            if turned_strong_positive or strong_positive_improvement:
                signals['DIRECTIONAL_ATR_RATIO']['triggered'] = True
                signals['DIRECTIONAL_ATR_RATIO']['bias'] = "haussier"
                signals['DIRECTIONAL_ATR_RATIO']['direction'] = MARKET_STATE__DIRECTION_BULLISH_ID
                signals['DIRECTIONAL_ATR_RATIO']['messages'].append(
                    "Condition remplie : le Directional ATR Ratio devient fortement positif "
                    "-> momentum haussier récent."
                )
            elif turned_strong_negative or strong_negative_deterioration:
                signals['DIRECTIONAL_ATR_RATIO']['triggered'] = True
                signals['DIRECTIONAL_ATR_RATIO']['bias'] = "baissier"
                signals['DIRECTIONAL_ATR_RATIO']['direction'] = MARKET_STATE__DIRECTION_BEARISH_ID
                signals['DIRECTIONAL_ATR_RATIO']['messages'].append(
                    "Condition remplie : le Directional ATR Ratio devient fortement négatif "
                    "-> momentum baissier récent."
                )
            else:
                signals['DIRECTIONAL_ATR_RATIO']['messages'].append(
                    "Condition non remplie : pas de bascule forte du Directional ATR Ratio."
                )

    # ------------------------------------------------------
    # Normalisation finale de la direction numérique
    # ------------------------------------------------------
    # On garantit que chaque signal possède bien une direction
    # cohérente avec son biais textuel.
    # ------------------------------------------------------
    for sig in signals.values():
        sig['direction'] = _bias_to_direction_id(sig.get('bias', 'neutre'))

    return signals


def compute_trading_conditions(df: pd.DataFrame, verbose):
    """
    Affiche un print par indicateur avec les conditions de trading détectées.

    Retourne également, pour chaque indicateur, une valeur 'direction' :
    - -1 : baisse
    -  0 : range / neutre / à confirmer
    -  1 : hausse
    """
    signals = evaluate_trading_conditions(df=df)

    if verbose:
        print("\n" + "=" * 80)
        print("CONDITIONS DE TRADING BASÉES SUR LES DERNIÈRES BOUGIES")
        print("=" * 80)

    for key, sig in signals.items():
        direction = sig.get('direction', MARKET_STATE__DIRECTION_NEUTRAL_ID)

        if verbose:
            print(f"\n[Trading Conditions - {sig['label']}]")

        if sig['date'] is not None:
            if verbose:
                print(f"Date de la dernière bougie utilisée : {sig['date']}")
        else:
            if verbose:
                print("Date de la dernière bougie utilisée : N/A")

        if verbose:
            print(f"Direction normalisée : {direction} (-1 baisse, 0 range/neutre, 1 hausse).")

        for msg in sig['messages']:
            if verbose:
                print(f" - {msg}")

        if sig['triggered']:
            if verbose:
                print(f" => SIGNAL POTENTIEL : biais {sig['bias']} | direction={direction}.")
        else:
            if verbose:
                print(f" => Pas de signal de trading détecté | direction={direction}.")

    return signals


def setup_argparse() -> argparse.ArgumentParser:
    """
    Configures and returns the ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol to analyze.')
    parser.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='')
    return parser


def backtesting(df, verbose=False):
    """
    Petite boucle de backtesting / exploration.

    Cette fonction ne renvoie rien pour le moment, mais elle exploite
    désormais la valeur 'direction' retournée par compute_trading_conditions.
    """
    close_col = "Close"

    if close_col not in df.columns:
        return

    for back_bar in range(1, 10):
        if len(df) <= back_bar:
            continue

        df_clipped = df.iloc[:-back_bar].copy()

        if df_clipped.empty:
            continue

        df_clipped = compute_indicators(df=df_clipped, verbose=verbose)
        signals = compute_trading_conditions(df=df_clipped, verbose=False)

        close_value        = df_clipped[close_col].iloc[-1]
        future_close_value = df[close_col].iloc[-back_bar+1].values[0]

        directions = {
            key: sig.get('direction', MARKET_STATE__DIRECTION_NEUTRAL_ID)
            for key, sig in signals.items()
        }

        # Variables conservées pour utilisation ultérieure.
        _ = (close_value, future_close_value, directions)


def entry(args):
    for clip_n in range(0, 6):
        df = factory_load_data(
            _dataset_id=args.dataset_id,
            _ticker=args.ticker,
            _args={"clip_n": clip_n, "incomplete_at_start": True, "realtime": True}  # la première chandelle (la plus vieille) contiendra le reste (incomplète)
        ).copy()

        # backtesting(df=df, verbose=args.verbose)
        df_result = compute_indicators(df=df, verbose=args.verbose)

        # Print général expliquant comment interpréter les indicateurs
        if args.verbose: print_indicator_expectations()

        # Affichage des dernières valeurs numériques
        if args.verbose: print("\nDernières lignes avec les indicateurs :")

        columns_to_display = [
            'Close',
            'ADX',
            'Plus_DI',
            'Minus_DI',
            'KAMA',
            'Choppiness',
            'Directional_ATR_Ratio'
        ]

        # Afficher uniquement les colonnes réellement présentes
        columns_to_display = [
            col for col in columns_to_display if col in df_result.columns
        ]

        if columns_to_display:
            if args.verbose: print(df_result[columns_to_display].tail())
        else:
            if args.verbose: print("Aucune colonne d'indicateur à afficher.")

        # Diagnostic détaillé, un print par indicateur
        if args.verbose: print("\nDiagnostic détaillé indicateur par indicateur :")

        adx = compute_adx_diagnosis(df=df_result, verbose=args.verbose)
        #kama = compute_kama_diagnosis(df=df_result, verbose=args.verbose)
        #choppiness = compute_choppiness_diagnosis(df=df_result, verbose=args.verbose)
        directional_atr_ratio = compute_directional_atr_ratio_diagnosis(df=df_result, verbose=args.verbose)
        input("Press Enter")
    sys.exit(0)
    # Fusion propre des diagnostics
    result_tendance = {}
    for part in (adx, kama, choppiness, directional_atr_ratio):
        if part: result_tendance.update(part)

    # Conditions de trading basées sur les dernières bougies
    signals = compute_trading_conditions(df=df_result, verbose=args.verbose)

    vert = "\033[32m"
    rouge_clair = "\033[91m"
    gris_fonce = "\033[38;5;239m"
    jaune = "\033[33m"
    gris = "\033[90m"
    gris_moyen = "\033[38;5;244m"
    reset = "\033[0m"
    carre = "█"
    vert_fond = "\033[42m"
    rouge_clair_fond = "\033[101m"
    gris_fonce_fond = "\033[48;5;239m"
    jaune_fond = "\033[43m"
    bleu_fond = "\033[44m"  # Fond bleu standard
    bleu_clair_fond = "\033[104m"  # Fond bleu clair / brillant
    bleu = "\033[34m"
    aucune_couleur = ""
    blanc = "\033[97m"  # Texte blanc brillant
    blanc_fond = "\033[107m"  # Fond blanc brillant
    str_res = f"{args.dataset_id}("
    for k, v in result_tendance.items():
        regime = v.get("regime", MARKET_STATE__REGIME_RANGE_ID)
        direction = v.get("direction", MARKET_STATE__DIRECTION_NEUTRAL_ID)

        if direction   == MARKET_STATE__DIRECTION_BEARISH_ID: dir_col = rouge_clair_fond
        elif direction == MARKET_STATE__DIRECTION_BULLISH_ID: dir_col = vert_fond
        elif direction == MARKET_STATE__DIRECTION_NEUTRAL_ID: dir_col = bleu_clair_fond
        else: assert False

        if   regime == MARKET_STATE__REGIME_RANGE_ID: str_res      = f"{str_res}{dir_col}-{reset}"
        elif regime == MARKET_STATE__REGIME_TRANSITION_ID: str_res = f"{str_res}{dir_col}x{reset}"
        elif regime == MARKET_STATE__REGIME_TREND_ID: str_res      = f"{str_res}{dir_col}+{reset}"
        else: assert False
    str_res += ")"
    print(str_res)

    return result_tendance


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args=args)