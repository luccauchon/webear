try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import os
import argparse
import math
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import optuna
import joblib
import time
from multiprocessing import freeze_support, Lock, Process, Queue, Value
import argparse
import sys
import psutil
from argparse import Namespace
import os
import sys
import time
from runners.atr import entry as atr_entry
from utils import WEBEARStyle, send_html_email, get_and_clean_stub_dir
from constants import GET_EMAILS
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def _make_plot_matplotlib(current_price, low_levels, high_levels, display_plot=True, output_file=None):
    # Ensure data is sorted by price
    low_levels.sort(key=lambda x: x[0])
    high_levels.sort(key=lambda x: x[0])

    # --- Dynamic Y-Axis Limits ---
    # Extract the absolute lowest and highest prices from the data
    min_price = min(price for price, _, _ in low_levels)
    max_price = max(price for price, _, _ in high_levels)
    padding = 30  # Adjust this value to change the space above/below the outermost levels

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(14, 18), facecolor='#121212')
    ax.set_facecolor('#121212')

    # Plot Current Price
    ax.axhline(current_price, color='#FFD700', linewidth=3, linestyle='--')
    ax.text(0.5, current_price, f'  CURRENT: {current_price}  ', color='#FFD700', fontsize=13, fontweight='bold', va='center', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#121212", ec="#FFD700", lw=2, alpha=0.9))

    # Helper function for probability-based coloring
    def get_prob_color(prob):
        if prob >= 95: return '#00FF00'      # Bright Green
        elif prob >= 85: return '#ADFF2F'     # Green Yellow
        elif prob >= 75: return '#FFFF00'     # Yellow
        else: return '#FF4500'                # Orange Red

    # Text background style for high contrast (Dark rounded box)
    text_bbox = dict(boxstyle="round,pad=0.15", fc="#1E1E1E", ec="none", alpha=0.95)

    # Plot Low Levels (Left-aligned with cascade effect)
    for i, (price, prob, be) in enumerate(low_levels):
        color = get_prob_color(prob)
        ax.axhline(price, color=color, linewidth=1.5, linestyle='-', alpha=0.5)
        x_pos = 0.03 + (i * 0.02)
        ax.text(x_pos, price, f'{price:4d} | {prob}% | ${be:3d}', color=color, fontsize=9, fontweight='bold', va='center', fontfamily='monospace', bbox=text_bbox)

    # Plot High Levels (Right-aligned with cascade effect)
    for i, (price, prob, be) in enumerate(high_levels):
        color = get_prob_color(prob)
        ax.axhline(price, color=color, linewidth=1.5, linestyle='-', alpha=0.5)
        x_pos = 0.97 - (i * 0.02)
        ax.text(x_pos, price, f'{price:4d} | {prob}% | ${be:3d}', color=color, fontsize=9, fontweight='bold', va='center', ha='right', fontfamily='monospace', bbox=text_bbox)

    # --- Formatting & Styling ---
    # Use the dynamically calculated limits instead of hardcoding them
    ax.set_ylim(min_price - padding, max_price + padding)
    ax.set_xlim(0, 1)
    ax.set_ylabel('Price Level', color='white', fontsize=14)
    ax.tick_params(axis='y', colors='white', labelsize=12)
    ax.get_xaxis().set_visible(False)
    ax.grid(axis='y', color='#333333', linestyle=':', alpha=0.3)

    # Legend has been removed

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if display_plot:
        plt.show()


def _make_plot_plotly(current_price, low_levels, high_levels, display_plot=True, output_file=None):
    # Ensure data is sorted by price
    low_levels.sort(key=lambda x: x[0])
    high_levels.sort(key=lambda x: x[0])

    # --- Dynamic Y-Axis Limits ---
    min_price = min(price for price, _, _ in low_levels)
    max_price = max(price for price, _, _ in high_levels)
    padding = 30

    # --- Helper function for probability-based coloring ---
    def get_prob_color(prob):
        if prob >= 95:
            return '#00FF00'
        elif prob >= 85:
            return '#ADFF2F'
        elif prob >= 75:
            return '#FFFF00'
        else:
            return '#FF4500'

    # --- Plot Setup ---
    fig = go.Figure()

    # Plot Low Levels (Left-aligned with cascade effect)
    for i, (price, prob, be) in enumerate(low_levels):
        color = get_prob_color(prob)
        # Horizontal line
        fig.add_hline(
            y=price,
            line=dict(color=color, width=1.5, dash='solid'),
            opacity=0.5
        )
        # Text label with cascade
        x_pos = 0.03 + (i * 0.02)
        fig.add_annotation(
            x=x_pos,
            y=price,
            xref='paper',
            yref='y',
            text=f'{price:4d} | {prob}% | ${be:3d}',
            showarrow=False,
            font=dict(color=color, size=11, family='monospace'),
            bgcolor='#1E1E1E',
            borderpad=3,
            opacity=0.95
        )

    # Plot High Levels (Right-aligned with cascade effect)
    for i, (price, prob, be) in enumerate(high_levels):
        color = get_prob_color(prob)
        # Horizontal line
        fig.add_hline(
            y=price,
            line=dict(color=color, width=1.5, dash='solid'),
            opacity=0.5
        )
        # Text label with cascade
        x_pos = 0.97 - (i * 0.02)
        fig.add_annotation(
            x=x_pos,
            y=price,
            xref='paper',
            yref='y',
            text=f'{price:4d} | {prob}% | ${be:3d}',
            showarrow=False,
            font=dict(color=color, size=11, family='monospace'),
            bgcolor='#1E1E1E',
            borderpad=3,
            opacity=0.95,
            xanchor='right'
        )

    # Plot Current Price (golden dashed line)
    fig.add_hline(
        y=current_price,
        line=dict(color='#FFD700', width=3, dash='dash')
    )
    fig.add_annotation(
        x=0.5,
        y=current_price,
        xref='paper',
        yref='y',
        text=f'  CURRENT: {current_price}  ',
        showarrow=False,
        font=dict(color='#FFD700', size=13),
        bgcolor='#121212',
        bordercolor='#FFD700',
        borderwidth=2,
        borderpad=6,
        opacity=0.9
    )

    # --- Layout & Styling ---
    fig.update_layout(
        width=1000,
        height=900,
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        yaxis=dict(
            range=[min_price - padding, max_price + padding],
            title=dict(text='Price Level', font=dict(color='white', size=14)),
            tickfont=dict(color='white', size=12),
            gridcolor='#333333',
            gridwidth=1,
            griddash='dot',
            showgrid=True,
            zeroline=False
        ),
        xaxis=dict(
            visible=False,
            range=[0, 1]
        ),
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False
    )

    if output_file is not None:
        if output_file.endswith('.html'):
            fig.write_html(output_file)
        else:
            fig.write_image(output_file, scale=3)  # scale=3 for high DPI

    if display_plot:
        fig.show()


def make_plot(current_price, low_levels, high_levels, display_plot=True, output_file=None, use_plotly=False):
    if use_plotly:
        _make_plot_plotly(current_price=int(current_price), low_levels=low_levels, high_levels=high_levels, display_plot=display_plot, output_file=output_file)
    else:
        _make_plot_matplotlib(current_price=int(current_price), low_levels=low_levels, high_levels=high_levels, display_plot=display_plot, output_file=output_file)


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset-id", type=str, default="day")
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode. Don't send the emails.")
    parser.add_argument("--n-trials", type=int, default=999)
    parser.add_argument("--use-close-for-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--production-setup", action=argparse.BooleanOptionalAction, default=False)

    return parser


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Traitement des requêtes
    all_results_computed = []
    while True:
        use_case_batch = []
        try:
            item = use_cases__shared.get(timeout=0.1)
            use_case_batch.append(item)
        except:
            break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        for use_case in use_case_batch:
            atr_config = use_case
            result = atr_entry(args=atr_config)
            predicted_high = math.ceil((result['realtime']['predicted_high'] + 0.01) / 5) * 5
            predicted_low = math.floor((result['realtime']['predicted_low'] + 0.01) / 5) * 5
            vix_regime = result['realtime']['vix_regime']
            probability_predicted_high = result['regime_metrics'][vix_regime]['Borne Haute Respectée']
            probability_predicted_low = result['regime_metrics'][vix_regime]['Borne Basse Respectée']
            actual_high, actual_low, actual_close, actual_open = result['realtime']['actual_high'], result['realtime']['actual_low'], result['realtime']['actual_close'], result['realtime']['actual_open']
            atr_config.dataframe=None
            all_results_computed.append({'actual_high': actual_high, 'actual_low': actual_low, 'actual_close': actual_close, 'actual_open': actual_open,
                                         'predicted_high': predicted_high, 'predicted_low': predicted_low, 'vix_regime': vix_regime,
                                         'probability_predicted_high': probability_predicted_high, 'probability_predicted_low': probability_predicted_low,
                                         'atr_config': atr_config, 'dataset_configuration': result['dataset_configuration']})
    out__shared.put(all_results_computed)


def entry(args):
    nb_worker = 15
    verbose = True
    destinataires = GET_EMAILS() if args.production_setup else GET_EMAILS(dev=True)
    spread_width = 500.0

    # Obtention du dataframe en realtime (pour éviter que les workers aient à le faire)
    atr_config = Namespace(ticker=args.ticker, dataset_id=args.dataset_id, dataframe=None, verbose=False, n_trials=9, use_realtime_data=True, atr_window=14,
                           n_split=0.9, tightness_weight=0., use_close_for_range=args.use_close_for_range, clip_n=0, timeout=9999)
    result = atr_entry(args=atr_config)
    dataframe = result['dataframe_and_cols']

    # Données des workers
    data_from_workers = []

    # Construction des cas à traiter
    use_cases = []
    tightness_weights = np.linspace(0, 1, 30).tolist()
    for tightness_weight in tightness_weights + [2., 4., 8., 16.]:
        atr_config = Namespace(ticker=args.ticker, dataset_id=args.dataset_id, dataframe=dataframe, verbose=False, n_trials=args.n_trials, use_realtime_data=True, atr_window=14,
                               n_split=0.9, tightness_weight=tightness_weight, use_close_for_range=args.use_close_for_range, clip_n=0, timeout=9999)
        use_cases.append(atr_config)

    use_cases__shared, master_cmd__shared = Queue(256000), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, nb_worker)]
    # Lancement des workers
    for k in range(0, nb_worker):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
        p.start()
    # Envoie les informations aux workers pour traitement
    # Préparation des lots de travail
    for use_case in use_cases:
        use_cases__shared.put(use_case)
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1
    # Récupération des résultats
    for k in range(0, nb_worker):
        data_from_workers.extend(out__shared[k].get())
    actual_high, actual_low, actual_close, actual_open, vix_regime = next(({k: v for k, v in item.items() if k in ['actual_high', 'actual_low', 'actual_close', 'actual_open', 'vix_regime']}.values() for item in data_from_workers))
    low_levels_for_graphics, high_levels_for_graphics = [], []
    ref_range = "Plage maintenue à la clôture (close)" if args.use_close_for_range else "Plage maintenue max/min (High/Low) en cours de séance"
    subject = (f"[@{datetime.now().strftime("%Y%m%d_%H%M")}] | {ref_range} | {args.ticker}:{args.dataset_id} | VIX Regime is {vix_regime} | O:{actual_open:.0f} H:{actual_high:.0f} L:{actual_low:.0f} C:{actual_close:.0f} | [BREAK EVEN ON 5-POINT WIDE SPREAD]")
    string_generated, vlow_text, vhigh_text, dataset_configuration = subject + "\n", None, None, None
    for col_for_sort in ["predicted_low", "predicted_high"]:
        # 1. Tri du plus GRAND au plus PETIT
        liste_triee = sorted(data_from_workers, key=lambda x: x[f"probability_{col_for_sort}"], reverse=True)
        # 2. Le dictionnaire garde la DERNIÈRE valeur lue (donc la plus basse)
        dictionnaire_unique = {elem[col_for_sort]: elem for elem in liste_triee}
        # 3. Récupération du résultat final sous forme de liste
        sorted_probabilities_for_predicted_ = list(dictionnaire_unique.values())
        # Liste pour stocker les éléments filtrés
        filtered_probabilities = []
        for i in range(len(sorted_probabilities_for_predicted_)):
            current = sorted_probabilities_for_predicted_[i]

            # Si on n'est pas au premier élément, on compare avec le précédent (i - 1)
            if i > 0:
                previous = sorted_probabilities_for_predicted_[i - 1]

                same_low = current['probability_predicted_low'] == previous['probability_predicted_low']
                same_high = current['probability_predicted_high'] == previous['probability_predicted_high']

                # Si les deux probabilités sont identiques au précédent, on l'ignore
                if same_low and same_high:
                    continue

            # On garde l'élément s'il est unique par rapport au précédent
            filtered_probabilities.append(current)
        if 'low' in col_for_sort:
            string_generated += (f"\t::  {WEBEARStyle.BOLD}Low  @P    BE$   {WEBEARStyle.END}")+ "\n"
        else:
            string_generated += (f"\t::  {' ':<16}{WEBEARStyle.BOLD}High @P    BE${WEBEARStyle.END}")+ "\n"
        for sorted_probability in filtered_probabilities:
            predicted_low, predicted_high = int(sorted_probability['predicted_low']), int(sorted_probability['predicted_high'])
            probability_predicted_low, probability_predicted_high = int(sorted_probability['probability_predicted_low']), int(sorted_probability['probability_predicted_high'])
            breakeven_high = int((1.0 - probability_predicted_high / 100.) * spread_width)
            breakeven_low = int((1.0 - probability_predicted_low / 100.) * spread_width)
            low_levels_for_graphics.append((predicted_low, probability_predicted_low, breakeven_low))
            high_levels_for_graphics.append((predicted_high, probability_predicted_high, breakeven_high))
        for sorted_probability in filtered_probabilities:
            predicted_low, predicted_high = int(sorted_probability['predicted_low']), int(sorted_probability['predicted_high'])
            probability_predicted_low, probability_predicted_high = int(sorted_probability['probability_predicted_low']), int(sorted_probability['probability_predicted_high'])
            assert actual_high == sorted_probability['actual_high']
            assert actual_low == sorted_probability['actual_low']
            assert actual_close == sorted_probability['actual_close']
            assert actual_open == sorted_probability['actual_open']
            assert vix_regime == sorted_probability['vix_regime']
            breakeven_high = int((1.0 - probability_predicted_high/100.) * spread_width)
            breakeven_low = int((1.0 - probability_predicted_low/100.) * spread_width)
            if 'low' in col_for_sort:
                string_generated += (f"\t"                      
                      f"{' ':<4}{predicted_low:04d} @{probability_predicted_low:02d}% {breakeven_low:03d}$")+ "\n"
                vlow_text = (predicted_low, probability_predicted_low,breakeven_low) if vlow_text is None else vlow_text
            else:
                string_generated += (f"\t"                      
                      f"{' ':<20}{predicted_high:04d} @{probability_predicted_high:02d}% {breakeven_high:03d}$")+ "\n"
                vhigh_text = (predicted_high, probability_predicted_high, breakeven_high) if vhigh_text is None else vhigh_text
            if dataset_configuration is None:
                dataset_configuration = sorted_probability['dataset_configuration']
    string_explicative = (f"Entraînement du {dataset_configuration['train_info']['start_date']} au {dataset_configuration['train_info']['end_date']} ({dataset_configuration['train_info']['bars']} chandelles)\n"
                          f"Test du {dataset_configuration['test_info']['start_date']} au {dataset_configuration['test_info']['end_date']} ({dataset_configuration['test_info']['bars']} chandelles)")
    string_generated += ("\n\n"+string_explicative)
    string_generated += ("\n\nBonjour,\n"
                         f"Voici les zones de probabilités pour la valeur de clôture du {args.ticker}. "
                         f"Par exemple, pour la fermeture des marchés d'aujourd'hui ({datetime.now().strftime('%Y-%m-%d')}), "
                         f"le {args.ticker} a {vlow_text[1]}% de chances de terminer au-dessus de {vlow_text[0]} et {vhigh_text[1]}% de chances de clôturer sous la barre des {vhigh_text[0]}."
                         f"Pour monétiser cette analyse, vous pouvez utiliser un Put Credit Spread (pour la borne inférieure) et un Call Credit Spread (pour la borne supérieure). "
                         f"Idéalement, assurez-vous de percevoir une prime qui correspond au seuil de rentabilité (Break-Even) spécifié ci-dessus. "
                         f"Par exemple, lors de la vente d'un Put Credit Spread 0DTE de 5 points de large (vendre le PUT {vlow_text[0]} et acheter le PUT {vlow_text[0]-5}), "
                         f"la prime minimale à recevoir devrait être de {vlow_text[2]}$. "
                         f"Cela représente un risque maximal de {500-vlow_text[2]}$ si le {args.ticker} clôture en dessous de {vlow_text[0]-5}.")
    print(string_generated)
    output_file = os.path.join(get_and_clean_stub_dir("atr_make_graphics"), f"{args.ticker}_{args.dataset_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png")
    make_plot(current_price=actual_close, low_levels=low_levels_for_graphics, high_levels=high_levels_for_graphics, display_plot=False, output_file=output_file, use_plotly=False)
    if not args.debug:
        send_html_email(destinataires=destinataires, sujet=subject, corps=string_generated, pieces_jointes=output_file)
    return None


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)