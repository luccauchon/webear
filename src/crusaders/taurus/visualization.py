"""
Credit Spread Probability Visualizer (Interactive Standalone HTML)
------------------------------------------------------------------
Parses luc.json and displays the best validation win rate for each
(lookahead, threshold) combination, split into:
  - Put Credit Spread side  (buy_wr, signal == 1)
  - Call Credit Spread side (sell_wr, signal == -1)

Features:
  - White squares indicate no data / system did not emit probability.
  - Hover over any colored box to see a detailed tooltip.
  - Automatically generates and opens a 100% standalone HTML file.
"""
import argparse
import json
import re
import webbrowser
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import os


# ---------- IO helpers ----------
def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def strip_ws(obj):
    """Recursively strip whitespace from dict keys and string values."""
    if isinstance(obj, dict):
        return {k.strip(): strip_ws(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_ws(i) for i in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def get_thresh_from_key(s):
    """Extract the numeric threshold from keys like '0.986::' or '::1.014'."""
    m = re.search(r'([\d.]+)', s)
    return float(m.group(1)) if m else None


# ---------- Parsing ----------
def parse_entry(entry, lookahead, optimize, thresh):
    signal = entry.get('signal', 0)
    if optimize == 'buy_wr' and signal != 1.0: return None
    if optimize == 'sell_wr' and signal != -1.0: return None

    try:
        val_wr = float(str(entry.get('val_win_rate_str', '0')).replace('%', ''))
        train_wr = float(str(entry.get('train_win_rate_str', '0')).replace('%', ''))
        tgt = float(entry.get('target_price_str', 0))
        cur = float(entry.get('current_price_str', 0))
    except ValueError:
        return None

    return {
        'lookahead': lookahead,
        'optimize': optimize,
        'threshold': thresh,
        'val_win_rate': val_wr,
        'train_win_rate': train_wr,
        'target_price': tgt,
        'current_price': cur,
        'pct_move': (tgt / cur - 1) * 100 if cur else 0,
        'target_date': entry.get('target_date_str', ''),
    }


def parse_data(raw):
    data = strip_ws(raw)
    records = []
    all_thresholds = set()
    all_lookaheads = set()

    # Layout A: lookahead -> optimize -> thresh -> [entries]
    for la_str, opt_dict in data.items():
        if not la_str.isdigit():
            continue
        la = int(la_str)
        all_lookaheads.add(la)

        if not isinstance(opt_dict, dict):
            continue

        for opt, th_dict in opt_dict.items():
            if not isinstance(th_dict, dict):
                continue

            for th_str, entries in th_dict.items():
                th = get_thresh_from_key(th_str)
                if th is None:
                    continue
                all_thresholds.add(th)

                if not isinstance(entries, list):
                    continue

                for e in entries:
                    r = parse_entry(e, la, opt, th)
                    if r:
                        records.append(r)

    # Fallback to Layout B if Layout A found nothing
    if not records:
        day_re = re.compile(r'::day\s*::(\d+)')
        for th_str, entries in data.items():
            if not isinstance(entries, list): continue
            th = get_thresh_from_key(th_str)
            if th is None: continue
            all_thresholds.add(th)
            for e in entries:
                m = day_re.search(e.get('info', ''))
                if not m: continue
                la = int(m.group(1))
                all_lookaheads.add(la)
                opt = e.get('optimize', '')
                r = parse_entry(e, la, opt, th)
                if r: records.append(r)

    df = pd.DataFrame(records)
    return df, sorted(list(all_thresholds)), sorted(list(all_lookaheads))


# ---------- Plotting & Export ----------
def plot_and_export(df, all_thresholds, all_lookaheads):
    put_df = df[df['optimize'] == 'buy_wr']
    call_df = df[df['optimize'] == 'sell_wr']

    # Separate thresholds logically: Put (<1.0) and Call (>1.0)
    put_thresholds = sorted([t for t in all_thresholds if t < 1.0])
    call_thresholds = sorted([t for t in all_thresholds if t > 1.0])

    if not put_thresholds: put_thresholds = all_thresholds
    if not call_thresholds: call_thresholds = all_thresholds

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Put Credit Spread</b> <br><sup>Probability price stays ABOVE strike</sup>',
            '<b>Call Credit Spread</b> <br><sup>Probability price stays BELOW strike</sup>'
        ),
        horizontal_spacing=0.12
    )

    specs_list = [
        (put_df, put_thresholds, 1),
        (call_df, call_thresholds, 2)
    ]

    # Variables to store the original unfiltered matrices for the slider
    z_put, text_put = None, None
    z_call, text_call = None, None

    for subset, thresholds, col_idx in specs_list:
        y_vals = all_lookaheads
        x_vals = thresholds

        # Initialize grid with NaN (which Plotly renders as white/transparent gaps)
        z = np.full((len(y_vals), len(x_vals)), np.nan)
        text = np.empty((len(y_vals), len(x_vals)), dtype=object)
        text[:] = ''  # Default empty string for gaps

        if not subset.empty:
            best = subset.loc[subset.groupby(['lookahead', 'threshold'])['val_win_rate'].idxmax()]
            for _, row in best.iterrows():
                la = row['lookahead']
                th = row['threshold']
                y_idx = y_vals.index(la)
                x_idx = x_vals.index(th)

                z[y_idx, x_idx] = row['val_win_rate']
                # Build the rich tooltip text
                text[y_idx, x_idx] = (
                    f"<b>Lookahead:</b> {la} days<br>"
                    f"<b>Threshold:</b> {th:.3f} ({(th - 1) * 100:+.1f}%)<br>"
                    f"<b>Val Win Rate:</b> {row['val_win_rate']:.2f}%<br>"
                    f"<b>Target Price:</b> {row['target_price']:.2f}<br>"
                    f"<b>Target Date:</b> {row['target_date']}"
                )

        # Store the original matrices for the slider logic
        if col_idx == 1:
            z_put, text_put = z.copy(), text.copy()
        else:
            z_call, text_call = z.copy(), text.copy()

        # Format x-axis labels to show both the multiplier and the % distance
        x_labels = [f"{(t - 1) * 100:+.1f}%" for t in x_vals]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_vals,
                text=text,
                hovertemplate="%{text}<extra></extra>",  # <extra></extra> hides the trace name
                hoverongaps=False,  # Disables hover on white/NaN cells
                colorscale='YlOrRd',
                zmin=50, zmax=100,
                xgap=2, ygap=2,  # Creates a crisp grid line effect
                showscale=(col_idx == 1),  # Only show one colorbar for both charts
                colorbar=dict(title="Val Win Rate (%)", len=0.8, y=0.5)
            ),
            row=1, col=col_idx
        )

    # --- Create Slider for filtering by Val Win Rate ---
    steps = []
    # Create steps from 50% to 99% in increments of 1%
    for T in range(50, 100, 1):
        # For Put side: keep values >= T, set others to NaN (white gap)
        z_put_T = np.where(z_put >= T, z_put, np.nan)
        text_put_T = np.where(z_put >= T, text_put, '')

        # For Call side: keep values >= T, set others to NaN (white gap)
        z_call_T = np.where(z_call >= T, z_call, np.nan)
        text_call_T = np.where(z_call >= T, text_call, '')

        step = dict(
            method='update',
            args=[
                {'z': [z_put_T.tolist(), z_call_T.tolist()],
                 'text': [text_put_T.tolist(), text_call_T.tolist()]}
            ],
            label=f"{T}%"
        )
        steps.append(step)

    fig.update_layout(
        title_text="<b>Best Validation Win Rate per Lookahead × Price Level</b><br><sup>(Hover over colored boxes for details. White boxes = No signal and/or no data)</sup>",
        title_x=0.5,
        height=750,
        width=1400,
        template="plotly_white",
        hovermode="closest",
        hoverlabel_font_size=16,
        margin=dict(b=120, t=100),  # Increased bottom margin to make room for the slider
        sliders=[dict(
            active=0,  # Start at 50% (shows everything)
            currentvalue={"prefix": "Min Val Win Rate: ", "font": {"size": 16}},
            pad={"t": 30},
            len=0.9,
            x=0.05,
            y=-0.15,
            xanchor="left",
            yanchor="top",
            steps=steps,
            bgcolor="lightgray",
            activebgcolor="darkblue",
            tickcolor="black",
            font=dict(color="black", size=12)
        )]
    )

    fig.update_yaxes(title_text="Lookahead (days ahead)", autorange="reversed", tickmode='linear')
    fig.update_xaxes(title_text="Threshold (target / current price)")

    # --- STANDALONE HTML EXPORT ---
    current_date = datetime.now()

    # Format the date as YYYY_MM_DD
    date_string = current_date.strftime("%Y_%m_%d")

    out_html = Path(f'credit_spread_interactive_{date_string}.html').resolve()

    # include_plotlyjs=True bakes the entire JS library into the file.
    # This guarantees it works 100% offline as a single standalone file.
    fig.write_html(
        out_html,
        include_plotlyjs=True,
        full_html=True,
        config={'displayModeBar': True, 'modeBarButtonsToAdd': ['resetScale2d']}
    )

    print(f"\n✅ Successfully generated standalone HTML file:")
    print(f"👉 {out_html}")
    print("🌐 Opening in your default web browser...")

    # Automatically open the HTML file in the browser
    webbrowser.open_new_tab(f"file://{out_html}")


# ---------- Main ----------
def parse_args():
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Credit Spread Probability Visualizer"
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to the JSON file to parse"
    )
    return parser.parse_args()


def entry(args):
    if not os.path.exists(args.filepath):
        print(f"❌ File not found: {args.filepath}");
        return

    raw = load_data(args.filepath)
    print(f"Total top-level keys: {len(raw)}\n")

    df, all_thresholds, all_lookaheads = parse_data(raw)
    print(f"Records kept after signal filtering: {len(df)}")
    if df.empty:
        print("Nothing to plot. Check the file structure.");
        return

    print(f"Lookaheads found : {all_lookaheads}")
    print(f"Total thresholds : {len(all_thresholds)}")

    put_thresholds = sorted([t for t in all_thresholds if t < 1.0])
    call_thresholds = sorted([t for t in all_thresholds if t > 1.0])
    print(f"Put thresholds   : {len(put_thresholds)}")
    print(f"Call thresholds  : {len(call_thresholds)}\n")

    plot_and_export(df, all_thresholds, all_lookaheads)


if __name__ == '__main__':
    args = parse_args()
    entry(args)