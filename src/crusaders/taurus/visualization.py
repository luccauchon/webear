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
from datetime import datetime, timedelta
import calendar
import os
import math


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


# ---------- Date Math Helpers (Trading System Logic) ----------
def add_trading_days(start_dt, num_days):
    """Adds exactly num_days of trading days (Mon-Fri), skipping weekends."""
    current = start_dt
    added = 0
    while added < num_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 0=Mon, 4=Fri
            added += 1
    return current


def add_months_and_adjust(start_dt, num_months):
    """Adds calendar months and rolls forward to Monday if it lands on a weekend."""
    month = start_dt.month - 1 + num_months
    year = start_dt.year + month // 12
    month = month % 12 + 1
    day = min(start_dt.day, calendar.monthrange(year, month)[1])
    dt = start_dt.replace(year=year, month=month, day=day)

    if dt.weekday() == 5:  # Saturday
        dt += timedelta(days=2)
    elif dt.weekday() == 6:  # Sunday
        dt += timedelta(days=1)
    return dt


def get_lookahead_date(now_str, dataset_id, la):
    """Convert an integer lookahead to a formatted date string YYYYMMDD, skipping weekends."""
    now_str_clean = str(now_str).replace("-", "_").replace("/", "_")
    try:
        dt = datetime.strptime(now_str_clean, "%Y_%m_%d")
    except ValueError:
        try:
            dt = datetime.strptime(now_str_clean, "%Y%m%d")
        except ValueError:
            return str(la)  # Fallback to integer string if parsing fails

    if dataset_id == "day":
        dt = add_trading_days(dt, la)
    elif dataset_id == "week":
        # 1 trading week = 5 trading days
        dt = add_trading_days(dt, la * 5)
    elif dataset_id == "month":
        dt = add_months_and_adjust(dt, la)
    elif dataset_id in ["quaterly", "quarterly"]:
        dt = add_months_and_adjust(dt, la * 3)
    elif dataset_id == "year":
        try:
            dt = dt.replace(year=dt.year + la)
        except ValueError:  # Handle leap year (e.g., Feb 29 to non-leap year)
            dt = dt.replace(year=dt.year + la, day=28)
    else:
        dt = add_trading_days(dt, la)  # Fallback to trading days

    # Final weekend adjustment just in case la=0 lands on a weekend
    if dt.weekday() == 5:
        dt += timedelta(days=2)
    elif dt.weekday() == 6:
        dt += timedelta(days=1)

    return dt.strftime("%Y%m%d")


def break_even_credit(win_rate, spread_width):
    # 2. Core Math
    loss_rate = 1.0 - win_rate / 100.

    # The Break-Even Credit formula: Credit = Loss_Rate * Spread_Width
    breakeven_credit = loss_rate * spread_width
    return breakeven_credit


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
    indicator = entry.get("indicator", "?")
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
        'indicator': indicator,
    }


def parse_data(raw):
    data = strip_ws(raw)
    now, dataset_id, current_price = data["now"], data["dataset_id"], data["current_price"]
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

    df = pd.DataFrame(records)
    return df, sorted(list(all_thresholds)), sorted(list(all_lookaheads)), now, dataset_id, current_price


# ---------- Plotting & Export ----------
def plot_and_export(df, all_thresholds, all_lookaheads, now, current_price, dataset_id, output_dir, file_id):
    put_df = df[df['optimize'] == 'buy_wr']
    call_df = df[df['optimize'] == 'sell_wr']

    # Precompute date strings for all integer lookaheads to format the Y-Axis
    la_to_date = {la: get_lookahead_date(now, dataset_id, la) for la in all_lookaheads}

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

    # Variables for JS filtering
    val_wr_put, val_wr_call = [], []
    indicator_put, indicator_call = [], []
    text_put_list, text_call_list = [], []

    for subset, thresholds, col_idx in specs_list:
        # Replace numeric lookaheads with mapped YYYYMMDD string dates
        y_vals = [la_to_date[la] for la in all_lookaheads]
        x_vals = thresholds

        # Initialize grid with NaN (which Plotly renders as white/transparent gaps)
        z = np.full((len(y_vals), len(x_vals)), np.nan)
        text = np.empty((len(y_vals), len(x_vals)), dtype=object)
        text[:] = ''  # Default empty string for gaps

        z_val_wr = np.full((len(y_vals), len(x_vals)), np.nan)
        z_indicator = np.full((len(y_vals), len(x_vals)), '', dtype=object)

        if not subset.empty:
            best = subset.loc[subset.groupby(['lookahead', 'threshold'])['val_win_rate'].idxmax()]
            for _, row in best.iterrows():
                la = row['lookahead']
                th = row['threshold']
                if la not in all_lookaheads: continue

                y_idx = all_lookaheads.index(la)
                if th not in x_vals: continue
                x_idx = x_vals.index(th)

                z[y_idx, x_idx] = row['val_win_rate']
                z_val_wr[y_idx, x_idx] = row['val_win_rate']
                z_indicator[y_idx, x_idx] = row['indicator']

                # Build the rich tooltip text with the updated date string
                date_str = la_to_date[la]

                # Calculate rounded target price based on Put or Call section
                price = row['target_price']
                if col_idx == 1:  # Put section: round down to nearest 5
                    rounded_price = math.floor(price / 5) * 5
                else:  # Call section: round up to nearest 5
                    rounded_price = math.ceil(price / 5) * 5
                minimum_break_even_credit = break_even_credit(win_rate=row['val_win_rate'], spread_width=500)
                text[y_idx, x_idx] = (
                    f"<b>Lookahead:</b> {date_str} ({la} {dataset_id})<br>"
                    f"<b>Threshold:</b> {th:.3f} ({(th - 1) * 100:+.1f}%)<br>"
                    f"<b>Val Win Rate:</b> {row['val_win_rate']:.2f}%<br>"
                    f"<b>Target Rounded Price:</b> {rounded_price:.2f}<br>"
                    f"<b>Target Date:</b> {row['target_date']}<br>"
                    f"<b>Break-Even (500$ Max Loss):</b> {minimum_break_even_credit:.0f}$<br>"
                    f"<b>Indicator:</b> {row['indicator']}<br>"
                )

        # Store the original matrices for the slider logic
        if col_idx == 1:
            z_put, text_put = z.copy(), text.copy()
            val_wr_put = z_val_wr.tolist()
            val_wr_put = [[None if pd.isna(v) else v for v in row] for row in val_wr_put]
            indicator_put = z_indicator.tolist()
            text_put_list = text.tolist()
        else:
            z_call, text_call = z.copy(), text.copy()
            val_wr_call = z_val_wr.tolist()
            val_wr_call = [[None if pd.isna(v) else v for v in row] for row in val_wr_call]
            indicator_call = z_indicator.tolist()
            text_call_list = text.tolist()

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
    # Create steps from 33% to 99% in increments of 1%
    for T in range(33, 100, 1):
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
        title_text=f"<b>[{now}:{current_price:.0f}]Best Validation Win Rate per Lookahead × Price Level</b><br><sup>(Hover over colored boxes for details. White boxes = No signal)</sup>",
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

    # tickmode='linear' is removed because we are now passing string arrays to the Y axis.
    fig.update_yaxes(title_text=f"Lookahead Date ({dataset_id} ahead)", autorange="reversed")
    fig.update_xaxes(title_text="Threshold (target / current price)")

    # --- STANDALONE HTML EXPORT ---
    out_html = Path(os.path.join(output_dir, f'{file_id}_{now}_{dataset_id}.html')).resolve()

    # Generate HTML string
    html_str = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={'displayModeBar': True, 'modeBarButtonsToAdd': ['resetScale2d']}
    )

    # Custom HTML and JS for checkboxes
    custom_html = f"""
<div id="indicator-checkboxes" style="text-align: center; margin: 20px 0; font-family: sans-serif; font-size: 16px; color: #333;">
    <label style="margin-right: 20px; cursor: pointer; user-select: none;">
        <input type="checkbox" id="chkPrimeRSI" checked style="margin-right: 5px; transform: scale(1.2); cursor: pointer; vertical-align: middle;"> 
        <span style="vertical-align: middle;">Prime RSI</span>
    </label>
    <label style="cursor: pointer; user-select: none;">
        <input type="checkbox" id="chkDGDR" checked style="margin-right: 5px; transform: scale(1.2); cursor: pointer; vertical-align: middle;"> 
        <span style="vertical-align: middle;">DGDR</span>
    </label>
</div>
<script>
const valWrPut = {json.dumps(val_wr_put)};
const valWrCall = {json.dumps(val_wr_call)};
const indicatorPut = {json.dumps(indicator_put)};
const indicatorCall = {json.dumps(indicator_call)};
const origTextPut = {json.dumps(text_put_list)};
const origTextCall = {json.dumps(text_call_list)};

function applyFilters() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) return;

    // Get current slider threshold
    const activeIdx = plotDiv.layout.sliders[0].active;
    const currentThreshold = activeIdx + 33;

    const showPrimeRSI = document.getElementById('chkPrimeRSI').checked;
    const showDGDR = document.getElementById('chkDGDR').checked;

    let newZPut = valWrPut.map((row, i) => 
        row.map((val, j) => {{
            if (val === null || val < currentThreshold) return null;
            let ind = indicatorPut[i][j];
            if (ind === "Prime RSI" && !showPrimeRSI) return null;
            if (ind === "DGDR" && !showDGDR) return null;
            return val;
        }})
    );

    let newTextPut = origTextPut.map((row, i) => 
        row.map((txt, j) => {{
            if (newZPut[i][j] === null) return '';
            return txt;
        }})
    );

    let newZCall = valWrCall.map((row, i) => 
        row.map((val, j) => {{
            if (val === null || val < currentThreshold) return null;
            let ind = indicatorCall[i][j];
            if (ind === "Prime RSI" && !showPrimeRSI) return null;
            if (ind === "DGDR" && !showDGDR) return null;
            return val;
        }})
    );

    let newTextCall = origTextCall.map((row, i) => 
        row.map((txt, j) => {{
            if (newZCall[i][j] === null) return '';
            return txt;
        }})
    );

    Plotly.restyle(plotDiv, {{'z': [newZPut, newZCall], 'text': [newTextPut, newTextCall]}});
}}

document.addEventListener('DOMContentLoaded', (event) => {{
    document.getElementById('chkPrimeRSI').addEventListener('change', applyFilters);
    document.getElementById('chkDGDR').addEventListener('change', applyFilters);

    const plotDiv = document.querySelector('.plotly-graph-div');
    if (plotDiv) {{
        plotDiv.on('plotly_sliderchange', function(eventData) {{
            setTimeout(applyFilters, 50);
        }});
        plotDiv.on('plotly_restyle', function(eventData) {{
            setTimeout(applyFilters, 50);
        }});
    }}
}});
</script>
"""

    # Insert custom HTML before closing body tag
    final_html = html_str.replace('</body>', custom_html + '\n</body>')

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(final_html)

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
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help=f"Directory where to save the html file"
    )
    parser.add_argument(
        "--file_id", type=str, default="",
        help=f"Id be to included in the html filename"
    )
    return parser.parse_args()


def entry(args):
    if not os.path.exists(args.filepath):
        print(f"❌ File not found: {args.filepath}");
        return

    raw = load_data(args.filepath)
    print(f"Total top-level keys: {len(raw)}\n")

    df, all_thresholds, all_lookaheads, now, dataset_id, current_price = parse_data(raw)
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

    plot_and_export(df=df, all_thresholds=all_thresholds, all_lookaheads=all_lookaheads, now=now, current_price=current_price, dataset_id=dataset_id, output_dir=args.output_dir, file_id=args.file_id)


if __name__ == '__main__':
    args = parse_args()
    entry(args)