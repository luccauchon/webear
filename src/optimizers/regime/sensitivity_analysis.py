import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Tuple


def parse_arguments():
    """Parse command-line arguments for trade parameters."""
    parser = argparse.ArgumentParser(
        description="Analyze trade metrics sensitivity to credit received changes."
    )

    parser.add_argument(
        '--regime-win-prob',
        type=float,
        default=0.8264,
        help='Historical win probability from regime (0.0 to 1.0). Default: 0.8264'
    )

    parser.add_argument(
        '--spread-width',
        type=float,
        default=5.00,
        help='Total width of the spread in dollars. Default: 5.00'
    )

    parser.add_argument(
        '--current-credit',
        type=float,
        default=2.00,
        help='Current credit received in dollars. Default: 2.00'
    )

    parser.add_argument(
        '--min-edge-ratio',
        type=float,
        default=0.04,
        help='Minimum edge ratio required for trade approval. Default: 0.04 (4%%)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting (useful for headless servers)'
    )

    return parser.parse_args()


def analyze_credit_sensitivity(
        regime_win_prob: float,
        spread_width: float,
        min_edge_ratio: float = 0.04,
        credit_range: Tuple[float, float] = (0.5, 4.0),
        num_points: int = 50
) -> pd.DataFrame:
    """
    Analyze trade metrics sensitivity to credit received changes.

    Parameters:
    -----------
    regime_win_prob : float
        Historical win probability from regime (0.0 to 1.0)
    spread_width : float
        Total width of the spread in dollars
    min_edge_ratio : float
        Minimum edge ratio required for trade approval
    credit_range : tuple
        (min_credit, max_credit) to analyze
    num_points : int
        Number of credit values to test

    Returns:
    --------
    pd.DataFrame with sensitivity analysis
    """

    credits = np.linspace(credit_range[0], credit_range[1], num_points)
    results = []

    for credit in credits:
        max_loss = spread_width - credit

        # Skip invalid configurations
        if max_loss <= 0:
            continue

        # Calculate metrics
        prob_loss = 1 - regime_win_prob
        expectancy = (regime_win_prob * credit) - (prob_loss * max_loss)
        edge_ratio = expectancy / max_loss if max_loss > 0 else 0

        # Break-even probability (minimum win rate needed)
        break_even_prob = max_loss / (credit + max_loss)

        # Trade decision
        trade_approved = (edge_ratio >= min_edge_ratio) and (regime_win_prob > break_even_prob)

        # Risk metrics
        risk_reward_ratio = max_loss / credit if credit > 0 else np.inf
        roi_per_trade = (expectancy / max_loss) * 100 if max_loss > 0 else 0

        results.append({
            'credit_received': credit,
            'max_loss': max_loss,
            'expectancy': expectancy,
            'edge_ratio': edge_ratio,
            'edge_ratio_pct': edge_ratio * 100,
            'break_even_prob': break_even_prob,
            'break_even_pct': break_even_prob * 100,
            'trade_approved': trade_approved,
            'risk_reward_ratio': risk_reward_ratio,
            'roi_per_trade_pct': roi_per_trade,
            'expected_win_amount': credit * regime_win_prob,
            'expected_loss_amount': max_loss * prob_loss
        })

    return pd.DataFrame(results)


def print_sensitivity_summary(df: pd.DataFrame, current_credit: float):
    """Print formatted sensitivity analysis summary"""

    print("\n" + "═" * 80)
    print(" CREDIT RECEIVED SENSITIVITY ANALYSIS")
    print("═" * 80)

    # Find current credit row
    current_row = df.iloc[(df['credit_received'] - current_credit).abs().argsort()[:1]]

    if current_row.empty:
        print("\n⚠️  Current credit not found within analyzed range.")
        return

    current_credit_actual = current_row['credit_received'].iloc[0]

    print(f"\n📊 CURRENT CONFIGURATION (Credit: ${current_credit_actual:.2f})")
    print("─" * 80)
    print(f"  • Edge Ratio:      {current_row['edge_ratio_pct'].iloc[0]:.2f}%")
    print(f"  • Expectancy:      ${current_row['expectancy'].iloc[0]:.3f}/share")
    print(f"  • Break-even Win%: {current_row['break_even_pct'].iloc[0]:.1f}%")
    print(f"  • Risk/Reward:     1:{current_row['risk_reward_ratio'].iloc[0]:.2f}")
    print(f"  • ROI/Trade:       {current_row['roi_per_trade_pct'].iloc[0]:.1f}%")

    # Find optimal credit (max edge ratio)
    optimal = df.loc[df['edge_ratio'].idxmax()]

    print(f"\n🎯 OPTIMAL CONFIGURATION (Maximizes Edge)")
    print("─" * 80)
    print(f"  • Credit Received: ${optimal['credit_received']:.2f}")
    print(f"  • Edge Ratio:      {optimal['edge_ratio_pct']:.2f}%")
    print(f"  • Expectancy:      ${optimal['expectancy']:.3f}/share")
    print(f"  • Max Loss:        ${optimal['max_loss']:.2f}")
    print(f"  • Risk/Reward:     1:{optimal['risk_reward_ratio']:.2f}")

    # Find break-even credit (where trade stops being profitable)
    profitable = df[df['expectancy'] > 0]
    if not profitable.empty:
        min_profitable = profitable['credit_received'].min()
        print(f"\n⚠️  MINIMUM PROFITABLE CREDIT: ${min_profitable:.2f}")
        print(f"   (Below this, expectancy becomes negative)")

    # Find approval threshold
    approved = df[df['trade_approved']]
    if not approved.empty:
        min_approved = approved['credit_received'].min()
        print(f"\n✅ MINIMUM APPROVAL CREDIT: ${min_approved:.2f}")
        print(f"   (Below this, edge ratio falls below {df['edge_ratio'].min():.1%})")

    print("\n" + "═" * 80)


def plot_sensitivity_analysis(df: pd.DataFrame, current_credit: float):
    """Create visualization of credit sensitivity"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Credit Received Sensitivity Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Edge Ratio vs Credit
    ax1 = axes[0, 0]
    ax1.plot(df['credit_received'], df['edge_ratio_pct'], 'b-', linewidth=2, label='Edge Ratio')
    ax1.axhline(y=4.0, color='r', linestyle='--', label='Min Edge (4%)')
    ax1.axvline(x=current_credit, color='g', linestyle='--', alpha=0.7, label=f'Current: ${current_credit:.2f}')
    ax1.fill_between(df['credit_received'], 0, df['edge_ratio_pct'],
                     where=(df['edge_ratio_pct'] >= 4), alpha=0.3, color='green')
    ax1.set_xlabel('Credit Received ($)')
    ax1.set_ylabel('Edge Ratio (%)')
    ax1.set_title('Edge Ratio vs Credit Received')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Expectancy vs Credit
    ax2 = axes[0, 1]
    ax2.plot(df['credit_received'], df['expectancy'], 'g-', linewidth=2, label='Expected Value')
    ax2.axhline(y=0, color='r', linestyle='--', label='Breakeven')
    ax2.axvline(x=current_credit, color='g', linestyle='--', alpha=0.7, label=f'Current: ${current_credit:.2f}')
    ax2.fill_between(df['credit_received'], 0, df['expectancy'],
                     where=(df['expectancy'] > 0), alpha=0.3, color='green')
    ax2.set_xlabel('Credit Received ($)')
    ax2.set_ylabel('Expected Value ($/share)')
    ax2.set_title('Expected Value vs Credit Received')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Break-even vs Actual Win Probability
    ax3 = axes[1, 0]
    # Note: regime_win_prob is constant across the DF rows for this calculation
    # We derive it from the break_even logic inverse or pass it in.
    # For plotting consistency, we use the break_even column to infer the line
    regime_win_prob = 1 - (df['expected_loss_amount'] / df['max_loss']).iloc[0] if df['max_loss'].iloc[0] > 0 else 0.8264

    ax3.plot(df['credit_received'], df['break_even_pct'], 'r-', linewidth=2, label='Break-even Win Rate')
    ax3.axhline(y=regime_win_prob * 100, color='b', linestyle='--',
                label=f'Regime Win Rate: {regime_win_prob * 100:.1f}%')
    ax3.axvline(x=current_credit, color='g', linestyle='--', alpha=0.7, label=f'Current: ${current_credit:.2f}')
    ax3.fill_between(df['credit_received'], df['break_even_pct'], regime_win_prob * 100,
                     where=(df['break_even_pct'] <= regime_win_prob * 100),
                     alpha=0.3, color='green')
    ax3.set_xlabel('Credit Received ($)')
    ax3.set_ylabel('Win Rate Required (%)')
    ax3.set_title('Break-even Analysis: Higher Credit = Lower Required Win Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Risk/Reward Ratio
    ax4 = axes[1, 1]
    ax4.plot(df['credit_received'], df['risk_reward_ratio'], 'purple', linewidth=2, label='Risk/Reward Ratio')
    ax4.axvline(x=current_credit, color='g', linestyle='--', alpha=0.7, label=f'Current: ${current_credit:.2f}')
    ax4.set_xlabel('Credit Received ($)')
    ax4.set_ylabel('Risk/Reward Ratio (Risk:Credit)')
    ax4.set_title('Risk/Reward Ratio: Lower is Better')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def find_optimal_credit_strategy(
        df: pd.DataFrame,
        risk_tolerance: str = 'balanced'
) -> Dict:
    """
    Find optimal credit based on different risk preferences.

    Parameters:
    -----------
    df : pd.DataFrame
        Sensitivity analysis dataframe
    risk_tolerance : str
        'conservative', 'balanced', or 'aggressive'

    Returns:
    --------
    Dictionary with optimal strategies
    """

    strategies = {}

    # Conservative: Maximize probability of profit (higher credit)
    conservative = df[df['credit_received'] <= df['credit_received'].quantile(0.75)]
    if not conservative.empty:
        strategies['conservative'] = conservative.loc[conservative['expectancy'].idxmax()]

    # Balanced: Maximize edge ratio
    strategies['balanced'] = df.loc[df['edge_ratio'].idxmax()]

    # Aggressive: Maximize raw expectancy (may require more capital)
    aggressive = df[df['max_loss'] <= df['max_loss'].quantile(0.75)]
    if not aggressive.empty:
        strategies['aggressive'] = aggressive.loc[aggressive['expectancy'].idxmax()]

    return strategies


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Parse arguments from command line
    args = parse_arguments()

    # Assign parameters from args
    REGIME_WIN_PROB = args.regime_win_prob
    SPREAD_WIDTH = args.spread_width
    CURRENT_CREDIT = args.current_credit
    MIN_EDGE_RATIO = args.min_edge_ratio

    # Run sensitivity analysis
    print("🔍 Running credit sensitivity analysis...")
    print(f"   Regime Win Probability: {REGIME_WIN_PROB:.1%}")
    print(f"   Spread Width: ${SPREAD_WIDTH:.2f}")
    print(f"   Current Credit: ${CURRENT_CREDIT:.2f}")
    print(f"   Min Edge Ratio: {MIN_EDGE_RATIO:.1%}")

    # Generate analysis dataframe
    # Adjust range dynamically based on spread width to ensure valid data
    max_possible_credit = SPREAD_WIDTH * 0.99  # Keep slightly below spread width
    credit_min = max(0.50, SPREAD_WIDTH * 0.1)

    sensitivity_df = analyze_credit_sensitivity(
        regime_win_prob=REGIME_WIN_PROB,
        spread_width=SPREAD_WIDTH,
        min_edge_ratio=MIN_EDGE_RATIO,
        credit_range=(credit_min, max_possible_credit),
        num_points=100
    )

    if sensitivity_df.empty:
        print("\n❌ Error: No valid configurations found. Check spread width and credit range.")
    else:
        # Print summary
        print_sensitivity_summary(sensitivity_df, CURRENT_CREDIT)

        # Find optimal strategies
        print("\n" + "═" * 80)
        print(" OPTIMAL CREDIT STRATEGIES BY RISK PROFILE")
        print("═" * 80)

        strategies = find_optimal_credit_strategy(sensitivity_df, 'balanced')

        for risk_type, strategy in strategies.items():
            print(f"\n{risk_type.upper()} STRATEGY:")
            print(f"  • Credit to Receive: ${strategy['credit_received']:.2f}")
            print(f"  • Edge Ratio:        {strategy['edge_ratio_pct']:.2f}%")
            print(f"  • Expectancy:        ${strategy['expectancy']:.3f}/share")
            print(f"  • Max Loss:          ${strategy['max_loss']:.2f}")
            print(f"  • Risk/Reward:       1:{strategy['risk_reward_ratio']:.2f}")

        # Visualization
        if not args.no_plot:
            try:
                plot_sensitivity_analysis(sensitivity_df, CURRENT_CREDIT)
            except Exception as e:
                print(f"\n⚠️  Could not display plot: {e}")
                print("   (Use --no-plot to disable visualization)")

        # Export to CSV for further analysis
        sensitivity_df.to_csv('credit_sensitivity_analysis.csv', index=False)
        print(f"\n📁 Full analysis saved to 'credit_sensitivity_analysis.csv'")

        # Interactive: What-if scenarios
        print("\n" + "═" * 80)
        print(" WHAT-IF SCENARIOS")
        print("═" * 80)

        scenarios = {
            'Market becomes volatile (lower credit)': CURRENT_CREDIT * 0.75,
            'Current market (baseline)': CURRENT_CREDIT,
            'High premium environment': CURRENT_CREDIT * 1.25,
            'Exceptional premium (rare)': CURRENT_CREDIT * 1.50
        }

        for scenario, credit in scenarios.items():
            # Ensure scenario credit is within analyzed range
            if credit < sensitivity_df['credit_received'].min() or credit > sensitivity_df['credit_received'].max():
                continue

            row = sensitivity_df.iloc[(sensitivity_df['credit_received'] - credit).abs().argsort()[:1]]
            if not row.empty:
                print(f"\n{scenario}:")
                print(f"  Credit: ${credit:.2f} → Edge: {row['edge_ratio_pct'].iloc[0]:.1f}% | EV: ${row['expectancy'].iloc[0]:.2f}")
                print(f"  Decision: {'✅ APPROVE' if row['trade_approved'].iloc[0] else '❌ REJECT'}")