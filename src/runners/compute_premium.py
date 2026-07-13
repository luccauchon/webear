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


def calculate_options_expectancy():
    print("--- Options Credit Spread / Iron Condor Calculator ---")
    print("Determine your minimum required credit for long-term profitability.\n")

    while True:
        try:
            # 1. Get User Inputs
            win_rate_input = float(input("Enter your estimated Win Rate (e.g., 75 for 75%): "))
            win_rate = win_rate_input / 100.0

            # Note: In options, "Worst case loss" is usually defined by the spread width.
            # E.g., a $5 wide spread on SPX = $500 max loss.
            spread_width = float(input("Enter the Spread Width / Max Loss in $ (e.g., 500 for a 5pts wide spread): "))

            if win_rate <= 0 or win_rate >= 1:
                print("Win rate must be between 1% and 99%.\n")
                continue
            if spread_width <= 0:
                print("Spread width must be greater than 0.\n")
                continue

            # 2. Core Math
            loss_rate = 1.0 - win_rate

            # The Break-Even Credit formula: Credit = Loss_Rate * Spread_Width
            breakeven_credit = loss_rate * spread_width
            breakeven_actual_loss = spread_width - breakeven_credit

            # 3. Display Results
            print("\n" + "=" * 50)
            print(f"RESULTS FOR {win_rate_input}% WIN RATE & ${spread_width:.0f} SPREAD WIDTH")
            print("=" * 50)
            print(f"Loss Rate: {loss_rate * 100:.8f}%")
            print(f"MINIMUM BREAK-EVEN CREDIT: ${breakeven_credit:.2f}")
            print(f"(At this credit, your actual worst-case loss is ${breakeven_actual_loss:.2f})")
            print("=" * 50)

            # 4. Scenario Analyzer (Target Profit)
            print("\n--- Scenario Analyzer (Expected Value per trade) ---")
            print("How much credit do you need to make a specific average profit?\n")

            # Test a few common scenarios
            scenarios = [0, 1, 5, 10, 20, 25, 50, 100]
            for target_profit in scenarios:
                # Formula: Required Credit = Target_Profit + (Loss_Rate * Spread_Width)
                req_credit = target_profit + breakeven_credit
                req_actual_loss = spread_width - req_credit

                if target_profit == 0:
                    label = "Break-Even"
                else:
                    label = f"Target +${target_profit} EV"

                print(f"{label:15} | Collect ${req_credit:6.2f} | Risk ${req_actual_loss:6.2f}")

            print("=" * 50 + "\n")

        except ValueError:
            print("Invalid input. Please enter numerical values.\n")

        # Ask to run again
        again = input("Calculate another setup? (y/n): ").lower()
        if again != 'y':
            print("Happy trading!")
            break


if __name__ == "__main__":
    calculate_options_expectancy()