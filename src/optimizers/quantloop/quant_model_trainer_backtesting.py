try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import json
import argparse
from argparse import ArgumentParser, Namespace

from optimizers.quantloop.quant_model_trainer import entry_point as quant_model_trainer
from fetchers.data_factory import factory_load_data
from utils import (
    get_next_step,
    round_price_for_put_credit_spread,
    round_price_for_call_credit_spread,
    previous_weekday_with_check,
)
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """
    Keep argparse default values in help output while preserving
    multiline formatting for the epilog.
    """


def _safe_divide(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _fraction_below(values) -> float:
    return _safe_divide(sum(1 for value in values if value < 0), len(values))


def _fraction_above(values) -> float:
    return _safe_divide(sum(1 for value in values if value > 0), len(values))


def _optional_string(value: str):
    """
    Convert empty/'none'/'null' CLI strings to None.
    """
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def _target_type(value: str):
    """
    Validate target_type while allowing omission/None.
    """
    value = _optional_string(value)
    if value is None:
        return None

    value = value.strip().lower()
    if value not in {"soft_higher", "in_between"}:
        raise argparse.ArgumentTypeError(
            "target_type must be 'soft_higher', 'in_between', or omitted/none"
        )
    return value


def parse_arg(argv=None) -> Namespace:
    """
    Parse all parameters that were previously hard-coded in entry().

    Use --help to display the commented parameter descriptions.
    """
    parser = ArgumentParser(
        description=(
            "Backtest/evaluate quant model realtime predictions for "
            "credit-spread-style option strategies."
        ),
        formatter_class=_HelpFormatter,
        epilog="",
    )
    parser.add_argument('--dataset-id', type=str, default='month', help='Dataset identifier')
    parser.add_argument(
        "--modifier-pct-pred-0",
        dest="modifier_pct_pred_0",
        type=float,
        default=1.,
        help=(
            "soft_higher PRED==0 modifier. "
            "Target = 1 + modifier * target_pourcentage_used. "
            "Used to sell a CALL CREDIT SPREAD."
        ),
    )

    parser.add_argument(
        "--modifier-pct-pred-1",
        dest="modifier_pct_pred_1",
        type=float,
        default=1.,
        help=(
            "soft_higher PRED==1 modifier. "
            "Target = 1 - modifier * target_pourcentage_used. "
            "Used to sell a PUT CREDIT SPREAD."
        ),
    )

    parser.add_argument(
        "--modifier-pct-pred-low",
        dest="modifier_pct_pred_low",
        type=float,
        default=1.0,
        help=(
            "in_between low-end modifier. "
            "Target low = 1 - modifier * target_pourcentage_used."
        ),
    )

    parser.add_argument(
        "--modifier-pct-pred-high",
        dest="modifier_pct_pred_high",
        type=float,
        default=1.0,
        help=(
            "in_between high-end modifier. "
            "Target high = 1 + modifier * target_pourcentage_used."
        ),
    )

    parser.add_argument(
        "--start-go-back-length",
        dest="start_go_back_length",
        type=int,
        default=2,
        help=(
            "First clip_n/go-back length used in the loop. "
            "Must be greater than 1."
        ),
    )

    parser.add_argument(
        "--end-go-back-length",
        dest="end_go_back_length",
        type=int,
        default=30,
        help="Exclusive end of clip_n/go-back length range.",
    )

    parser.add_argument(
        "--quant-dataset",
        dest="quant_dataset",
        type=str,
        default=None,
        help="Path to the compiled quant dataset.",
    )

    parser.add_argument(
        "--quant-model",
        dest="quant_model",
        type=str,
        default=None,
        help="Path to the trained quant model pickle.",
    )

    parser.add_argument(
        "--use-tqdm",
        dest="use_tqdm",
        action="store_true",
        help="Show tqdm progress bar instead of printing each trade line.",
    )

    parser.add_argument(
        "--no-use-tqdm",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm and print each trade line.",
    )

    parser.set_defaults(use_tqdm=False)

    parser.add_argument(
        "--target-type",
        dest="target_type",
        type=_target_type,
        default=None,
        help=(
            "Force target type: 'soft_higher' or 'in_between'. "
            "If omitted/None, inferred from first trainer result."
        ),
    )

    parser.add_argument(
        "--mode-in-between",
        dest="mode_in_between",
        type=str,
        default="pure_iron_condor",
        choices=["one_leg", "iron_condor", "pure_iron_condor"],
        help=(
            "Mode used when target_type='in_between'. "
            "one_leg: use only one side based on prediction. "
            "iron_condor: prediction-dependent range logic. "
            "pure_iron_condor: win only if expiry stays inside range."
        ),
    )

    parser.add_argument(
        "--most-recent-dday-date",
        dest="most_recent_dday_date",
        type=_optional_string,
        default=None,
        help=(
            "Optional display override for the most recent D-day date. "
            "Normally inferred from the first processed result."
        ),
    )

    parser.add_argument(
        "--least-recent-dday-date",
        dest="least_recent_dday_date",
        type=_optional_string,
        default=None,
        help=(
            "Optional display override for the least recent D-day date. "
            "When omitted/None, automatically updated during the loop."
        ),
    )
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    return parser.parse_args(argv)


def _run_backtesting(config, df_mom, dataset_id, dry_run=False):
    modifier_pct_pred_0 = config.modifier_pct_pred_0
    modifier_pct_pred_1 = config.modifier_pct_pred_1
    modifier_pct_pred_low = config.modifier_pct_pred_low
    modifier_pct_pred_high = config.modifier_pct_pred_high

    start_go_back_length = config.start_go_back_length
    end_go_back_length = config.end_go_back_length

    quant_dataset = config.quant_dataset
    quant_model = config.quant_model

    use_tqdm = config.use_tqdm
    target_type = config.target_type

    most_recent_dday_date = config.most_recent_dday_date
    least_recent_dday_date = config.least_recent_dday_date
    least_recent_dday_date_override = least_recent_dday_date is not None

    mode_in_between = config.mode_in_between

    results, results_pred_0, results_pred_1 = {}, {}, {}
    results_pred_0_diff_on_close, results_pred_1_diff_on_close = {}, {}

    assert df_mom is not None

    fleche_haut = "↑"
    fleche_bas = "↓"

    rouge_clair = "\033[91m"
    vert = "\033[32m"
    reset = "\033[0m"

    context_displayed = False

    target_pourcentage_used_pred_0 = None
    target_pourcentage_used_pred_1 = None

    target_pourcentage_used_pred_low = None
    target_pourcentage_used_pred_high = None

    # sinon, on a pas les données du future complètement
    assert start_go_back_length > 1
    end_go_back_length = start_go_back_length+1 if end_go_back_length <= start_go_back_length else end_go_back_length
    iterator = (
        tqdm(range(start_go_back_length, end_go_back_length))
        if use_tqdm
        else range(start_go_back_length, end_go_back_length)
    )

    for clip_n in iterator:
        try:
            result = quant_model_trainer(
                args=Namespace(
                    realtime=True,
                    model_path=quant_model,
                    dataset=quant_dataset,
                    clip_n=clip_n,
                    verbose=False,
                )
            )
            if dry_run: return result
        except IndexError:
            # Plus de données
            break
        assert dataset_id == result["dataset_id"]
        prediction = int(result["realtime_prediction"])

        dday_value = df_mom.loc[result["last_date"]]["Close"].values[0]
        assert result["last_value"] == dday_value

        dday_date = result["last_date"]

        most_recent_dday_date = (
            dday_date if most_recent_dday_date is None else most_recent_dday_date
        )

        if not least_recent_dday_date_override:
            least_recent_dday_date = dday_date

        expiring_date = get_next_step(
            the_date=dday_date,
            dataset_id=result["dataset_id"],
            nn=result["look_ahead"],
        )

        if target_type is None:
            target_type = result["target_type"]
        else:
            assert target_type == result["target_type"]

        try:
            value_on_expiring_day = df_mom.loc[expiring_date]["Close"].values[0]
        except KeyError:
            value_on_expiring_day = df_mom.loc[
                previous_weekday_with_check(expiring_date, df_mom)
            ]["Close"].values[0]

        if target_type == "soft_higher":
            if target_pourcentage_used_pred_0 is None:
                target_pourcentage_used_pred_0 = 1 + modifier_pct_pred_0 * result["target_pourcentage_used"]

            if target_pourcentage_used_pred_1 is None:
                target_pourcentage_used_pred_1 = 1 - modifier_pct_pred_1 * result["target_pourcentage_used"]

            if not context_displayed and config.verbose:
                context_displayed = True
                print(f"\tTarget type: {target_type}")
                print(f"\tDataset  : {result['dataset_id']}")
                print(f"\tLookahead: {result['look_ahead']}")
                print(
                    f"\tTarget pourcentage modification"
                    f"\n\t\t when PRED==0: {target_pourcentage_used_pred_0} :: we sell a CALL CREDIT SPREAD"
                    f"\n\t\t when PRED==1: {target_pourcentage_used_pred_1} :: we sell a PUT CREDIT SPREAD"
                )

            win = 0

            if prediction == 0:
                strike_price = int(dday_value * target_pourcentage_used_pred_0)
                strike_price = int(round_price_for_call_credit_spread(strike_price))
            else:
                strike_price = int(dday_value * target_pourcentage_used_pred_1)
                strike_price = int(round_price_for_put_credit_spread(strike_price))

            diff_on_close = (value_on_expiring_day - strike_price) / strike_price

            if prediction == 1 and strike_price <= value_on_expiring_day:
                win = 1

            if prediction == 0 and strike_price >= value_on_expiring_day:
                win = 1

            if not use_tqdm:
                if config.verbose:
                    prediction_display = (
                        f"{vert}{fleche_haut}{reset}"
                        if prediction == 1
                        else f"{rouge_clair}{fleche_bas}{reset}"
                    )

                    print(
                        f"Entering on {dday_date} @{strike_price} expiring on {expiring_date} "
                        f"(future value is {value_on_expiring_day:.0f}) "
                        f"Predicted {prediction_display} , got diff of {diff_on_close:.2%} "
                        f"{' ' * 10}{'✅' if win else '❌'}"
                    )

            results.update({dday_date: win})

            if prediction == 0:
                results_pred_0.update({dday_date: win})
                results_pred_0_diff_on_close.update({dday_date: diff_on_close})

            if prediction == 1:
                results_pred_1.update({dday_date: win})
                results_pred_1_diff_on_close.update({dday_date: diff_on_close})

        if target_type == "in_between":
            if target_pourcentage_used_pred_high is None:
                target_pourcentage_used_pred_high = (
                        1 + modifier_pct_pred_high * result["target_pourcentage_used"]
                )

            if target_pourcentage_used_pred_low is None:
                target_pourcentage_used_pred_low = (
                        1 - modifier_pct_pred_low * result["target_pourcentage_used"]
                )

            if not context_displayed:
                context_displayed = True
                print(f"\tTarget type: {target_type}")
                print(f"\tMode for {target_type}: {mode_in_between}")
                print(f"\tDataset  : {result['dataset_id']}")
                print(f"\tLookahead: {result['look_ahead']}")

                if mode_in_between == "one_leg":
                    print(
                        f"\tTarget pourcentage modification"
                        f"\n\t\t when PRED==0: {target_pourcentage_used_pred_high} :: we sell a CALL CREDIT SPREAD"
                        f"\n\t\t when PRED==1: {target_pourcentage_used_pred_low} :: we sell a PUT CREDIT SPREAD"
                    )
                else:
                    print(
                        f"\tTarget pourcentage modification"
                        f"\n\t\t low end: {target_pourcentage_used_pred_low}"
                        f"\n\t\t high end: {target_pourcentage_used_pred_high}"
                    )

            win = 0

            strike_price_low = int(
                round_price_for_put_credit_spread(
                    dday_value * target_pourcentage_used_pred_low
                )
            )

            strike_price_high = int(
                round_price_for_call_credit_spread(
                    dday_value * target_pourcentage_used_pred_high
                )
            )

            diff_on_close_low = (
                                        value_on_expiring_day - strike_price_low
                                ) / strike_price_low

            diff_on_close_high = (
                                         value_on_expiring_day - strike_price_high
                                 ) / strike_price_high

            if mode_in_between in ["iron_condor", "pure_iron_condor"]:
                if mode_in_between in ["iron_condor"]:
                    if (
                            prediction == 1
                            and strike_price_low <= value_on_expiring_day <= strike_price_high
                    ):
                        win = 1

                    if prediction == 0 and (
                            value_on_expiring_day < strike_price_low
                            or value_on_expiring_day > strike_price_high
                    ):
                        win = 1

                if mode_in_between in ["pure_iron_condor"]:
                    if strike_price_low <= value_on_expiring_day <= strike_price_high:
                        win = 1

            elif mode_in_between == "one_leg":
                if prediction == 1 and strike_price_low <= value_on_expiring_day:
                    win = 1

                if prediction == 0 and strike_price_high >= value_on_expiring_day:
                    win = 1

            else:
                assert False, f"{mode_in_between=}"

            if not use_tqdm:
                if config.verbose:
                    if prediction == 1:
                        predicted_display = (
                            f"{vert}[{strike_price_low}::{strike_price_high}]{reset}"
                        )
                    else:
                        predicted_display = (
                            f"{rouge_clair}]{strike_price_low}::{strike_price_high}[{reset}"
                        )

                    print(
                        f"Entering on {dday_date} @{strike_price_low}@{strike_price_high} expiring on {expiring_date} "
                        f"(future value is {value_on_expiring_day:.0f}) "
                        f"Predicted {predicted_display} , "
                        f"got diff of {diff_on_close_low:.2%}/{diff_on_close_high:.2%} "
                        f"{' ' * 10}{'✅' if win else '❌'}"
                    )

            results.update({dday_date: win})

            if prediction == 0:
                results_pred_0.update({dday_date: win})
                results_pred_0_diff_on_close.update({dday_date: diff_on_close_low})

            if prediction == 1:
                results_pred_1.update({dday_date: win})
                results_pred_1_diff_on_close.update({dday_date: diff_on_close_high})

    assert len(results) > 0

    win_rate = _safe_divide(sum(results.values()), len(results))
    win_rate_pred_0 = _safe_divide(sum(results_pred_0.values()), len(results_pred_0))
    win_rate_pred_1 = _safe_divide(sum(results_pred_1.values()), len(results_pred_1))

    if target_type == "soft_higher":
        if config.verbose:
            print(
                f"{target_type} DDate from {most_recent_dday_date} to {least_recent_dday_date}   "
                f"WR: {win_rate:.2%}    WR pred 0: {win_rate_pred_0:.2%}    WR pred 1: {win_rate_pred_1:.2%}"
            )

            print(f"\t PRED=0 {rouge_clair}{fleche_bas}{reset} --> CALL CREDIT SPREAD ")
            print(f"\t PRED=1 {vert}{fleche_haut}{reset} --> PUT CREDIT SPREAD ")

            print("Le pourcentage de (close_t+1 - close_t)/close_t selon la prediction")

        pct = _fraction_below(results_pred_0_diff_on_close.values())
        pct2 = _fraction_above(results_pred_0_diff_on_close.values())
        if config.verbose:print(f"\t PRED=0 % < 0 : {pct:.2%}  {' ' * 5}% > 0 : {pct2:.2%}")

        pct = _fraction_below(results_pred_1_diff_on_close.values())
        pct2 = _fraction_above(results_pred_1_diff_on_close.values())
        if config.verbose:print(f"\t PRED=1 % < 0 : {pct:.2%}  {' ' * 5}% > 0 : {pct2:.2%}")
        return {"win_rate": win_rate, "win_rate_pred_0": win_rate_pred_0, "win_rate_pred_1": win_rate_pred_1,}

    if target_type == "in_between":
        print(
            f"{target_type}::{mode_in_between} DDate from {most_recent_dday_date} to {least_recent_dday_date}   "
            f"WR: {win_rate:.2%}    WR pred 0: {win_rate_pred_0:.2%}    WR pred 1: {win_rate_pred_1:.2%}"
        )

        print("   Pourcentage resultant de (close t - close t+1) / close t -- Low leg")

        pct = _fraction_below(results_pred_0_diff_on_close.values())
        pct2 = _fraction_above(results_pred_0_diff_on_close.values())

        extra = (
            "(hors RANGE), utiliser CALL CREDIT SPREAD"
            if mode_in_between == "one_leg"
            else ""
        )

        print(
            f"\t PRED=0 % < 0 : {pct:.2%}   {' ' * 5}% > 0 : {pct2:.2%}"
            + (f"   {extra}" if extra else "")
        )

        print("   Pourcentage resultant de (close t - close t+1) / close t -- High leg")

        pct = _fraction_below(results_pred_1_diff_on_close.values())
        pct2 = _fraction_above(results_pred_1_diff_on_close.values())

        extra = (
            "(en RANGE), utiliser PUT CREDIT SPREAD"
            if mode_in_between == "one_leg"
            else ""
        )

        print(
            f"\t PRED=1 % > 0 : {pct2:.2%}   {' ' * 5}% < 0 : {pct:.2%}"
            + (f"   {extra}" if extra else "")
        )

    return None


def _run_iterate_backtesting(args, df_mom):
    end_go_back_length__2__optimal_target_percentage = {}
    for _end_go_back_length in range(6, 492, 6):
        args_copie = argparse.Namespace(**vars(args) | {"verbose": False, "end_go_back_length": _end_go_back_length})

        # Run it once to get the parameters used by the model
        model_parameters = _run_backtesting(config=argparse.Namespace(**vars(args_copie) | {"verbose": False, "end_go_back_length": 2}), df_mom=df_mom.copy(), dataset_id=args.dataset_id, dry_run=True)
        assert args.dataset_id == model_parameters["dataset_id"]

        print(f"Using:\n\tData: {args_copie.quant_dataset}"
              f"\n\tModel: {args_copie.quant_model}"
              f"\n\tLength backtesting : {args_copie.end_go_back_length} {args.dataset_id}"
              f"\n\tTarget type: {model_parameters['target_type']}"
              f"\n\tTarget pourcentage used by model: {model_parameters['target_pourcentage_used']}")

        if "soft_higher" == model_parameters['target_type']:
            winrate__2__all_target_percentage_if_pred_0, winrate__2__all_target_percentage_if_pred_1 = None, None
            # strike *= 1 + M*cte  where factor = 1 + M*cte and 0<cte<1
            decreasing_modifier_call_credit_spread = np.linspace(0, 1, 1001)
            increasing_modifier_call_credit_spread = np.linspace(1, 15, 30)
            # strike *= 1 - M*cte  where factor = 1 - M*cte and 0<cte<1
            increasing_modifier_put_credit_spread = np.linspace(1, 33, 66)
            decreasing_modifier_put_credit_spread = np.linspace(0, 10, 101)
            # Baseline
            baseline = _run_backtesting(config=argparse.Namespace(**vars(args_copie)), df_mom=df_mom.copy(), dataset_id=args.dataset_id)
            print(f"[BASELINE]  Win Rate: {baseline['win_rate']:.2%}   "
                  f"Win Rate Pred 0: {baseline['win_rate_pred_0']:.2%}   "
                  f"Win Rate Pred 1: {baseline['win_rate_pred_1']:.2%}")
            winrate__2__optimal_modifiers__call_credit_spread = {baseline['win_rate_pred_0']: model_parameters['target_pourcentage_used']}
            if baseline["win_rate_pred_0"] == 1:
                # Squeeze it
                print(f"[PRED=0] Need to squeeze (decrease the factor/decrease M , S *= 1+M*{model_parameters['target_pourcentage_used']}) "
                      f"the CALL CREDIT SPREAD because baseline has a win rate of 100%")
                for _modifier__call_credit_spread in decreasing_modifier_call_credit_spread:
                    result = _run_backtesting(config=argparse.Namespace(**vars(args_copie) | {"modifier_pct_pred_0": _modifier__call_credit_spread}),
                                              df_mom=df_mom.copy(), dataset_id=args.dataset_id, )
                    if result["win_rate_pred_0"] == 1:
                        winrate__2__optimal_modifiers__call_credit_spread.update({result["win_rate_pred_0"]: float(_modifier__call_credit_spread)})
                        break
            else:
                # Relax it
                print(f"[PRED=0] Need to relax (increase the factor/increase M , because S *= 1+M*{model_parameters['target_pourcentage_used']}) "
                      f"the CALL CREDIT SPREAD because baseline has a win rate of {baseline['win_rate_pred_0']:.0%}, which is below 100%")
                for _modifier__call_credit_spread in increasing_modifier_call_credit_spread:
                    result = _run_backtesting(config=argparse.Namespace(**vars(args_copie) | {"modifier_pct_pred_0": _modifier__call_credit_spread}),
                                              df_mom=df_mom.copy(), dataset_id=args.dataset_id, )
                    if all(k < result["win_rate_pred_0"] for k in winrate__2__optimal_modifiers__call_credit_spread):
                        winrate__2__optimal_modifiers__call_credit_spread.update({result["win_rate_pred_0"]: float(_modifier__call_credit_spread)})
                    if result["win_rate_pred_0"] == 1:
                        break
            winrate__2__all_target_percentage_if_pred_0 = {k: v * model_parameters['target_pourcentage_used'] for k, v in winrate__2__optimal_modifiers__call_credit_spread.items()}
            max_win_rate, optimal_modifier__call_credit_spread = max(winrate__2__optimal_modifiers__call_credit_spread.items(), key=lambda item: item[1])
            print(f"[PRED=0] From F={1 + model_parameters['target_pourcentage_used']}({baseline['win_rate_pred_0']:.0%}) "
                  f"to F={1 + optimal_modifier__call_credit_spread * model_parameters['target_pourcentage_used']:.8f}({max_win_rate:.2%}) "
                  f"")
            for k, v in winrate__2__optimal_modifiers__call_credit_spread.items():
                print(f"[PRED=0] \t{1 + v * model_parameters['target_pourcentage_used']:.8f}({k:.2%})")

            win_rate__2__optimal_modifiers__put_credit_spread = {baseline['win_rate_pred_1']: model_parameters['target_pourcentage_used']}
            if baseline["win_rate_pred_1"] == 1:
                # Squeeze it
                print(f"[PRED=1] Need to squeeze (increase the factor/decrease M , because S *= 1-M*{model_parameters['target_pourcentage_used']}) "
                      f"the PUT CREDIT SPREAD because baseline has a win rate of 100%")
                for _modifier__put_credit_spread in decreasing_modifier_put_credit_spread:
                    result = _run_backtesting(config=argparse.Namespace(**vars(args_copie) | {"modifier_pct_pred_1": _modifier__put_credit_spread}),
                                              df_mom=df_mom.copy(), dataset_id=args.dataset_id, )
                    if result["win_rate_pred_1"] == 1:
                        win_rate__2__optimal_modifiers__put_credit_spread.update({result["win_rate_pred_1"]: float(_modifier__put_credit_spread)})
                        break
            else:
                # Relax it
                print(f"[PRED=1] Need to relax (decrease the factor/increase M , S *= 1-M*{model_parameters['target_pourcentage_used']}) "
                      f"the PUT CREDIT SPREAD because baseline has a win rate of {baseline['win_rate_pred_1']:.0%}, which is below 100%")
                for _modifier__put_credit_spread in increasing_modifier_put_credit_spread:
                    result = _run_backtesting(config=argparse.Namespace(**vars(args_copie) | {"modifier_pct_pred_1": _modifier__put_credit_spread}),
                                              df_mom=df_mom.copy(), dataset_id=args.dataset_id, )
                    if all(k < result["win_rate_pred_1"] for k in win_rate__2__optimal_modifiers__put_credit_spread):
                        win_rate__2__optimal_modifiers__put_credit_spread.update({result["win_rate_pred_1"]: float(_modifier__put_credit_spread)})
                    if result["win_rate_pred_1"] == 1:
                        break
            winrate__2__all_target_percentage_if_pred_1 = {k: v * model_parameters['target_pourcentage_used'] for k, v in win_rate__2__optimal_modifiers__put_credit_spread.items()}
            max_win_rate, optimal_modifier__put_credit_spread = max(win_rate__2__optimal_modifiers__put_credit_spread.items(), key=lambda item: item[1])
            print(f"[PRED=1] From F={1 - model_parameters['target_pourcentage_used']}({baseline['win_rate_pred_1']:.0%}) "
                  f"to F={1 - optimal_modifier__put_credit_spread * model_parameters['target_pourcentage_used']:.8f}({max_win_rate:.2%}) "
                  f"")
            for k, v in win_rate__2__optimal_modifiers__put_credit_spread.items():
                print(f"[PRED=1] \t{1 - v * model_parameters['target_pourcentage_used']:.8f}({k:.2%})")
            print(f"[PRED=0]\n\t{json.dumps({k: f"{v:.8f}" if isinstance(v, (int, float)) else v for k, v in winrate__2__all_target_percentage_if_pred_0.items()}, indent=4, sort_keys=True)}")
            print(f"[PRED=1]\n\t{json.dumps({k: f"{v:.8f}" if isinstance(v, (int, float)) else v for k, v in winrate__2__all_target_percentage_if_pred_1.items()}, indent=4, sort_keys=True)}")
            end_go_back_length__2__optimal_target_percentage.update({_end_go_back_length: {"pred_0":winrate__2__all_target_percentage_if_pred_0,
                                                                        "pred_1":winrate__2__all_target_percentage_if_pred_1}})
    return end_go_back_length__2__optimal_target_percentage


def entry(args):
    dataset_id, ticker = "month", "^GSPC"
    df_mom = factory_load_data(_dataset_id=dataset_id, _ticker=ticker, _args={}, )

    args.quant_dataset = r"D:\Finance\compiled_models\quant\2026.05.06\generated__combined_month_macro.data"
    args.quant_model = r"D:\Finance\compiled_models\quant\2026.05.06\best_model_RandomForestClassifier_soft_higher_20260506_004718.pkl"

    end_go_back_length__2__optimal_target_percentage = _run_iterate_backtesting(args=args, df_mom=df_mom)
    output_filename_result = os.path.join(Path(args.quant_dataset).parent, "optimal_target_percentage.json")
    with open(output_filename_result, "w", encoding="utf-8") as f: json.dump(end_go_back_length__2__optimal_target_percentage, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = parse_arg()
    entry(args=args)