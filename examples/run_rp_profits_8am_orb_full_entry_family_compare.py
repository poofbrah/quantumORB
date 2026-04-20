from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from run_rp_profits_8am_orb_vwap_displacement_full import INPUT_PATH, format_metric, run_mode

OUTPUT_DIR = ROOT / "outputs" / "rp_profits_8am_orb_full_entry_family_compare"
ENTRY_MODES = ["displacement_vwap_pullback", "range_reentry_vwap", "hybrid"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for entry_mode in ENTRY_MODES:
        payload, _ = run_mode(entry_mode, frame)
        rows.append(payload)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    print(f"rows={len(frame)}")
    print("entry_mode                  setups_detected  trades_executed  win_rate  profit_factor  net_pnl  max_drawdown")
    for row in rows:
        print(
            f"{row['entry_mode']:<27} {row['setups_detected']:<15} {row['trades_executed']:<16} {format_metric(row['win_rate']):<8} {format_metric(row['profit_factor']):<14} {format_metric(row['net_pnl']):<8} {format_metric(row['max_drawdown'])}"
        )
    print(f"outputs={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
