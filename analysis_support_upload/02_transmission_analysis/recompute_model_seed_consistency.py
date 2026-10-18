"""Reproduce the model-stratified seed statistics reported in Section 4.5.

Uses archived unit-level seed contrasts; no model training is performed.
Run this after recompute_three_seeds.py if regenerating the whole analysis.
"""
from pathlib import Path
import argparse
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=here/'seed_units_2304.csv')
    parser.add_argument('--out', type=Path, default=here/'model_seed_consistency.csv')
    args = parser.parse_args()
    units = pd.read_csv(args.input)
    assert len(units) == 2304
    assert set(units.model) == {'DLinear', 'LSTM', 'PatchTST'}
    rows = []
    for model, group in units.groupby('model', sort=True):
        valid = group[group.valid]
        same = int(group.all_three_same_sign.sum())
        pure_all = int(valid.pure_seed_count.eq(3).sum())
        pure_any = int(valid.pure_seed_count.gt(0).sum())
        assert len(group) == 768 and len(valid) == 709
        rows.append(dict(
            model=model, nonmean_n=len(group), sign_agreement_count=same,
            sign_agreement_pct=100*same/len(group),
            any_disagreement_count=len(group)-same,
            any_disagreement_pct=100*(len(group)-same)/len(group),
            mean_pairwise_disagreement_pct=100*group.pairwise_disagreement_fraction.mean(),
            valid_n=len(valid), pure_all_three_count=pure_all,
            pure_all_three_pct=100*pure_all/len(valid),
            pure_any_seed_count=pure_any,
            pure_any_seed_pct=100*pure_any/len(valid),
        ))
    result = pd.DataFrame(rows)
    assert result.sign_agreement_count.sum() == 1720
    assert result.pure_all_three_count.sum() == 200
    assert result.pure_any_seed_count.sum() == 677
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
