"""Recompute leave-one-dataset-out influence checks from 8,640 archived runs.

This is a composition sensitivity analysis, not held-out-dataset training.
Forecasting errors are averaged across seeds before contrasts with Mean.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

KEY = ['dataset', 'model', 'missing_mode', 'missing_rate', 'pred_len', 'impute_method']
BASE = KEY[:-1]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=here.parent/'01_raw_data/online_results_8640.csv')
    parser.add_argument('--out', type=Path, default=here/'leave_one_dataset_out.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    assert len(raw) == 8640 and not raw.duplicated(KEY+['pred_seed']).any()
    assert raw.status.eq('success').all()
    assert np.isfinite(raw[['ie_mae', 'fe_mae']]).all().all()
    assert raw.groupby(KEY).pred_seed.agg(set).map(lambda s: s == {42, 123, 2026}).all()
    means = raw.groupby(KEY, sort=True)[['ie_mae', 'fe_mae']].mean().reset_index()
    assert len(means) == 2880
    anchors = means[means.impute_method.eq('mean')][BASE+['ie_mae', 'fe_mae']]
    contrasts = means[means.impute_method.ne('mean')].merge(
        anchors.rename(columns={'ie_mae': 'anchor_ie', 'fe_mae': 'anchor_fe'}),
        on=BASE, validate='many_to_one')
    assert len(contrasts) == 2304
    contrasts['delta_ie'] = contrasts.ie_mae-contrasts.anchor_ie
    contrasts['delta_fe'] = contrasts.fe_mae-contrasts.anchor_fe
    contrasts['valid'] = contrasts.delta_ie.abs() > 0.01
    contrasts['pure'] = contrasts.valid & (contrasts.delta_ie < 0) & (contrasts.delta_fe > 0)
    contrasts['tau'] = (contrasts.delta_fe/contrasts.delta_ie).where(contrasts.valid)
    contrasts['relative_ie'] = 100*contrasts.delta_ie/contrasts.anchor_ie
    contrasts['relative_fe'] = 100*contrasts.delta_fe/contrasts.anchor_fe
    rows = []
    for excluded in [None, *sorted(contrasts.dataset.unique())]:
        retained = contrasts if excluded is None else contrasts[contrasts.dataset.ne(excluded)]
        valid = retained[retained.valid]
        row = dict(excluded_dataset=excluded or 'None (all datasets)',
                   datasets_retained=retained.dataset.nunique(),
                   nonmean_n=len(retained), valid_n=len(valid),
                   pure_count=int(valid.pure.sum()), pure_pct=100*valid.pure.mean(),
                   tau_median=valid.tau.median(),
                   relative_ie_median=valid.relative_ie.median(),
                   relative_fe_median=valid.relative_fe.median())
        for model, group in valid.groupby('model', sort=True):
            row[model+'_valid_n'] = len(group)
            row[model+'_pure_count'] = int(group.pure.sum())
            row[model+'_pure_pct'] = 100*group.pure.mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
