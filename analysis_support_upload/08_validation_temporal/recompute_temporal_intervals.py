import time
import math
import numpy as np
import pandas as pd
from pathlib import Path

from argparse import ArgumentParser

args_parser = ArgumentParser()
args_parser.add_argument('--origin-stats', type=Path, required=True)
args_parser.add_argument('--out', type=Path, required=True)
arguments = args_parser.parse_args()
ORG_DIR = arguments.origin_stats
RES_DIR = arguments.out
RES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [('ETT', 'ETTh1'), ('KDD', 'KDD-Beijing'), ('SWaT', 'SWaT'), ('Weather', 'Weather')]
HORIZONS = [24, 48, 96, 192]
MODELS = ['DLinear', 'LSTM', 'PatchTST']
MODES = ['MCAR', 'MAR_Block', 'MNAR']
RATES = [0.1, 0.3, 0.5, 0.7]
SEEDS = [42, 123, 2026]
NON_MEAN = ['spline', 'knn']

def macro_metric_from_sums(abs_sum, cnt):
    # abs_sum, cnt: (..., C)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_feat = np.where(cnt > 0, abs_sum / cnt, np.nan)
    return np.nanmean(per_feat, axis=-1)

def compute_tau_stats(ie_phi, ie_mean, fe_phi_3seed, fe_mean_3seed):
    # Arrays of shape (..., P) where P = 288 per dataset or 1152 total
    d_ie = ie_phi - ie_mean
    d_fe = fe_phi_3seed - fe_mean_3seed
    valid = np.abs(d_ie) > 0.01
    pure_inv = (d_ie < -0.01) & (d_fe > 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(valid, d_fe / d_ie, np.nan)
    return d_ie, d_fe, valid, pure_inv, tau

def build_index_maps(ds_key):
    # Load h=24 origins as reference timeline
    data_h = {}
    for h in HORIZONS:
        fe_z = np.load(ORG_DIR / f"{ds_key}_h{h}_fe_per_origin.npz", allow_pickle=False)
        ie_z = np.load(ORG_DIR / f"{ds_key}_h{h}_ie_per_origin.npz", allow_pickle=False)
        origins_fe = fe_z['origin_indices']
        origins_ie = ie_z['origin_indices']
        assert np.array_equal(origins_fe, origins_ie)
        fe_keys = [str(k) for k in fe_z['meta_keys']]
        ie_keys = [str(k) for k in ie_z['meta_keys']]
        fe_map = {k: i for i, k in enumerate(fe_keys)}
        ie_map = {k: i for i, k in enumerate(ie_keys)}

        # Build aligned pair indices for the 72 non-Mean pairs at this horizon (3 models x 3 modes x 4 rates x 2 imps)
        # Each pair has 3 seed indices for phi, 3 seed indices for mean, 1 ie index for phi, 1 ie index for mean
        fe_phi_idx = []
        fe_mean_idx = []
        ie_phi_idx = []
        ie_mean_idx = []
        pair_meta = []
        for m in MODELS:
            for mode in MODES:
                for rate in RATES:
                    for imp in NON_MEAN:
                        fe_phi_idx.append([fe_map[f"{m}|{mode}|{rate}|{imp}|{s}"] for s in SEEDS])
                        fe_mean_idx.append([fe_map[f"{m}|{mode}|{rate}|mean|{s}"] for s in SEEDS])
                        ie_phi_idx.append(ie_map[f"{mode}|{rate}|{imp}"])
                        ie_mean_idx.append(ie_map[f"{mode}|{rate}|mean"])
                        pair_meta.append((m, mode, rate, h, imp))

        data_h[h] = {
            'origins': origins_fe,
            'N': len(origins_fe),
            'fe_abs': fe_z['fe_abs_sum'],   # (324, N, C)
            'fe_cnt': fe_z['fe_count'],     # (324, N, C)
            'ie_abs': ie_z['ie_abs_sum'],   # (36, N, C)
            'ie_cnt': ie_z['ie_count'],     # (36, N, C)
            'fe_phi_idx': np.array(fe_phi_idx, dtype=np.int64),   # (72, 3)
            'fe_mean_idx': np.array(fe_mean_idx, dtype=np.int64), # (72, 3)
            'ie_phi_idx': np.array(ie_phi_idx, dtype=np.int64),   # (72,)
            'ie_mean_idx': np.array(ie_mean_idx, dtype=np.int64), # (72,)
            'pair_meta': pair_meta,
        }

    # Verify prefix property across horizons
    o24 = data_h[24]['origins']
    for h in HORIZONS:
        nh = data_h[h]['N']
        assert np.array_equal(data_h[h]['origins'], o24[:nh]), f"{ds_key} h={h} origins are not prefix of h=24!"
    return data_h

def main():
    t0 = time.time()
    all_ds_data = {}
    full_rows = []

    print("=== 1. Full-Origin Reconstruction Verification ===")
    for ds_key, ds_label in DATASETS:
        dh = build_index_maps(ds_key)
        all_ds_data[ds_key] = dh
        counts_str = ", ".join([f"h{h}={dh[h]['N']}" for h in HORIZONS])
        print(f"  {ds_label}: test origins per horizon: {counts_str} (stride=24, prefix-aligned=True)")

        for h in HORIZONS:
            d = dh[h]
            fe_mae_all = macro_metric_from_sums(d['fe_abs'].sum(axis=1), d['fe_cnt'].sum(axis=1)) # (324,)
            ie_mae_all = macro_metric_from_sums(d['ie_abs'].sum(axis=1), d['ie_cnt'].sum(axis=1)) # (36,)

            fe_phi = fe_mae_all[d['fe_phi_idx']].mean(axis=1)
            fe_mean = fe_mae_all[d['fe_mean_idx']].mean(axis=1)
            ie_phi = ie_mae_all[d['ie_phi_idx']]
            ie_mean = ie_mae_all[d['ie_mean_idx']]

            d_ie, d_fe, valid, pure_inv, tau = compute_tau_stats(ie_phi, ie_mean, fe_phi, fe_mean)
            for idx, (m, mode, rate, _, imp) in enumerate(d['pair_meta']):
                full_rows.append({
                    'dataset': ds_label,
                    'model': m,
                    'missing_mode': mode,
                    'missing_rate': rate,
                    'pred_len': h,
                    'impute_method': imp,
                    'n_origins': d['N'],
                    'ie_phi': float(ie_phi[idx]),
                    'ie_mean': float(ie_mean[idx]),
                    'delta_ie': float(d_ie[idx]),
                    'fe_phi_3seed': float(fe_phi[idx]),
                    'fe_mean_3seed': float(fe_mean[idx]),
                    'delta_fe': float(d_fe[idx]),
                    'is_valid': bool(valid[idx]),
                    'tau': float(tau[idx]) if valid[idx] else np.nan,
                    'is_pure_inversion': bool(pure_inv[idx]),
                })

    df_full = pd.DataFrame(full_rows)
    df_full.to_csv(RES_DIR / 'r13_full_origins_reconstructed_1152.csv', index=False)

    n_tot = len(df_full)
    n_val = int(df_full['is_valid'].sum())
    n_inv = int(df_full['is_pure_inversion'].sum())
    med_tau = float(df_full.loc[df_full['is_valid'], 'tau'].median())
    print(f"\nFull-Origin Subgrid (Spline+KNN vs Mean):")
    print(f"  Total pairs = {n_tot}, Valid(|dIE|>0.01) = {n_val}, Pure Inversions = {n_inv} ({n_inv/n_val*100:.2f}%), Median tau = {med_tau:.4f}")
    for ds_label in ['ETTh1', 'KDD-Beijing', 'SWaT', 'Weather']:
        sub = df_full[df_full['dataset'] == ds_label]
        v = int(sub['is_valid'].sum())
        iv = int(sub['is_pure_inversion'].sum())
        mt = float(sub.loc[sub['is_valid'], 'tau'].median())
        print(f"    {ds_label:11s}: total={len(sub)}, valid={v}, inv={iv} ({iv/v*100:.2f}%), med_tau={mt:.4f}")

    # ------------------------------------------------------------------
    # 2. Moving Block Bootstrap (B=2000, seed=42, L in [4, 8, 16])
    # ------------------------------------------------------------------
    print("\n=== 2. Running Moving Block Bootstrap (B=2000, seed=42, L=4,8,16) ===")
    B = 2000
    mbb_summary_rows = []

    for L in [4, 8, 16]:
        rng = np.random.default_rng(42)
        ds_boot = {}

        for ds_key, ds_label in DATASETS:
            dh = all_ds_data[ds_key]
            N_base = dh[24]['N']
            n_blocks = math.ceil(N_base / L)
            max_start = N_base - L
            # Draw (B, n_blocks) block starts in [0, max_start]
            starts = rng.integers(0, max_start + 1, size=(B, n_blocks))
            # Expand to (B, n_blocks * L) and trim to N_base
            offsets = np.arange(L)[None, None, :]
            drawn = (starts[:, :, None] + offsets).reshape(B, -1)[:, :N_base] # (B, N_base)

            # Collect across 4 horizons: each horizon produces (B, 72) for d_ie, d_fe, valid, pure_inv, tau
            ds_valid_list, ds_inv_list, ds_tau_list = [], [], []
            for h in HORIZONS:
                d = dh[h]
                Nh = d['N']
                # Count occurrences of each origin index 0..Nh-1 in drawn for each replicate
                w = np.zeros((B, Nh), dtype=np.float64)
                for b_i in range(B):
                    row_idx = drawn[b_i]
                    valid_pos = row_idx[row_idx < Nh]
                    w[b_i] = np.bincount(valid_pos, minlength=Nh)

                # Sum over origins: (B, Nh) @ (324, Nh, C) -> (B, 324, C)
                fe_abs_b = np.einsum('bn,mnc->bmc', w, d['fe_abs'], optimize=True)
                fe_cnt_b = np.einsum('bn,mnc->bmc', w, d['fe_cnt'], optimize=True)
                fe_mae_b = macro_metric_from_sums(fe_abs_b, fe_cnt_b) # (B, 324)

                ie_abs_b = np.einsum('bn,mnc->bmc', w, d['ie_abs'], optimize=True)
                ie_cnt_b = np.einsum('bn,mnc->bmc', w, d['ie_cnt'], optimize=True)
                ie_mae_b = macro_metric_from_sums(ie_abs_b, ie_cnt_b) # (B, 36)

                fe_phi = fe_mae_b[:, d['fe_phi_idx']].mean(axis=2)   # (B, 72)
                fe_mean = fe_mae_b[:, d['fe_mean_idx']].mean(axis=2) # (B, 72)
                ie_phi = ie_mae_b[:, d['ie_phi_idx']]                # (B, 72)
                ie_mean = ie_mae_b[:, d['ie_mean_idx']]              # (B, 72)

                _, _, v_b, iv_b, t_b = compute_tau_stats(ie_phi, ie_mean, fe_phi, fe_mean)
                ds_valid_list.append(v_b)
                ds_inv_list.append(iv_b)
                ds_tau_list.append(t_b)

            ds_valid = np.concatenate(ds_valid_list, axis=1) # (B, 288)
            ds_inv = np.concatenate(ds_inv_list, axis=1)     # (B, 288)
            ds_tau = np.concatenate(ds_tau_list, axis=1)     # (B, 288)
            ds_boot[ds_label] = (ds_valid, ds_inv, ds_tau)

            # Dataset-level bootstrap stats
            val_cnt_b = ds_valid.sum(axis=1)
            inv_cnt_b = ds_inv.sum(axis=1)
            inv_rate_b = np.where(val_cnt_b > 0, inv_cnt_b / val_cnt_b * 100.0, np.nan)
            med_tau_b = np.nanmedian(ds_tau, axis=1)

            sub_full = df_full[df_full['dataset'] == ds_label]
            v_f = int(sub_full['is_valid'].sum())
            iv_f = int(sub_full['is_pure_inversion'].sum())
            mt_f = float(sub_full.loc[sub_full['is_valid'], 'tau'].median())

            mbb_summary_rows.append({
                'block_len': L,
                'dataset': ds_label,
                'n_pairs': 288,
                'full_valid_count': v_f,
                'boot_valid_lo': int(np.percentile(val_cnt_b, 2.5)),
                'boot_valid_hi': int(np.percentile(val_cnt_b, 97.5)),
                'full_inv_count': iv_f,
                'full_inv_rate_pct': round(iv_f / v_f * 100.0, 2),
                'boot_inv_rate_lo_pct': round(float(np.nanpercentile(inv_rate_b, 2.5)), 2),
                'boot_inv_rate_hi_pct': round(float(np.nanpercentile(inv_rate_b, 97.5)), 2),
                'full_median_tau': round(mt_f, 4),
                'boot_median_tau_lo': round(float(np.nanpercentile(med_tau_b, 2.5)), 4),
                'boot_median_tau_hi': round(float(np.nanpercentile(med_tau_b, 97.5)), 4),
            })

        # Overall 4-dataset combined (1,152 pairs)
        all_valid = np.concatenate([ds_boot[d][0] for _, d in DATASETS], axis=1)
        all_inv = np.concatenate([ds_boot[d][1] for _, d in DATASETS], axis=1)
        all_tau = np.concatenate([ds_boot[d][2] for _, d in DATASETS], axis=1)

        val_cnt_all = all_valid.sum(axis=1)
        inv_cnt_all = all_inv.sum(axis=1)
        inv_rate_all = inv_cnt_all / val_cnt_all * 100.0
        med_tau_all = np.nanmedian(all_tau, axis=1)

        mbb_summary_rows.append({
            'block_len': L,
            'dataset': 'ALL_4_DATASETS',
            'n_pairs': 1152,
            'full_valid_count': n_val,
            'boot_valid_lo': int(np.percentile(val_cnt_all, 2.5)),
            'boot_valid_hi': int(np.percentile(val_cnt_all, 97.5)),
            'full_inv_count': n_inv,
            'full_inv_rate_pct': round(n_inv / n_val * 100.0, 2),
            'boot_inv_rate_lo_pct': round(float(np.nanpercentile(inv_rate_all, 2.5)), 2),
            'boot_inv_rate_hi_pct': round(float(np.nanpercentile(inv_rate_all, 97.5)), 2),
            'full_median_tau': round(med_tau, 4),
            'boot_median_tau_lo': round(float(np.nanpercentile(med_tau_all, 2.5)), 4),
            'boot_median_tau_hi': round(float(np.nanpercentile(med_tau_all, 97.5)), 4),
        })

    df_mbb = pd.DataFrame(mbb_summary_rows)
    df_mbb.to_csv(RES_DIR / 'r13_mbb_summary_L4_8_16.csv', index=False)
    print(df_mbb.to_string(index=False))
    print(f"\nR1.3 MBB complete in {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()
