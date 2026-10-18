"""Three-seed error-transmission and inversion analysis (Tables 2-9 and Table E.16).
Usage: python recompute_three_seeds.py
Primary estimand: average FE within configuration across three seeds BEFORE computing contrasts.
"""
from pathlib import Path
import argparse,json,itertools,hashlib
import numpy as np
import pandas as pd
KEY=['dataset','model','missing_mode','missing_rate','pred_len','impute_method']
BASE=KEY[:-1]
CLUSTER=[k for k in KEY if k!='pred_len']
SEEDS=[42,123,2026]
DATASETS=['ETTh1','KDD-Beijing','SWaT','Weather']
MODELS=['LSTM','DLinear','PatchTST']
METHODS=['mean','spline','knn','brits','saits']
NAMES={'mean':'Mean','spline':'Spline','knn':'KNN','brits':'BRITS','saits':'SAITS'}
def contrast(x,eps=.01,metric='mae',anchor='mean'):
    ie='ie_'+metric;fe='fe_'+metric
    a=x[x.impute_method.eq(anchor)][BASE+[ie,fe]].rename(columns={ie:'ie_anchor',fe:'fe_anchor'})
    t=x[x.impute_method.ne(anchor)].merge(a,on=BASE,validate='many_to_one')
    t['delta_ie']=t[ie]-t.ie_anchor;t['delta_fe']=t[fe]-t.fe_anchor
    t['valid']=t.delta_ie.abs()>eps
    t['tau']=(t.delta_fe/t.delta_ie).where(t.valid)
    t['pure']=t.valid&(t.delta_ie<0)&(t.delta_fe>0)
    t['global_inv']=t.valid&(t.delta_ie*t.delta_fe<0)
    t['relative_fe']=100*t.delta_fe/t.fe_anchor
    t['relative_ie']=100*t.delta_ie/t.ie_anchor
    t['eta']=(t.relative_fe/t.relative_ie).where(t.valid & t.ie_anchor.gt(0) & t.fe_anchor.gt(0))
    return t
def stats(t):
    v=t[t.valid];a=np.sort(v.tau.to_numpy());n=len(a)
    def trim(p):k=int(n*p);return float(a[k:n-k].mean())
    result=dict(total=len(t),valid_n=n,excluded=len(t)-n,pure_count=int(v.pure.sum()),pure_pct=100*v.pure.mean(),global_pct=100*v.global_inv.mean(),tau_median=v.tau.median(),tau_mean=v.tau.mean(),tau_q25=v.tau.quantile(.25),tau_q75=v.tau.quantile(.75),tau_iqr=v.tau.quantile(.75)-v.tau.quantile(.25),trim5=trim(.05),trim10=trim(.1),delta_ie_median=v.delta_ie.median(),delta_fe_median=v.delta_fe.median(),eta_median=v.eta.median(),relative_ie_median=v.relative_ie.median(),relative_fe_median=v.relative_fe.median())
    pure=v[v.pure];result.update(pure_deterioration_median=pure.relative_fe.median(),pure_deterioration_q25=pure.relative_fe.quantile(.25),pure_deterioration_q75=pure.relative_fe.quantile(.75))
    improved=int((t.valid & (t.delta_ie<0)).sum());result.update(ie_improved=improved,pure_given_ie_improved_pct=100*v.pure.sum()/improved if improved else np.nan)
    return result
def interval(t):
    c=t[t.valid].groupby(CLUSTER,sort=True).agg(k=('pure','sum'),n=('pure','size'))
    rng=np.random.RandomState(42);idx=rng.randint(len(c),size=(20000,len(c)))
    z=100*c.k.to_numpy()[idx].sum(1)/c.n.to_numpy()[idx].sum(1)
    lo,hi=np.quantile(z,[.025,.975]);return dict(clusters=len(c),ci_low=lo,ci_high=hi)
def save(out,name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig');return d
def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=here.parent/'01_raw_data'/'online_results_8640.csv');p.add_argument('--out',type=Path,default=here);args=p.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    raw=pd.read_csv(args.input);metrics=['ie_mae','ie_mse','fe_mae','fe_mse']
    assert len(raw)==8640 and not raw.duplicated(KEY+['pred_seed']).any()
    assert raw.status.eq('success').all() and np.isfinite(raw[metrics]).all().all()
    expected=set(itertools.product(DATASETS,MODELS,['MCAR','MAR_Block','MNAR'],[.1,.3,.5,.7],[24,48,96,192],METHODS))
    for s in SEEDS:assert set(raw[raw.pred_seed.eq(s)][KEY].itertuples(index=False,name=None))==expected
    assert raw.groupby(KEY)[['ie_mae','ie_mse']].nunique().max().max()==1
    point=raw.groupby(KEY,sort=True)[metrics].mean().reset_index()
    cells=raw.groupby(KEY)[metrics].agg(['mean','std']).reset_index();cells.columns=['_'.join(c).rstrip('_') for c in cells.columns]
    save(out,'configuration_mean_sd_2880',cells);save(out,'configuration_means_2880',point)
    t=contrast(point);save(out,'main_contrasts_2304',t)
    summaries=[dict(scope='Overall',**stats(t))]+[dict(scope=d,**stats(g)) for d,g in t.groupby('dataset')]
    save(out,'headline_and_dataset',summaries)
    perf=[]
    for (d,m),g in raw.groupby(['dataset','impute_method']):
        v=g.groupby('pred_seed')[['fe_mae','fe_mse']].mean()
        perf.append(dict(dataset=d,impute_method=m,mae_mean=v.fe_mae.mean(),mae_sd=v.fe_mae.std(ddof=1),mse_mean=v.fe_mse.mean(),mse_sd=v.fe_mse.std(ddof=1)))
    save(out,'method_performance',perf)
    st=[]
    for dim in ['dataset','model','impute_method','missing_mode','missing_rate','pred_len']:
        for cat,g in t.groupby(dim):st.append(dict(dimension=dim,category=cat,**stats(g),**interval(g)))
    save(out,'strata',st)
    save(out,'heatmap',[dict(model=m,impute_method=i,**stats(g)) for (m,i),g in t.groupby(['model','impute_method'])])
    save(out,'threshold_sensitivity',[dict(epsilon=e,metric='MAE',**stats(contrast(point,eps=e))) for e in [.001,.005,.01,.02,.05]]+[dict(epsilon=.01,metric='MSE',**stats(contrast(point,metric='mse')))])
    save(out,'anchor_sensitivity',[dict(anchor=a,**stats(contrast(point,anchor=a))) for a in ['mean','spline']])
    save(out,'forecast_tolerance',[dict(relative_fe_threshold_pct=e,numerator=int((t.pure & t.relative_fe.gt(e)).sum()),denominator=int(t.valid.sum()),pct=100*(t.pure & t.relative_fe.gt(e)).sum()/t.valid.sum()) for e in [0,.1,.5,1,2,5]])
    seed_tables={s:contrast(raw[raw.pred_seed.eq(s)]) for s in SEEDS}
    save(out,'seed_specific_statistics',[dict(seed=s,scope=d,**stats(g)) for s,x in seed_tables.items() for d,g in [('Overall',x)]+list(x.groupby('dataset'))])
    save(out,'seed_model_statistics',[dict(seed=s,dataset=d,model=m,**stats(g)) for s,x in seed_tables.items() for (d,m),g in x.groupby(['dataset','model'])])
    u=seed_tables[42][KEY+['valid']].copy()
    for s,x in seed_tables.items():u=u.merge(x[KEY+['delta_fe','pure']].rename(columns={'delta_fe':f'delta_fe_{s}','pure':f'pure_{s}'}),on=KEY,validate='one_to_one')
    signs=np.sign(u[[f'delta_fe_{s}' for s in SEEDS]].to_numpy());u['all_three_same_sign']=(signs==signs[:,[0]]).all(1)
    u['pairwise_disagreement_fraction']=np.mean([signs[:,i]!=signs[:,j] for i,j in [(0,1),(0,2),(1,2)]],axis=0)
    u['pure_seed_count']=u[[f'pure_{s}' for s in SEEDS]].sum(axis=1)
    save(out,'seed_units_2304',u)
    save(out,'seed_consistency',[dict(scope=d,units=len(g),all_three_same_count=int(g.all_three_same_sign.sum()),all_three_same_pct=100*g.all_three_same_sign.mean(),any_disagreement_pct=100*(~g.all_three_same_sign).mean(),mean_pairwise_disagreement_pct=100*g.pairwise_disagreement_fraction.mean(),pure_all_three=int(g.pure_seed_count.eq(3).sum()),pure_any_seed=int(g.pure_seed_count.gt(0).sum())) for d,g in [('Overall',u)]+list(u.groupby('dataset'))])
    rankings=[]
    for base,g in point.groupby(BASE):
        a=set(g.loc[g.ie_mae.eq(g.ie_mae.min()),'impute_method']);b=set(g.loc[g.fe_mae.eq(g.fe_mae.min()),'impute_method'])
        rankings.append(dict(zip(BASE,base),best_ie='|'.join(sorted(a)),best_fe='|'.join(sorted(b)),mismatch=not bool(a&b)))
    ranking=save(out,'method_ranking',rankings)
    save(out,'ranking_summary',ranking.groupby('dataset').agg(units=('mismatch','size'),mismatches=('mismatch','sum'),fraction=('mismatch','mean')).reset_index())
    save(out,'global_ie_medians',point.groupby('impute_method').ie_mae.median().reset_index())
    coords=save(out,'figure1_coordinates',point.groupby(['model','impute_method','missing_rate']).fe_mae.median().reset_index())
    comparisons=[('model','LSTM','DLinear'),('model','LSTM','PatchTST'),('model','PatchTST','DLinear'),('impute_method','spline','saits'),('impute_method','spline','knn'),('impute_method','brits','saits')]
    paired=[];v=t[t.valid]
    for col,a,b in comparisons:
        keys=[k for k in KEY if k!=col]
        z=v[v[col].eq(a)][keys+['pure']].merge(v[v[col].eq(b)][keys+['pure']],on=keys,suffixes=('_a','_b'),validate='one_to_one')
        z['difference']=z.pure_a.astype(int)-z.pure_b.astype(int)
        c=z.groupby([k for k in keys if k!='pred_len']).agg(k=('difference','sum'),n=('difference','size'));k=c.k.to_numpy();n=c.n.to_numpy()
        rng=np.random.RandomState(42);idx=rng.randint(len(c),size=(20000,len(c)));ci=np.quantile(100*k[idx].sum(1)/n[idx].sum(1),[.025,.975])
        rng=np.random.RandomState(42);null=rng.choice([-1,1],size=(20000,len(c)))@k
        pval=(1+(np.abs(null)>=abs(k.sum())).sum())/20001
        paired.append(dict(comparison=a+' vs '+b,n=len(z),clusters=len(c),a_pct=100*z.pure_a.mean(),b_pct=100*z.pure_b.mean(),difference_pp=100*z.difference.mean(),ci_low=ci[0],ci_high=ci[1],p=pval))
    order=np.argsort([r['p'] for r in paired]);running=0
    for i,j in enumerate(order):running=max(running,min(1,(len(paired)-i)*paired[j]['p']));paired[j]['holm_p']=running
    save(out,'paired_comparisons',paired)
    (out/'summary.json').write_text(json.dumps(dict(run_rows=8640,configurations=2880,primary=stats(t),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),estimand='configuration FE averaged over three seeds before contrasts',sd='sample SD, ddof=1',bootstrap_seed=42,bootstrap_replicates=20000),indent=2,default=lambda x:x.item() if hasattr(x,'item') else x),encoding='utf8')
    print(json.dumps(summaries,indent=2));print(pd.read_csv(out/'seed_consistency.csv').to_string(index=False))
if __name__=='__main__':main()
