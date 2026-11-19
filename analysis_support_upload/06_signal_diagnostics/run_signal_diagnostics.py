"""Read-only exploratory signal diagnostics on the frozen manuscript caches.

No imputer or forecaster is imported, fitted, or evaluated. Only test arrays,
masks, metadata and existing numeric FE records are read. Raw series and cache
arrays are never exported. This trial does not modify the manuscript or GitHub.
"""
from pathlib import Path
import argparse, ast, hashlib, json, itertools, platform, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy, sklearn

METHODS=['mean','spline','knn','brits','saits']
MODES=['MCAR','MAR_Block','MNAR'];RATES=[.1,.3,.5,.7];HORIZONS=[24,48,96,192]
CONFIG=['dataset','model','missing_mode','missing_rate','pred_len','impute_method']
SIGNALKEY=['dataset','missing_mode','missing_rate','pred_len','impute_method']
METRICS=['roughness_distortion','acf_distortion']
EPS=1e-6;MIN_PAIRS=8


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')


def preprocess(folder,dataset):
    filename={'ETTh1':'ETTh1.csv','KDD-Beijing':'kdd_beijing_raw.csv','SWaT':'normal.csv','Weather':'weather.csv'}[dataset]
    df=pd.read_csv(folder/filename)
    if dataset=='ETTh1':
        numeric=df[['OT','HUFL','HULL','MUFL','MULL','LUFL','LULL']].copy()
    else:
        excluded={'timestamp','datetime','time','date','normal/attack'}
        cols=[c for c in df if str(c).strip().lower() not in excluded and not str(c).strip().lower().startswith('unnamed:')]
        numeric=df[cols].apply(pd.to_numeric,errors='coerce') if dataset=='SWaT' else df[cols].select_dtypes(include=[np.number]).copy()
        numeric=numeric.replace([np.inf,-np.inf],np.nan)
    n=len(numeric);train_end=int(n*.7);val_end=train_end+int(n*.1)
    if dataset in ['KDD-Beijing','SWaT']:
        numeric=numeric.loc[:,numeric.iloc[:train_end].var(ddof=0,skipna=True)>1e-5]
    elif dataset!='ETTh1':numeric=numeric.loc[:,numeric.iloc[:train_end].notna().any()]
    dtype=np.float32 if dataset=='SWaT' else np.float64
    raw=numeric.to_numpy(dtype=dtype)
    scaler=StandardScaler().fit(raw[:train_end]);scaler_origin='recomputed using the recorded training-only protocol'
    saved=folder/'results/preprocessing.json'
    if saved.is_file():
        info=json.loads(saved.read_text(encoding='utf8'))
        assert info['value_cols']==list(numeric.columns),f'{dataset}: saved feature order mismatch'
        assert info['split_info']==dict(train_end=train_end,val_end=val_end,total_len=n)
        scaler.mean_=np.asarray(info['scaler_mean'],dtype=float)
        scaler.scale_=np.asarray(info['scaler_scale'],dtype=float)
        scaler_origin='recorded preprocessing.json parameters'
    clean=scaler.transform(raw).astype(dtype,copy=False)
    return clean,dict(train_end=train_end,val_end=val_end,total_len=n),dict(data_file=str(folder/filename),data_sha256=hashlib.sha256((folder/filename).read_bytes()).hexdigest(),shape=clean.shape,features=list(numeric.columns),scaler_source=scaler_origin,clean_dtype=str(clean.dtype))


def cache_paths(cache_dir,mode,rate,method):
    pct=round(rate*100)
    matches=[p for p in cache_dir.glob('*.npz') if ((f'_{mode}_{method}_rate{pct}.' in p.name or p.name.startswith(f'{mode}_{method}_rate{pct}_')))]
    if method=='spline':matches=[p for p in matches if 'v3' in p.name.lower()]
    assert len(matches)==1,(str(cache_dir),mode,rate,method,[p.name for p in matches])
    # Some authors' archives contain an alternate copy in cy/. Use only an
    # identical named file after an explicit read failure and audit the choice.
    return matches+[p for p in cache_dir.rglob(matches[0].name) if p!=matches[0]]


def read_cache(paths,mode,rate,method,split,channels):
    errors=[]
    for path in paths:
        try:
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item()))
                settings=meta.get('settings',meta.get('identity',{}))
                assert settings['mode']==mode and settings['method']==method
                assert abs(settings['rate']-rate)<1e-12
                assert settings['seq_len']==96 and settings['split_info']==split
                assert settings['missing_seed']==42 and settings['imputation_seed']==42
                if method=='spline':assert 'v3' in str(meta.get('imputation_variant',settings.get('imputation_variant',''))).lower()
                starts=z['test_starts'];x=z['test_x']
                mask=z['missing_mask'] if 'missing_mask' in z.files else None
            assert x.shape==(len(starts),96,channels) and np.isfinite(x).all()
            expected=np.arange(max(0,split['val_end']-96),split['total_len']-96-24+1,24)
            assert np.array_equal(starts,expected)
            return x,starts,mask,meta,dict(path=str(path),file_bytes=path.stat().st_size,test_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),skipped_unreadable_candidates=errors)
        except Exception as e:errors.append(f'{path}: {type(e).__name__}: {e}')
    raise RuntimeError('\n'.join(errors))


def corr_lag(window,pairs):
    """Pearson correlation between genuine adjacent endpoints; no gap joining."""
    n=pairs.sum(axis=1)
    a=window[:,:-1,:].astype(np.float64);b=window[:,1:,:].astype(np.float64)
    a=np.where(pairs,a,0);b=np.where(pairs,b,0)
    count=np.maximum(n,1)
    a=a-(a.sum(axis=1)/count)[:,None,:];b=b-(b.sum(axis=1)/count)[:,None,:]
    a=np.where(pairs,a,0);b=np.where(pairs,b,0)
    va=(a*a).sum(axis=1);vb=(b*b).sum(axis=1)
    usable=(n>=MIN_PAIRS)&(va/count>1e-16)&(vb/count>1e-16)
    rho=np.divide((a*b).sum(axis=1),np.sqrt(va*vb),out=np.full(n.shape,np.nan),where=usable)
    return np.clip(rho,-1,1),usable


def macro(values,support,rows):
    use=support[rows]&np.isfinite(values[rows]);count=use.sum(axis=0)
    sums=np.where(use,values[rows],0).sum(axis=0)
    keep=count>0
    return float(np.mean(sums[keep]/count[keep])) if keep.any() else np.nan,int(keep.sum()),int(use.sum())


def extract(args,out):
    raw=pd.read_csv(args.results)
    expected=raw.groupby(CONFIG,sort=True)[['ie_mae','ie_mse','fe_mae','fe_mse']].mean().reset_index()
    setups=[('ETTh1','ETT_online_42','combine_npz'),('KDD-Beijing','KDD_online','KDD_NPZ_file_online'),('SWaT','SWaT_online','SWaT_NPZ_file_online'),('Weather','Weather_online','Weather_NPZ_file_online')]
    audit=dict(stage='extraction',training_performed=False,parameters=dict(epsilon=EPS,min_adjacent_pairs=MIN_PAIRS,acf_min_std=1e-8,window_length=96,aggregation='equal windows within feature; equal valid features; common support across all five imputers'),python=platform.python_version(),packages=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),datasets=[],caches=[],issues=[])
    rows=[];checks=[]
    for dataset,foldername,cachedir in setups:
        folder=args.root/foldername;clean,split,info=preprocess(folder,dataset);audit['datasets'].append(dict(dataset=dataset,**info))
        print('DATASET',dataset,clean.shape,flush=True)
        for mode,rate in itertools.product(MODES,RATES):
            loaded={m:read_cache(cache_paths(folder/cachedir,mode,rate,m),mode,rate,m,split,clean.shape[1]) for m in METHODS}
            canonical=loaded['spline'][2]
            assert canonical is not None and canonical.shape==clean.shape and canonical.dtype==np.bool_
            starts=loaded['mean'][1];pos=starts[:,None]+np.arange(96)[None,:]
            truth=clean[pos];mask=canonical[pos];finite=np.isfinite(truth)
            testpos=pos>=split['val_end'];cell=mask&finite&testpos[:,:,None]
            pairs=finite[:,:-1]&finite[:,1:]&testpos[:,:-1,None]&testpos[:,1:,None]
            changed_pairs=pairs&(cell[:,:-1]|cell[:,1:]);count=changed_pairs.sum(axis=1)
            truth_diff=np.abs(np.diff(truth.astype(np.float64),axis=1))
            rough=np.where(changed_pairs,truth_diff,0).sum(axis=1)/np.maximum(count,1)
            rough_support=count>=MIN_PAIRS
            rho_true,acf_support=corr_lag(truth,pairs)
            signed={};rho={};dacf={}
            for m,(x,s,stored,meta,ca) in loaded.items():
                assert np.array_equal(s,starts)
                if stored is not None:assert np.array_equal(stored,canonical),(dataset,mode,rate,m,'mask mismatch')
                identity=meta.get('identity',{})
                if 'mask_sha256' in identity:assert hashlib.sha256(canonical.tobytes()).hexdigest()==identity['mask_sha256']
                observed=finite&~mask
                assert np.allclose(x[observed],truth[observed],rtol=3e-6,atol=3e-6),(dataset,m,'observed cells changed')
                hatrough=np.where(changed_pairs,np.abs(np.diff(x.astype(np.float64),axis=1)),0).sum(axis=1)/np.maximum(count,1)
                signed[m]={epsilon:(hatrough-rough)/(rough+epsilon) for epsilon in [1e-8,EPS,1e-4]}
                rho[m],valid=corr_lag(x,pairs);acf_support&=valid
                audit['caches'].append(dict(dataset=dataset,mode=mode,rate=rate,method=m,**ca))
            for m in METHODS:dacf[m]=np.abs(rho[m]-rho_true)
            for h in HORIZONS:
                kept=starts+96+h<=split['total_len']
                for m in METHODS:
                    x=loaded[m][0];error=(x-truth).astype(np.float64)
                    selected=cell[kept];err=error[kept]
                    n=selected.sum(axis=(0,1));keep=n>0
                    mae=np.mean(np.abs(np.where(selected,err,0)).sum(axis=(0,1))[keep]/n[keep])
                    mse=np.mean(np.square(np.where(selected,err,0)).sum(axis=(0,1))[keep]/n[keep])
                    target=expected[(expected.dataset==dataset)&(expected.missing_mode==mode)&(expected.missing_rate==rate)&(expected.impute_method==m)&(expected.pred_len==h)]
                    assert len(target)==3
                    d_mae=float(np.max(abs(target.ie_mae-mae)));d_mse=float(np.max(abs(target.ie_mse-mse)))
                    tol=5.1e-7 if dataset=='ETTh1' else 2e-7
                    passed=d_mae<=tol and d_mse<=tol
                    checks.append(dict(dataset=dataset,missing_mode=mode,missing_rate=rate,pred_len=h,impute_method=m,ie_mae_recomputed=mae,ie_mae_recorded=target.ie_mae.iloc[0],ie_mae_abs_difference=d_mae,ie_mse_abs_difference=d_mse,passed=passed,tolerance=tol))
                    if not passed:
                        audit['issues'].append(checks[-1]);dump(out/'signal_diagnostic_audit.json',audit)
                        raise ValueError(f'IE mismatch: {checks[-1]}')
                    dr,features,nr=macro(abs(signed[m][EPS]),rough_support,kept)
                    sr,_,_=macro(signed[m][EPS],rough_support,kept)
                    da,af,na=macro(dacf[m],acf_support,kept)
                    drflat,_,_=macro(abs(signed[m][EPS]),rough_support&(rough>EPS),kept)
                    dr8,_,_=macro(abs(signed[m][1e-8]),rough_support,kept)
                    dr4,_,_=macro(abs(signed[m][1e-4]),rough_support,kept)
                    rows.append(dict(dataset=dataset,missing_mode=mode,missing_rate=rate,pred_len=h,impute_method=m,roughness_distortion=dr,signed_roughness=sr,roughness_nonflat=drflat,roughness_eps1e8=dr8,roughness_eps1e4=dr4,acf_distortion=da,roughness_features=features,acf_features=af,roughness_window_features=nr,acf_window_features=na,total_window_features=int(kept.sum()*clean.shape[1]),roughness_flat_window_features=int((rough_support[kept]&(rough[kept]<=EPS)).sum()),test_windows=int(kept.sum())))
            print('EXTRACTED',dataset,mode,rate,'/ 5 methods',flush=True)
            pd.DataFrame(rows).to_csv(out/'signal_metrics_960.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(checks).to_csv(out/'signal_ie_reproduction.csv',index=False,encoding='utf-8-sig')
        dump(out/'signal_diagnostic_audit.json',audit)
    audit.update(stage='extraction_complete',expected_upstream_cache_count=240,cache_count=len(audit['caches']),signal_rows=len(rows),ie_checks_passed=len(checks),all_ie_checks_pass=True)
    dump(out/'signal_diagnostic_audit.json',audit)
    return pd.DataFrame(rows)


def group_boot_indices(frame,b,seed=42):
    groups=[]
    for _,data in frame.groupby('dataset',sort=True):
        blocks=[g.index.to_numpy() for _,g in data.groupby(['missing_mode','missing_rate'],sort=True)]
        groups.append(blocks)
    rng=np.random.RandomState(seed)
    for _ in range(b):
        yield np.concatenate([blocks[i] for blocks in groups for i in rng.randint(len(blocks),size=len(blocks))])


def associations(c,out,b):
    allrows=[]
    measures=['delta_roughness_distortion','delta_acf_distortion','delta_roughness_nonflat','delta_roughness_eps1e8','delta_roughness_eps1e4']
    for scope,g0 in [('Overall',c)]+list(c.groupby('model',sort=True)):
        for measure in measures:
            g=g0[g0.valid & np.isfinite(g0[measure])].reset_index(drop=True)
            estimate=float(spearmanr(g[measure],g.delta_fe).statistic)
            draws=[]
            for indices in group_boot_indices(g,b):
                a=spearmanr(g.loc[indices,measure],g.loc[indices,'delta_fe']).statistic
                if np.isfinite(a):draws.append(float(a))
            lo,hi=np.quantile(draws,[.025,.975])
            allrows.append(dict(scope=scope,metric=measure,n=len(g),spearman_rho=estimate,ci_low=lo,ci_high=hi,bootstrap_replicates=b))
        print('CORRELATIONS',scope,flush=True)
    pd.DataFrame(allrows).to_csv(out/'signal_associations.csv',index=False,encoding='utf-8-sig')
    strata=[]
    for dim in ['dataset','missing_mode','impute_method']:
        for name,g in c[c.valid].groupby(dim,sort=True):
            for metric in measures[:2]:
                z=g[np.isfinite(g[metric])]
                strata.append(dict(dimension=dim,category=name,metric=metric,n=len(z),rho=spearmanr(z[metric],z.delta_fe).statistic))
    pd.DataFrame(strata).to_csv(out/'signal_stratified_correlations.csv',index=False,encoding='utf-8-sig')
    influence=[]
    for omitted in sorted(c.dataset.unique()):
        g=c[c.valid & c.dataset.ne(omitted)]
        for metric in measures[:2]:
            z=g[np.isfinite(g[metric])]
            influence.append(dict(excluded_dataset=omitted,metric=metric,n=len(z),rho=spearmanr(z[metric],z.delta_fe).statistic))
    pd.DataFrame(influence).to_csv(out/'signal_dataset_influence.csv',index=False,encoding='utf-8-sig')


def fit_incremental(c,out):
    results=[];predictions=[]
    cat=['dataset','model','impute_method','missing_mode','missing_rate','pred_len']
    for scope,g0 in [('Overall',c)]+list(c.groupby('model',sort=True)):
        for metric in ['delta_roughness_distortion','delta_acf_distortion']:
            g=g0[g0.valid & np.isfinite(g0[metric])].reset_index(drop=True)
            groups=g.groupby(['dataset','missing_mode','missing_rate'],sort=True).ngroup().to_numpy()
            y=g.delta_fe.to_numpy();base=np.empty(len(g));aug=np.empty(len(g));folds=np.empty(len(g),int)
            for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(g,y,groups)):
                for featurelist,dest in [(['delta_ie'],base),(['delta_ie',metric],aug)]:
                    transform=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat),('numeric',StandardScaler(),featurelist)])
                    model=make_pipeline(transform,LinearRegression())
                    model.fit(g.iloc[train],y[train]);dest[test]=model.predict(g.iloc[test])
                folds[test]=fold
            r0=r2_score(y,base);r1=r2_score(y,aug);m0=mean_squared_error(y,base);m1=mean_squared_error(y,aug)
            results.append(dict(scope=scope,metric=metric,n=len(g),groups=len(np.unique(groups)),baseline_oof_r2=r0,augmented_oof_r2=r1,oof_r2_gain=r1-r0,baseline_oof_mse=m0,augmented_oof_mse=m1,oof_mse_reduction_pct=100*(m0-m1)/m0))
            z=g[CONFIG].copy();z['scope']=scope;z['metric']=metric;z['fold']=folds;z['delta_fe']=y;z['baseline_prediction']=base;z['augmented_prediction']=aug
            predictions.append(z)
    pd.DataFrame(results).to_csv(out/'signal_incremental_fit.csv',index=False,encoding='utf-8-sig')
    pd.concat(predictions).to_csv(out/'signal_oof_predictions.csv',index=False,encoding='utf-8-sig')
    improved=c[c.delta_ie<-.01];summ=[]
    for scope,g in [('Overall',improved)]+list(improved.groupby('model',sort=True)):
        for sign,sub in [('worse',g[g.delta_fe>0]),('better',g[g.delta_fe<0]),('exact_zero',g[g.delta_fe==0])]:
            for metric in ['delta_roughness_distortion','delta_acf_distortion']:
                a=sub[metric].dropna();summ.append(dict(scope=scope,forecast_outcome=sign,metric=metric,n=len(a),median=a.median(),q25=a.quantile(.25),q75=a.quantile(.75)))
    pd.DataFrame(summ).to_csv(out/'signal_inversion_groups.csv',index=False,encoding='utf-8-sig')


def plot(c,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    models=['LSTM','DLinear','PatchTST'];datasets=sorted(c.dataset.unique());colors=dict(zip(datasets,['#0072B2','#D55E00','#009E73','#CC79A7']))
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharey=True)
    for row,metric in enumerate(['delta_roughness_distortion','delta_acf_distortion']):
        for col,model in enumerate(models):
            g=c[c.valid & c.model.eq(model) & np.isfinite(c[metric])].copy()
            g['percentile']=100*(rankdata(g[metric],method='average')-.5)/len(g)
            ax=axs[row,col]
            for dataset,a in g.groupby('dataset'):
                ax.scatter(a.percentile,a.delta_fe,s=11,alpha=.55,color=colors[dataset],label=dataset)
            ax.axhline(0,c='black',lw=.7);ax.set_title(model);ax.grid(alpha=.15)
            ax.set_xlabel('Percentile of '+('roughness distortion change' if row==0 else 'ACF distortion change'))
            if col==0:ax.set_ylabel('Forecasting MAE change vs Mean')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.suptitle('Exploratory associations on fixed archived inputs',y=.99)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/'signal_diagnostics.png',dpi=200);plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000);parser.add_argument('--resume-metrics',action='store_true')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(out/'signal_metrics_960.csv') if args.resume_metrics else extract(args,out)
    assert len(metrics)==960 and not metrics.duplicated(SIGNALKEY).any()
    raw=pd.read_csv(args.results)
    point=raw.groupby(CONFIG,sort=True)[['ie_mae','fe_mae']].mean().reset_index()
    point=point.merge(metrics,on=SIGNALKEY,validate='many_to_one')
    columns=['ie_mae','fe_mae','roughness_distortion','acf_distortion','signed_roughness','roughness_nonflat','roughness_eps1e8','roughness_eps1e4']
    basex=CONFIG[:-1]
    anchor=point[point.impute_method.eq('mean')][basex+columns].rename(columns={col:'anchor_'+col for col in columns})
    c=point[point.impute_method.ne('mean')].merge(anchor,on=basex,validate='many_to_one')
    for col in columns:c['delta_'+({'ie_mae':'ie','fe_mae':'fe'}.get(col,col))]=c[col]-c['anchor_'+col]
    c['valid']=c.delta_ie.abs()>.01;c['pure']=c.valid&(c.delta_ie<0)&(c.delta_fe>0)
    assert len(c)==2304 and c.valid.sum()==2127 and c.pure.sum()==415
    c.to_csv(out/'signal_contrasts_2304.csv',index=False,encoding='utf-8-sig')
    print('PAIRED',len(c),'VALID',int(c.valid.sum()),'PURE',int(c.pure.sum()),flush=True)
    associations(c,out,args.bootstrap);fit_incremental(c,out);plot(c,out)
    audit=json.loads((out/'signal_diagnostic_audit.json').read_text(encoding='utf8'))
    audit.update(stage='complete',bootstrap_replicates=args.bootstrap,bootstrap_seed=42,grouping='dataset x missing_mode x missing_rate; stratified by dataset for bootstrap',folds=5,training_performed=False,manuscript_modified=False,github_modified=False)
    dump(out/'signal_diagnostic_audit.json',audit)
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':main()
