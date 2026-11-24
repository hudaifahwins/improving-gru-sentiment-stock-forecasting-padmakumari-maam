#!/usr/bin/env python3
"""Crazy-but-disciplined leak-free research pass inspired by external resources.

No external repos cloned; reuses existing src/ modules only.
Outputs: results/crazy_research/crazy_research_summary.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import DataConfig, ModelConfig, resolve_sentiment_columns
from src.horizon_experiment import HConfig, run_one

DATA_FILES = {
    "AAPL": "data/prices_sentiment_AAPL.csv",
    "005930.KS": "data/prices_sentiment_005930.KS.csv",
}

def load_merged(ticker):
    df=pd.read_csv(DATA_FILES[ticker])
    if 'Date' in df.columns: df['Date']=pd.to_datetime(df['Date'])
    return df

def dm_hac(loss_a, loss_b):
    # loss differential d = loss_a - loss_b; positive means b better
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n=len(d)
    if n < 3: return float('nan'), float('nan')
    mean=d.mean()
    # Newey-West Bartlett lag
    lag=int(math.floor(1.3*(n**0.25)))
    gamma0=np.var(d, ddof=1) if n>1 else 0.0
    var=gamma0
    for k in range(1, min(lag,n-1)+1):
        w=1-k/(lag+1)
        cov=np.cov(d[k:], d[:-k], ddof=1)[0,1]
        var += 2*w*cov
    var = var/n if n else float('nan')
    if var <= 0 or not np.isfinite(var):
        return float('nan'), float('nan')
    stat=mean/math.sqrt(var)
    # normal approx
    from math import erf, sqrt
    p=2*(1-0.5*(1+erf(abs(stat)/sqrt(2))))
    return float(stat), float(p)

def run_cfg(df,ticker,horizon,features,loss,seed,out_dir):
    sent_cols=()
    if 'sentiment' in features:
        s=resolve_sentiment_columns(df)
        sent_cols=tuple(c.strip() for c in s if c.strip()) if s else ()
    data_cfg=DataConfig(sentiment_cols=sent_cols)
    cfg=HConfig(horizon=horizon, loss=loss, baseline='ridge', deadzone_eps=0.005,
                features=features, stride='nonoverlap', vol_deadzone=0.0, seed=seed,
                out_dir=Path(out_dir))
    model_cfg=ModelConfig(seed=seed, model='gru', hidden_size=50, num_layers=1,
                          epochs=200, lr=1e-3, patience=15, batch_size=64, device='auto')
    payload=run_one(df, data_cfg=data_cfg, cfg=cfg, model_cfg=model_cfg,
                    source_label=f'crazy_research:{ticker}')
    r=payload['results']['gru']; rm=r['return_metrics']
    w=payload['results']['windows']
    y=np.array(w['y'],float); pred=np.array(w['pred'],float)
    mom=np.array([np.nan if v is None else float(v) for v in w['momentum_sign']],float)
    valid=~np.isnan(mom)
    base_up=float((y>0).mean()); base_down=float((y<0).mean()); base_maj=max(base_up,base_down)
    d_acc=float(np.mean(np.sign(pred)==np.sign(y)))
    mom_acc=float(np.mean(mom[valid]==np.sign(y[valid]))) if valid.sum()>0 else float('nan')
    collapsed=bool(len(np.unique(np.sign(pred)))==1)
    # costs
    pos=np.sign(pred); pos[pos==0]=0
    turnover=float(np.mean(np.abs(np.diff(pos)))) if len(pos)>1 else 0.0
    cost=0.0005 # 5bps per side
    strat_ret=pos[:-1]*y[1:] - cost*np.abs(np.diff(pos))
    net=float(np.sum(strat_ret))
    # DM vs base: loss 1{wrong} vs 1{base wrong} using majority sign
    base_sign = 1.0 if base_up>=base_down else -1.0
    l_model=(np.sign(pred)!=np.sign(y)).astype(float)
    l_base=(base_sign!=np.sign(y)).astype(float)
    dm_stat, dm_p = dm_hac(l_model, l_base)
    return {
        'ticker':ticker,'horizon':horizon,'features':features,'loss':loss,'seed':seed,
        'dir_acc':float(rm['directional_accuracy']),
        'dir_acc_strict':float(rm.get('directional_accuracy_strict',float('nan'))),
        'excess_over_base':float(d_acc-base_maj),
        'base_majority':base_maj,'base_up':base_up,'n_windows':int(len(y)),
        'mom_dir_acc':mom_acc,'collapsed':collapsed,
        'turnover':turnover,'net_ret_5bps':net,
        'dm_vs_base_stat':dm_stat,'dm_vs_base_p':dm_p,
        'ridge_dir_acc': (float(payload['results']['ridge']['return_metrics']['directional_accuracy']) if 'ridge' in payload['results'] else None),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='results/crazy_research')
    ap.add_argument('--seeds', default='42,123,2026')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    tickers=['AAPL','005930.KS']; horizons=[3,5]
    configs=[
        ('price+tech+ohlcv_volnorm','mse','no_sent_mse'),
        ('price+tech+ohlcv_volnorm','dir_margin','no_sent_dir_margin'),
        ('price+tech+ohlcv_volnorm+sentiment','mse','sent_mse'),
    ]
    rows=[]
    for t in tickers:
        df=load_merged(t)
        for h in horizons:
            for seed in seeds:
                for feat,loss,label in configs:
                    rec=run_cfg(df,t,h,feat,loss,seed,out)
                    rec['label']=label
                    rows.append(rec)
                    print(json.dumps({'ticker':t,'h':h,'seed':seed,'label':label,'dir_acc':rec['dir_acc'],'excess':rec['excess_over_base'],'collapsed':rec['collapsed'],'n':rec['n_windows'],'dm_p':rec['dm_vs_base_p'],'net':rec['net_ret_5bps']}))
    # summary
    groups={}
    for r in rows:
        k=(r['ticker'],r['horizon'],r['label'])
        groups.setdefault(k,[]).append(r)
    summary=[]
    for k,rs in groups.items():
        da=[x['dir_acc'] for x in rs]; ex=[x['excess_over_base'] for x in rs]; net=[x['net_ret_5bps'] for x in rs]
        summary.append({
            'ticker':k[0],'horizon':k[1],'label':k[2],'n_seeds':len(rs),'n_windows':rs[0]['n_windows'],'base_majority':rs[0]['base_majority'],
            'dir_acc_mean':float(np.mean(da)),'dir_acc_std':float(np.std(da,ddof=1)) if len(da)>1 else 0.0,
            'excess_mean':float(np.mean(ex)),'excess_std':float(np.std(ex,ddof=1)) if len(ex)>1 else 0.0,
            'collapse_count':int(sum(1 for x in rs if x['collapsed'])),
            'net_ret_5bps_mean':float(np.mean(net)),
            'dm_vs_base_p_mean':float(np.mean([x['dm_vs_base_p'] for x in rs if np.isfinite(x['dm_vs_base_p'])])) if any(np.isfinite(x['dm_vs_base_p']) for x in rs) else None,
        })
    outj={'source':'scripts/run_crazy_research.py','seeds':seeds,'rows':rows,'summary':summary,
          'external_resources':{'arxiv_index':'data/external_resources/arxiv_index.json','github_repos':'data/external_resources/github_repos.json','snapshot':'data/external_resources/SNAPSHOT.md'}}
    (out/'crazy_research_summary.json').write_text(json.dumps(outj,indent=2))
    print('SUMMARY_ROWS',len(summary))

if __name__=='__main__':
    main()
