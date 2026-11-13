#!/usr/bin/env python3
"""Executable class-aware locally adaptive Parzen classifier experiment.
Requires diabetes_risk_prediction_dataset.csv beside this file for actual study results.
No test-set use: new method evaluated on validation only; outer independent test remains future work.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix, recall_score

FEATURES={'3D':['Age','HbA1c','Fasting_Blood_Sugar'], '6D':['Age','BMI','HbA1c','Fasting_Blood_Sugar','Blood_Pressure_Systolic','Sleep_Hours']}
EXCLUDED=['Patient_ID','Diabetes_Risk_Score','AI_Health_Recommendation','Doctor_Consultation_Needed','Diabetes_Risk']

def local_bandwidths(z,y,k=15,base=.45,alpha=.5,class_aware=False):
    """kNN distance within each class; geometric-mean normalized, clipped.
    class_aware=False pools reference kNN distances across classes for scale;
    True normalizes each class independently and uses inverse-sqrt class-count
    multiplier relative to majority (capped). Both are explicit design choices.
    """
    n=len(y); radii=np.empty(n); counts={}
    for c in (0,1):
        ids=np.flatnonzero(y==c);counts[c]=len(ids)
        if len(ids)<k+1: raise ValueError(f'class {c} has fewer than k+1 observations')
        nbrs=NearestNeighbors(n_neighbors=k+1,algorithm='auto').fit(z[ids])
        radii[ids]=np.maximum(nbrs.kneighbors(z[ids],return_distance=True)[0][:,-1],1e-6)
    out=np.empty(n)
    global_ref=np.exp(np.mean(np.log(radii)))
    for c in (0,1):
        ids=np.flatnonzero(y==c)
        ref=np.exp(np.mean(np.log(radii[ids]))) if class_aware else global_ref
        multiplier=np.sqrt(max(counts.values())/counts[c]) if class_aware else 1.
        multiplier=min(multiplier,1.75)
        out[ids]=np.clip(base*(radii[ids]/ref)**alpha*multiplier,.12,1.8)
    return out

def variable_logdensity(train,query,bw,chunk=125):
    """Properly normalized Gaussian mixture with one bandwidth per training observation."""
    n,d=train.shape
    norm=-.5*d*np.log(2*np.pi)-d*np.log(bw)-np.log(n)
    out=[]
    for a in range(0,len(query),chunk):
        dsq=cdist(query[a:a+chunk],train,metric='sqeuclidean')
        out.append(logsumexp(-.5*dsq/(bw[None,:]**2)+norm[None,:],axis=1))
    return np.concatenate(out)

def predict(z,y,q,mode,base,k=15,alpha=.5):
    if mode=='fixed': bw=np.full(len(y),base)
    else: bw=local_bandwidths(z,y,k,base,alpha,class_aware=(mode=='class_aware'))
    logs=[variable_logdensity(z[y==c],q,bw[y==c]) for c in (0,1)]
    prior=float(np.mean(y)); a=logs[0]+np.log1p(-prior); b=logs[1]+np.log(prior)
    return np.exp(b-logsumexp(np.stack((a,b),axis=1),axis=1))

def metrics(y,p):
    pred=(p>=.5).astype(int);cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(auc=float(roc_auc_score(y,p)),ap_high=float(average_precision_score(y,p)),ap_non_high=float(average_precision_score(1-y,1-p)),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,np.clip(p,1e-12,1-1e-12))),specificity_050=float(recall_score(y,pred,pos_label=0)),recall_high_050=float(recall_score(y,pred,pos_label=1)),confusion_matrix=cm.tolist())


def save_adaptive_kde_figures(out, folder):
    # Figures 7 and 8 are constructed directly from the complete validation metrics.
    methods = ['fixed', 'adaptive', 'class_aware']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), layout='constrained')
    for ax, dim in zip(axs, ['3D', '6D']):
        vals = [out['methods'][dim][m]['auc'] for m in methods]
        bars = ax.bar(range(3), vals, color=['#64748b', '#0f766e', '#c48b32'])
        ax.set_ylim(.80, .90)
        ax.set_xticks(range(3), ['Fixed', 'Adaptive', 'Class-aware'])
        ax.set_ylabel('ROC-AUC')
        ax.set_title(dim+' feature set')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+.002, f'{v:.3f}', ha='center', fontsize=9)
    fig.savefig(folder/'adaptive_kde_auc_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), layout='constrained')
    for ax, dim in zip(axs, ['3D', '6D']):
        vals = [out['methods'][dim][m]['brier'] for m in methods]
        bars = ax.bar(range(3), vals, color=['#64748b', '#0f766e', '#c48b32'])
        ax.set_ylim(.10, .15)
        ax.set_xticks(range(3), ['Fixed', 'Adaptive', 'Class-aware'])
        ax.set_ylabel('Brier score (lower is better)')
        ax.set_title(dim+' feature set')
        for bar,v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+.0007, f'{v:.3f}', ha='center', fontsize=9)
    fig.savefig(folder/'adaptive_kde_brier_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',default='diabetes_risk_prediction_dataset.csv');ap.add_argument('--fast',action='store_true',help='small smoke test; not reportable as final study');ap.add_argument('--synthetic-smoke-test',action='store_true',help='check implementation only; not project results');args=ap.parse_args()
    t=time.time();rng=np.random.default_rng(42)
    if args.synthetic_smoke_test:
        # Synthetic code-only sanity check; NEVER substitute for diabetes dataset.
        y=np.r_[np.zeros(90,dtype=int),np.ones(210,dtype=int)]
        z=rng.normal(size=(300,3))+y[:,None]*.55
        q=rng.normal(size=(90,3)); yv=(q[:,0]+q[:,1]>.3).astype(int)
        sets={'SMOKE':(z,y,q,yv)}
    else:
        p=Path(args.csv)
        if not p.exists(): raise SystemExit(f'Missing dataset: {p.resolve()} — upload the CSV to run genuine results.')
        df=pd.read_csv(p);y=df['Diabetes_Risk'].eq('High').astype(int).to_numpy()
        x=df.drop(columns=EXCLUDED)
        xt,rest,yt,yr=train_test_split(x,y,test_size=.4,stratify=y,random_state=42)
        xv,_unused,yv,_unused_y=train_test_split(rest,yr,test_size=.5,stratify=yr,random_state=42)
        n=1200 if args.fast else 8000
        xs,_,ys,_=train_test_split(xt,yt,train_size=n,stratify=yt,random_state=42)
        sets={}
        for name,feat in FEATURES.items():
            imp=SimpleImputer(strategy='median');sc=StandardScaler()
            z=sc.fit_transform(imp.fit_transform(xs[feat]));q=sc.transform(imp.transform(xv[feat]))
            # Validation subset is fixed, stratified; preserve original test untouched.
            if len(yv)> (300 if args.fast else 2500):
                _,q,_,yval=train_test_split(q,yv,test_size=(300 if args.fast else 2500),stratify=yv,random_state=42)
            else:yval=yv
            sets[name]=(z,ys,q,yval)
    out={'status':'synthetic_smoke_only' if args.synthetic_smoke_test else ('fast_diagnostic' if args.fast else 'validation_experiment'), 'test_set_used':False,'methods':{},'notes':'Validation is also used to choose parameters; performance is exploratory, not independent generalization evidence.'}
    for dim,(z,y,q,yv) in sets.items():
        out['methods'][dim]={}
        for mode in ('fixed','adaptive','class_aware'):
            # Same parameters across methods for interpretable first comparison; future inner-validation tuning required.
            prob=predict(z,y,q,mode,base=.45,k=15,alpha=.5)
            out['methods'][dim][mode]=metrics(yv,prob)
            print(dim,mode,'AUC',round(out['methods'][dim][mode]['auc'],4),'Brier',round(out['methods'][dim][mode]['brier'],4),flush=True)
    out['elapsed_seconds']=round(time.time()-t,2)
    path=Path(__file__).resolve().parent/('adaptive_kde_smoke_results.json' if args.synthetic_smoke_test else 'adaptive_kde_validation_results.json')
    path.write_text(json.dumps(out,indent=2));print('Saved',path)
    if not args.synthetic_smoke_test and not args.fast:
        save_adaptive_kde_figures(out, path.parent)
        print('Saved adaptive_kde_auc_comparison.png and adaptive_kde_brier_comparison.png')

if __name__=='__main__':main()
