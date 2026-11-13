#!/usr/bin/env python3
"""CS-871 Deliverable 2 reproducible follow-up experiments.

Run: python run_improvements.py [--fast]
Input: diabetes_risk_prediction_dataset.csv alongside this script.
Output: improvements.json and improvements_*.png (300 dpi).

IMPORTANT: Original locked test tables are not retrained/overwritten. All new
hyperparameter decisions are based only on the original validation split.
New models are reported on validation ONLY. The original score is used as a
test-set ERROR DIAGNOSTIC for the original 3D KDE and 3D logistic models ONLY.
The density scoring uses the same scikit-learn BallTree tolerance as the
original: atol=rtol=0.001, which is an approximate rather than exact evaluator.
"""
import argparse, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss,confusion_matrix, f1_score

P=argparse.ArgumentParser();P.add_argument('--fast',action='store_true',help='Only for smoke testing; fewer repeats; JSON states run mode.');P.add_argument('--skip-16000',action='store_true', help='Explicit runtime exception, recorded in JSON.');args=P.parse_args()
BASE=Path(__file__).resolve().parent
SEED=42
start=time.time()
DF=pd.read_csv(BASE/'diabetes_risk_prediction_dataset.csv')
y=DF['Diabetes_Risk'].eq('High').astype(int).to_numpy()
score=DF['Diabetes_Risk_Score']
excluded=['Patient_ID','Diabetes_Risk_Score','AI_Health_Recommendation','Doctor_Consultation_Needed','Diabetes_Risk']
X=DF.drop(columns=excluded)
Xtr,Xtemp,ytr,ytemp=train_test_split(X,y,test_size=.4,random_state=SEED,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtemp,ytemp,test_size=.5,random_state=SEED,stratify=ytemp)
Xk,_,yk,_=train_test_split(Xtr,ytr,train_size=8000,stratify=ytr,random_state=SEED)
F={'3D':['Age','HbA1c','Fasting_Blood_Sugar'],'6D':['Age','BMI','HbA1c','Fasting_Blood_Sugar','Blood_Pressure_Systolic','Sleep_Hours']}
OUT={'provenance':{'seed':SEED,'train_n':len(ytr),'validation_n':len(yv),'test_n':len(yte),'kde_n':len(yk),'fast_mode':args.fast,'skipped_16000':args.skip_16000,'post_hoc':True,'selection_data':'Validation only; no new test prediction metrics for newly selected models','dataset':'diabetes_risk_prediction_dataset.csv','excluded_input_fields':excluded},'locked_original':{},'bandwidth':{},'class_specific':{},'training_size':{},'nonlinear_comparators':{},'imputation':{},'error_by_score':{},'bootstrap':{},'todos':[]}

with open(BASE/'experimental_results.json') as h: LOCK=json.load(h)
with open(BASE/'revised_diagnostics.json') as h: PREVD=json.load(h)
# Locked values are COPY-ONLY. Counts used for F1 are obtained from the locked confusion matrices.
for key in ['majority_high_test','logistic_3d_test','KDE_3d_test','logistic_6d_test','KDE_6d_test','logistic_all_36_test']:
    x=LOCK['metrics'][key];cm=x['confusion_matrix'];tn,fp=cm[0];fn,tp=cm[1]
    f1 = (2*tp)/(2*tp+fp+fn)
    assert abs(f1 - x['f1_high']) < 1e-12,(key,f1,x['f1_high'])
    OUT['locked_original'][key]={'f1':float(f1),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'auc':x['roc_auc'],'ap':x['pr_auc'],'brier':float(PREVD['kde']['3d']['test_default']['brier']) if key=='KDE_3d_test' else None}
for short in ['3d','6d']:
    v=PREVD['matched_logistic'][short]['8000']['default']
    tp,fp,fn,tn=[int(v[k]) for k in ('tp','fp','fn','tn')]
    OUT['locked_original'][f'logistic_{short}_8000_test']={'f1':float(2*tp/(2*tp+fp+fn)),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'auc':v['auc'],'ap':v['ap'],'brier':v['brier']}

def prep(xtrain,xval, features):
    imp=SimpleImputer(strategy='median');sca=StandardScaler()
    a=sca.fit_transform(imp.fit_transform(xtrain[features]));b=sca.transform(imp.transform(xval[features]))
    return a,b,imp,sca

def densities(z,yfit,zval,h,c=None):
    if c is None:
        return [KernelDensity(kernel='gaussian',bandwidth=float(h),algorithm='ball_tree',atol=.001,rtol=.001).fit(z[yfit==j]).score_samples(zval) for j in (0,1)]
    return KernelDensity(kernel='gaussian',bandwidth=float(h),algorithm='ball_tree',atol=.001,rtol=.001).fit(z[yfit==c]).score_samples(zval)

def post(log0,log1,prior):
    a=log0+np.log(1-prior); b=log1+np.log(prior)
    return np.exp(b-logsumexp(np.column_stack((a,b)),axis=1))

def valmet(yt,p):
    return {'auc':float(roc_auc_score(yt,p)),'log_loss':float(log_loss(yt,np.clip(p,1e-15,1-1e-15))),'brier':float(brier_score_loss(yt,p)),'ap_high':float(average_precision_score(yt,p)),'ap_non_high':float(average_precision_score(1-yt,1-p)),'specificity_at_050':float(np.mean(p[yt==0]<.5)),'high_score_min':float(np.min(p)),'high_score_max':float(np.max(p)),'mean_high_score':float(np.mean(p))}

def tqdm_print(*parts):print(' '.join(map(str,parts)),'elapsed',round(time.time()-start,1),'s',flush=True)

# Original diagnostic models in same representation; no NEW test set selected models.
h_dense=np.unique(np.concatenate([np.geomspace(.10,2.0,15),[.25,.35,1.0]]))
vals={} ; base_predictions={}; trained={}
for d, features in F.items():
    z,zval,imp,sc=prep(Xk,Xv,features); yfit=np.asarray(yk);prior=float(yfit.mean());
    OUT['imputation'][d]={'train_8000_by_feature':{f:int(Xk[f].isna().sum()) for f in features},'train_8000_total':int(Xk[features].isna().sum().sum()),'train_8000_any_missing':int(Xk[features].isna().any(axis=1).sum()),'train_8000_complete_cases':int(Xk[features].notna().all(axis=1).sum())}
    s=[]; dens_cache={}
    for h in h_dense:
        log0,log1=densities(z,yfit,zval,h)
        p=post(log0,log1,prior);v=valmet(yv,p)
        s.append({'h':float(h),**v});dens_cache[float(h)]=(log0,log1)
    optauc=max(s,key=lambda q:q['auc']);optll=min(s,key=lambda q:q['log_loss']);optbrier=min(s,key=lambda q:q['brier'])
    # Scott's rule uses sample count in the multivariate joint density; class-specific counts differ.
    scott_all=8000**(-1/(len(features)+4));scott_high=int(yfit.sum())**(-1/(len(features)+4));scott_non=int((1-yfit).sum())**(-1/(len(features)+4))
    OUT['bandwidth'][d]={'grid':s,'selected_auc':optauc,'selected_log_loss':optll,'selected_brier':optbrier,'scott_n_all':float(scott_all),'scott_n_high':float(scott_high),'scott_n_non_high':float(scott_non),'fixed_original_h':.35 if d=='3D' else 1.,'n_train':len(z)}
    vals[d]={'data':s,'z':z,'v':zval,'p_prior':prior,'imp':imp,'sc':sc,'dens':dens_cache,'yfit':yfit}
    tqdm_print('BANDWIDTH',d,'AUC h',round(optauc['h'],4),'LL h',round(optll['h'],4),'6D min scores',round(optll['high_score_min'],4))
    # Independent class-specific h selection over validation log loss. h0 and h1 each in compact precomputed grid.
    hclass=np.unique(np.r_[np.geomspace(.12,1.8,8 if not args.fast else 4), optll['h']])
    log0=np.array([densities(z,yfit,zval,float(h),c=0) for h in hclass])
    log1=np.array([densities(z,yfit,zval,float(h),c=1) for h in hclass])
    scores=[]
    for i,h0 in enumerate(hclass):
      for j,h1 in enumerate(hclass):
        q=post(log0[i],log1[j],prior); scores.append({'h_non_high':float(h0),'h_high':float(h1),**valmet(yv,q)})
    chosen=min(scores,key=lambda x:x['log_loss'])
    OUT['class_specific'][d]={'selection':'minimum validation log-loss','best':chosen,'candidate_count':len(scores),'same_h_log_loss_best':optll['log_loss'],'log_loss_gain':float(optll['log_loss']-chosen['log_loss'])}
    tqdm_print('CLASS SPECIFIC',d,chosen['h_non_high'],chosen['h_high'],'ll gain',round(OUT['class_specific'][d]['log_loss_gain'],5))

# Figure: each criterion and its selected bandwidth, selected independently for each dimension.
fig,ax=plt.subplots(1,3,figsize=(11.2,3.4))
for d,col in [('3D','tab:blue'),('6D','tab:orange')]:
   arr=OUT['bandwidth'][d]['grid']; x=np.array([r['h'] for r in arr]);
   for i,(key,choice) in enumerate([('auc','selected_auc'),('log_loss','selected_log_loss'),('brier','selected_brier')]):
       yy=np.array([r[key] for r in arr]);sel=OUT['bandwidth'][d][choice]
       ax[i].plot(x,yy,'-o',ms=3,linewidth=1.3,label=d,color=col)
       ax[i].scatter([sel['h']],[sel[key]],s=55,marker='*',color=col,zorder=5)
for i,(key,ttl) in enumerate([('auc','ROC-AUC ↑'),('log_loss','Log-loss ↓'),('brier','Brier score ↓')]):
   ax[i].set_xscale('log');ax[i].set_xlabel('Bandwidth h (log scale)');ax[i].set_title(ttl);ax[i].grid(alpha=.18)
ax[0].legend(fontsize=8);fig.suptitle('Validation bandwidth analysis (stars = criterion-specific selected h)',fontsize=11)
fig.tight_layout();fig.savefig(BASE/'improvements_bandwidth.png',dpi=300,bbox_inches='tight');plt.close(fig)

# Paired validation bootstrap of h=.25 against h=.35 in 3D, conditional on the current split.
dens=vals['3D']['dens'];yva=np.asarray(yv); prior=vals['3D']['p_prior']
p25=post(*dens[.25],prior);p35=post(*dens[.35],prior)
rng=np.random.default_rng(20261109);pos=np.flatnonzero(yva==1);neg=np.flatnonzero(yva==0)
boot=[]
for _ in range(600):
    idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
    boot.append(roc_auc_score(yva[idx],p25[idx])-roc_auc_score(yva[idx],p35[idx]))
OUT['bootstrap']['3d_auc_h025_minus_h035_validation']={'point':float(roc_auc_score(yva,p25)-roc_auc_score(yva,p35)),'paired_stratified_bootstrap_95ci':list(map(float,np.quantile(boot,[.025,.975]))),'resamples':600,'note':'Conditional validation bootstrap; NOT repeated training or split-to-split variation.'}

# Repeated training-size analysis with fixed and per-repeat validation-selected bandwidths.
# For feasibility, all repeated sample-size experiments use the SAME precommitted 2500-record
# stratified validation evaluation subset, not the complete validation (10k). Selection remains validation-only.
Xvs,_,yvs,_=train_test_split(Xv,yv,train_size=2500,stratify=yv,random_state=42)
sizes=[1000,2000,4000,8000] + ([] if args.skip_16000 else [16000])
reps=3 if args.fast else 10
# Reduced, prespecified grid (not the dense grid above) to control compute in 100+ fits.
h_siz=np.geomspace(.20,1.60,6) if not args.fast else np.array([.25,.5,1.0])
for d,features in F.items():
    size_out={};fixed_h=.35 if d=='3D' else 1.
    for n in sizes:
        rows=[]
        for r in range(reps):
            sx,_,sy,_=train_test_split(Xtr,ytr,train_size=n,random_state=1000+10*n+r,stratify=ytr)
            z,v,_,_=prep(sx,Xvs,features);sy=np.asarray(sy);prior=float(sy.mean())
            # Fit one KDE per h and reuse fixed-h probability when h in candidate grid.
            vals_h=[]
            for h in sorted(set(h_siz.tolist()+[fixed_h])):
                lp0,lp1=densities(z,sy,v,h); pp=post(lp0,lp1,prior)
                vals_h.append({'h':float(h),'auc':float(roc_auc_score(yvs,pp))})
            fixed=next(t['auc'] for t in vals_h if np.isclose(t['h'],fixed_h))
            best=max(vals_h,key=lambda t:t['auc'])
            # Logistic fitted on the same exact random subset and evaluated on same validation rows.
            lr=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('lr',LogisticRegression(C=1,max_iter=500,random_state=SEED))])
            lr.fit(sx[features],sy);lp=lr.predict_proba(Xvs[features])[:,1]
            rows.append({'repeat':r,'seed':1000+10*n+r,'kde_fixed_h_auc':fixed,'kde_selected_h_auc':best['auc'],'kde_selected_h':best['h'],'logistic_auc':float(roc_auc_score(yvs,lp))})
        summary={k:{'mean':float(np.mean([a[k] for a in rows])),'sd':float(np.std([a[k] for a in rows],ddof=1))} for k in ['kde_fixed_h_auc','kde_selected_h_auc','logistic_auc','kde_selected_h']}
        size_out[str(n)]={'summary':summary,'runs':rows}
        tqdm_print('SIZE',d,n,'avg',round(summary['kde_selected_h_auc']['mean'],4),'logistic',round(summary['logistic_auc']['mean'],4))
    OUT['training_size'][d]={'repeats':reps,'size_grid':size_out,'h_candidates':list(map(float,h_siz)),'fixed_h':fixed_h,'validation_subset_n':len(yvs),'validation_subset_seed':SEED,'selection':'maximize subset validation ROC-AUC within each fit'}
fig,axs=plt.subplots(1,2,figsize=(10.6,3.65),sharey=True)
for j,d in enumerate(F):
    dct=OUT['training_size'][d]['size_grid'];xs=np.array(sorted(map(int,dct)))
    for key,lbl,clr in [('kde_fixed_h_auc','KDE fixed h','tab:blue'),('kde_selected_h_auc','KDE validation-chosen h','tab:green'),('logistic_auc','Logistic C=1','tab:orange')]:
        ms=np.array([dct[str(s)]['summary'][key]['mean'] for s in xs]);ss=np.array([dct[str(s)]['summary'][key]['sd'] for s in xs]);axs[j].errorbar(xs,ms,yerr=ss,marker='o',capsize=3,label=lbl,color=clr,lw=1.3)
    axs[j].set_title(d);axs[j].set_xlabel('Training n');axs[j].grid(alpha=.2)
axs[0].set_ylabel('Validation ROC-AUC (mean ± 1 SD)');axs[0].legend(fontsize=7);fig.suptitle('Repeated stratified fitting samples (10 repeats; validation subset n=2500)');fig.tight_layout()
fig.savefig(BASE/'improvements_training_size.png',dpi=300,bbox_inches='tight');plt.close(fig)

# Nonlinear and probabilistic comparators: ALL choices use full validation split; all train on original Xk.
for d,features in F.items():
    arr=[]
    for C in [.01,.1,1.,10.]:
      model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('poly',PolynomialFeatures(degree=2,include_bias=False)),('lr',LogisticRegression(C=C,max_iter=1200,random_state=SEED))]);model.fit(Xk[features],yk);pv=model.predict_proba(Xv[features])[:,1]
      arr.append({'name':'quadratic_logistic','param':{'C':C},**valmet(yv,pv)})
    for vs in [1e-9,1e-6,.001,.1]:
      model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('gnb',GaussianNB(var_smoothing=vs))]);model.fit(Xk[features],yk);pv=model.predict_proba(Xv[features])[:,1]
      arr.append({'name':'gaussian_naive_bayes','param':{'var_smoothing':vs},**valmet(yv,pv)})
    for reg in [0,.01,.1,.3]:
      model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('qda',QuadraticDiscriminantAnalysis(reg_param=reg))]);model.fit(Xk[features],yk);pv=model.predict_proba(Xv[features])[:,1]
      arr.append({'name':'qda','param':{'reg_param':reg},**valmet(yv,pv)})
    for k in [5,15,35,75,151]:
      for w in ['uniform','distance']:
        model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=k,weights=w,n_jobs=-1))]);model.fit(Xk[features],yk);pv=model.predict_proba(Xv[features])[:,1]
        arr.append({'name':'knn','param':{'k':k,'weights':w},**valmet(yv,pv)})
    best={name:max([v for v in arr if v['name']==name],key=lambda v:v['auc']) for name in sorted({v['name'] for v in arr})}
    OUT['nonlinear_comparators'][d]={'selection':'maximum validation ROC-AUC, fit n=8000; NEVER evaluated on test','best':best,'all_candidates':arr}
    tqdm_print('COMPARATORS',d,[(name,round(it['auc'],4)) for name,it in best.items()])

# Side-by-side 2D posterior slice: third feature Age fixed at training median; 3D models are original fitting rules.
features=F['3D'];vi=vals['3D'];z=vi['z'];imp=vi['imp'];sc=vi['sc'];prior=vi['p_prior']; yfit=vi['yfit']
# grids use observed 1st–99th percentile (raw units). Age fixed to median, not a fitted 'mean case'.
gx=np.linspace(float(Xtr['HbA1c'].quantile(.01)),float(Xtr['HbA1c'].quantile(.99)),90)
gy=np.linspace(float(Xtr['Fasting_Blood_Sugar'].quantile(.01)),float(Xtr['Fasting_Blood_Sugar'].quantile(.99)),90)
xx,yy=np.meshgrid(gx,gy);med_age=float(Xk['Age'].median());g=pd.DataFrame({'Age':np.full(xx.size,med_age),'HbA1c':xx.ravel(),'Fasting_Blood_Sugar':yy.ravel()})
zg=sc.transform(imp.transform(g[features]));h=.35
log0,log1=densities(z,yfit,zg,h);probs_kde=post(log0,log1,prior).reshape(xx.shape)
lr=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(C=1,max_iter=1500,random_state=SEED))]);lr.fit(Xk[features],yk);probs_lr=lr.predict_proba(g[features])[:,1].reshape(xx.shape)
OUT['posterior_contours']={'age_fixed_median':med_age,'HbA1c_range':[float(gx[0]),float(gx[-1])],'Fasting_Blood_Sugar_range':[float(gy[0]),float(gy[-1])],'grid_resolution':90,'models':'Original KDE h=0.35 and 8k logistic C=1; explanatory slices; not for selecting models'}
fig,ax=plt.subplots(1,2,figsize=(10.4,4.3),sharex=True,sharey=True)
for a,pr,title in zip(ax,[probs_kde,probs_lr],['3D KDE (h=0.35)','3D logistic (C=1)']):
    cf=a.contourf(xx,yy,pr,levels=np.linspace(0,1,21),cmap='viridis',vmin=0,vmax=1)
    if np.nanmin(pr)<.5<np.nanmax(pr):a.contour(xx,yy,pr,levels=[.5],colors='white',linewidths=1.4)
    a.set_title(title);a.set_xlabel('HbA1c')
ax[0].set_ylabel('Fasting blood sugar');fig.colorbar(cf,ax=ax,label='Predicted High probability',fraction=.023)
fig.suptitle(f'Posterior slices; Age fixed at training median = {med_age:g}')
fig.subplots_adjust(left=.08,right=.86,bottom=.13,top=.82,wspace=.11);fig.savefig(BASE/'improvements_posterior_contours.png',dpi=300,bbox_inches='tight');plt.close(fig)

# Score boundary diagnostics on ORIGINAL test model fits only, not for tuning or inference.
# Risk score NEVER in model inputs. Rebuild original 3D KDE and 8k logistic only.
zt=sc.transform(imp.transform(Xte[features]));l0,l1=densities(z,yfit,zt,.35);pkt=post(l0,l1,prior)
pltp=lr.predict_proba(Xte[features])[:,1]
# Original test membership must align with original DataFrame index.
score_te=DF.loc[Xte.index,'Diabetes_Risk_Score'].to_numpy()
assert np.array_equal(yte,(score_te>=65).astype(int))
bins=np.array([13,30,40,50,55,60,62,64,65,67,70,75,80,90,101]);labels=[];bt=[]
for a,b in zip(bins[:-1],bins[1:]):
 mask=(score_te>=a)&(score_te<b)
 if not mask.any():continue
 labels.append(f'{a}–{b-1}');bt.append({'score_min':int(a),'score_max':int(b-1),'n':int(mask.sum()),'kde_misclassification_at_050':float(np.mean((pkt[mask]>=.5)!=yte[mask])),'logistic_misclassification_at_050':float(np.mean((pltp[mask]>=.5)!=yte[mask])),'kde_mean_high_probability':float(pkt[mask].mean()),'logistic_mean_high_probability':float(pltp[mask].mean())})
OUT['error_by_score']={'diagnostic_only':True,'score_never_used_as_input':True,'locked_original_model_predictions_only':True,'bins':bt,'sample_counts':[int(i['n']) for i in bt]}
fig,ax=plt.subplots(figsize=(10.3,3.5));ind=np.arange(len(bt))
ax.plot(ind,[v['kde_misclassification_at_050'] for v in bt],'-o',ms=4,label='KDE 3D (h=.35)')
ax.plot(ind,[v['logistic_misclassification_at_050'] for v in bt],'-s',ms=4,label='Logistic 3D (n=8k)')
ax.axvline(next((i-0.5 for i,v in enumerate(bt) if v['score_min']>=65),len(bt)),ls='--',c='gray',label='Moderate/High 64/65 boundary')
ax.set_xticks(ind,labels,rotation=45,ha='right');ax.set_ylim(bottom=0);ax.set(ylabel='Misclassification rate at 0.50',xlabel='True synthetic risk-score bin (diagnostic only)');ax.grid(alpha=.2);ax.legend(fontsize=7.7)
fig.tight_layout();fig.savefig(BASE/'improvements_error_by_score.png',dpi=300,bbox_inches='tight');plt.close(fig)
OUT['locked_original']['non_high_average_precision']={
'kde3d':float(average_precision_score(1-yte,1-pkt)),
'logistic3d_8000':float(average_precision_score(1-yte,1-pltp)),
'non_high_prevalence':float(np.mean(1-yte))}

# Imputation diagnostic on validation; complete-case-only comparison with same h and explicit population/sample change.
for d,features in F.items():
    maskcc=Xk[features].notna().all(axis=1).to_numpy();xcc=Xk.loc[maskcc,features];yc=np.asarray(yk)[maskcc]
    vv=Xv[Xv[features].notna().all(axis=1)][features];yvcc=yv[np.isin(Xv.index,vv.index)]
    # avoid pd indexing error: yv is ndarray with Xv index in same order, vv index retains order
    assert len(yvcc)==len(vv)
    zcc,vcc,_,_=prep(xcc,vv,features)
    fixed_h=.35 if d=='3D' else 1.0
    dd0,dd1=densities(zcc,np.asarray(yc),vcc,fixed_h);pcc=post(dd0,dd1,float(yc.mean()))
    # compare imputed model on SAME complete-case validation subset
    zbase=vals[d]['sc'].transform(vals[d]['imp'].transform(vv[features]));l0,l1=densities(vals[d]['z'],vals[d]['yfit'],zbase,fixed_h);pbase=post(l0,l1,vals[d]['p_prior'])
    OUT['imputation'][d]['validation_complete_case_sensitivity']={'complete_case_train_n':len(yc),'complete_case_validation_n':len(yvcc),'complete_case_auc':float(roc_auc_score(yvcc,pcc)),'imputed_fit_auc_on_same_complete_val_rows':float(roc_auc_score(yvcc,pbase)),'fixed_h':fixed_h,'note':'Training distributions differ; descriptive selection-free sensitivity only.'}
    tqdm_print('COMPLETE CASE',d,len(yc),round(OUT['imputation'][d]['validation_complete_case_sensitivity']['complete_case_auc'],4))

OUT['runtime_seconds']=round(time.time()-start,2)
if args.fast:OUT['todos'].append('FAST MODE: 3 repeats, fewer class-specific candidates and training-size bandwidths. Rerun without --fast for requested full experiments.')
if args.skip_16000:OUT['todos'].append('Training n=16000 not run by command-line request; rerun without --skip-16000.')
with open(BASE/'improvements.json','w') as f:json.dump(OUT,f,indent=2)
print('DONE,',round(OUT['runtime_seconds'],1),'s; output:',BASE/'improvements.json',flush=True)
