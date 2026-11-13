"""Reproducible preliminary evaluation for NUST CS-871 Deliverable 2.
Input: diabetes_risk_prediction_dataset.csv in same folder.
No score-derived or post-label fields enter models. All preprocessing fit on train only.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

START=time.time()
BASE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(BASE,'diabetes_risk_prediction_dataset.csv'))
label='Diabetes_Risk'
y=(df[label]=='High').astype(int).to_numpy()
leakage=['Patient_ID','Diabetes_Risk_Score','AI_Health_Recommendation','Doctor_Consultation_Needed','Diabetes_Risk']
X=df.drop(columns=leakage)
num=X.select_dtypes(include='number').columns.tolist()
cat=X.select_dtypes(exclude='number').columns.tolist()
X_train, X_temp, y_train, y_temp=train_test_split(X,y,test_size=.40,stratify=y,random_state=42)
X_val, X_test, y_val, y_test=train_test_split(X_temp,y_temp,test_size=.50,stratify=y_temp,random_state=42)
# for KDE training, fixed stratified subset of training only
X_kde, _, y_kde,_ = train_test_split(X_train, y_train,train_size=8000,stratify=y_train,random_state=42)
metrics={}

def choose_threshold(ytrue, p, target=.90):
    # descending scores; select threshold maximizing specificity with recall >= target
    fpr,tpr,th=roc_curve(ytrue,p)
    ok=np.flatnonzero(tpr>=target)
    i=ok[np.argmin(fpr[ok])]
    return float(th[i])

def score(name,ytrue,p, threshold=.5):
    pred=(p>=threshold).astype(int)
    res=dict(n=int(len(ytrue)), accuracy=float(accuracy_score(ytrue,pred)), balanced_accuracy=float(balanced_accuracy_score(ytrue,pred)),precision_high=float(precision_score(ytrue,pred,zero_division=0)),recall_high=float(recall_score(ytrue,pred)),specificity_other=float(recall_score(ytrue,pred,pos_label=0)),f1_high=float(f1_score(ytrue,pred)),roc_auc=float(roc_auc_score(ytrue,p)),pr_auc=float(average_precision_score(ytrue,p)),threshold=float(threshold),confusion_matrix=confusion_matrix(ytrue,pred,labels=[0,1]).tolist())
    metrics[name]=res
    return res

print('started:',len(X_train),len(X_val),len(X_test),'KDE training:',len(X_kde),'features:',len(num),len(cat),flush=True)
print('split prevalence:',*[round(float(z.mean()),5) for z in [y_train,y_val,y_test,y_kde]],flush=True)
score('majority_high_test',y_test, np.ones_like(y_test,dtype=float))
full_pre=ColumnTransformer(transformers=[('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num), ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
logfull=Pipeline([('pre',full_pre),('model',LogisticRegression(max_iter=1200,solver='lbfgs',random_state=42))])
logfull.fit(X_train,y_train)
p_val_full=logfull.predict_proba(X_val)[:,1]
p_test_full=logfull.predict_proba(X_test)[:,1]
score('logistic_all_36_test',y_test,p_test_full)
full_thr=choose_threshold(y_val,p_val_full)
score('logistic_all_36_recall90_test',y_test,p_test_full,full_thr)
print('logistic full:',metrics['logistic_all_36_test'],'threshold90',full_thr,'time',round(time.time()-START),flush=True)

feature_sets={'3d':['Age','HbA1c','Fasting_Blood_Sugar'], '6d':['Age','BMI','HbA1c','Fasting_Blood_Sugar','Blood_Pressure_Systolic','Sleep_Hours']}
# Additional same-feature logistic comparison
ff=feature_sets['6d']
log6=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,random_state=42))])
log6.fit(X_train[ff],y_train)
p_val6=log6.predict_proba(X_val[ff])[:,1]
p_test6=log6.predict_proba(X_test[ff])[:,1]
score('logistic_6d_test',y_test,p_test6)
print('logistic 6d:',metrics['logistic_6d_test'],'time',round(time.time()-START),flush=True)
log3=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,random_state=42))])
log3.fit(X_train[feature_sets['3d']],y_train)
p_test3=log3.predict_proba(X_test[feature_sets['3d']])[:,1]
score('logistic_3d_test',y_test,p_test3)
print('logistic 3d:',metrics['logistic_3d_test'],'time',round(time.time()-START),flush=True)

chosen={}; tuning=[]; predictions={'Logistic (all 36)':p_test_full, 'Logistic (six features)':p_test6,'Logistic (three features)':p_test3}
for key,features in feature_sets.items():
    imputer=SimpleImputer(strategy='median'); scaler=StandardScaler()
    Ztr=scaler.fit_transform(imputer.fit_transform(X_kde[features]));Zval=scaler.transform(imputer.transform(X_val[features]));Ztest=scaler.transform(imputer.transform(X_test[features]))
    prior=np.bincount(y_kde)/len(y_kde)
    best=None
    for bandwidth in [0.35,0.60,1.00,1.50]:
        models=[]
        for c in (0,1):
            m=KernelDensity(bandwidth=bandwidth,kernel='gaussian',algorithm='ball_tree',atol=0.001,rtol=0.001)
            m.fit(Ztr[y_kde==c]); models.append(m)
        logd=np.column_stack([models[c].score_samples(Zval)+np.log(prior[c]) for c in (0,1)])
        prob=np.exp(logd[:,1]-logsumexp(logd,axis=1))
        auc=roc_auc_score(y_val,prob); pr=average_precision_score(y_val,prob)
        tuning.append(dict(feature_set=key,dimension=len(features),bandwidth=bandwidth,validation_roc_auc=float(auc), validation_pr_auc=float(pr)))
        print('KDE tuning',key,'h',bandwidth,'val AUC',round(auc,4),'AP',round(pr,4),'elapsed',round(time.time()-START),flush=True)
        if best is None or auc>best['auc']:
            best={'auc':auc,'h':bandwidth, 'models':models,'valprob':prob}
    # evaluate once test (untouched during tuning)
    m0,m1=best['models']
    vals=[]
    for c,m in enumerate((m0,m1)):
        vals.append(m.score_samples(Ztest)+np.log(prior[c]))
    logd=np.column_stack(vals)
    p_test=np.exp(logd[:,1]-logsumexp(logd,axis=1))
    chosen[key]=dict(features=features,bandwidth=best['h'],validation_auc=float(best['auc']),train_density_size=len(X_kde),class_prior=prior.tolist())
    score('KDE_'+key+'_test',y_test,p_test)
    threshold=choose_threshold(y_val,best['valprob'])
    score('KDE_'+key+'_recall90_test',y_test,p_test,threshold)
    predictions[f'Parzen KDE ({len(features)} features)']=p_test
    print('KDE best',key,'h',best['h'],'test',metrics['KDE_'+key+'_test'],'elapsed',round(time.time()-START),flush=True)

# Figures
fig,ax=plt.subplots(figsize=(7.2,3.1))
counts=df[label].value_counts().reindex(['Low','Moderate','High'])
bars=ax.bar(counts.index,counts.values,color=['#74a2b3','#e1aa57','#28618c'],width=.55)
for b,v in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+550,f'{v:,} ({v/len(df):.2%})',ha='center',fontsize=9)
ax.set_ylim(0,43000);ax.set_ylabel('Number of records');ax.set_title('Observed target-class distribution (N = 50,000)');ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(os.path.join(BASE,'class_distribution.png'),dpi=185);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.7,4.6))
for name, p in predictions.items():
    fpr,tpr,_=roc_curve(y_test,p); auc=roc_auc_score(y_test,p)
    ax.plot(fpr,tpr,label=f'{name}: {auc:.3f}',linewidth=1.8)
ax.plot([0,1],[0,1],'--',linewidth=.8,color='#777')
ax.set(xlabel='False-positive rate',ylabel='True-positive rate (High-risk recall)',title='Untouched test split: ROC curves');ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.14)
fig.tight_layout();fig.savefig(os.path.join(BASE,'preliminary_roc.png'),dpi=185);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.7,4.6))
for name,p in predictions.items():
    prec,rec,_=precision_recall_curve(y_test,p); ap=average_precision_score(y_test,p)
    ax.plot(rec,prec,label=f'{name}: AP {ap:.3f}',linewidth=1.8)
ax.axhline(y_test.mean(),ls='--',color='#777',lw=.8,label=f'Class prevalence: {y_test.mean():.3f}')
ax.set(xlabel='Recall (High risk)',ylabel='Precision (High risk)',title='Untouched test split: Precision-recall curves');ax.legend(loc='lower left',fontsize=8);ax.grid(alpha=.14)
fig.tight_layout();fig.savefig(os.path.join(BASE,'preliminary_precision_recall.png'),dpi=185);plt.close(fig)

obj={
'dataset':dict(n_rows=len(df),n_columns=len(df.columns),missing_cells=int(df.isna().sum().sum()),rows_with_missing=int(df.isna().any(axis=1).sum()),class_counts=counts.to_dict(), class_prevalences=(counts/len(df)).round(6).to_dict(),missing_by_column={k:int(v) for k,v in df.isna().sum().items() if v>0},target_score_intervals={k:{'min':int(v['min']),'max':int(v['max'])} for k,v in df.groupby(label)['Diabetes_Risk_Score'].agg(['min','max']).to_dict('index').items()},all_scores_unique_class=bool((df.groupby('Diabetes_Risk_Score')['Diabetes_Risk'].nunique()==1).all()),duplicate_rows_excluding_id=int(df.drop(columns='Patient_ID').duplicated().sum())),
'protocol':dict(random_state=42,train_n=len(X_train),val_n=len(X_val),test_n=len(X_test),density_train_n=len(X_kde),features_full_count=X.shape[1],excluded_columns=leakage,train_prev=float(y_train.mean()),test_prev=float(y_test.mean()),tuning_target='validation ROC-AUC',threshold_rule='lowest false-positive rate on validation with High-risk recall >=0.90'),
'kde_selection':chosen,'tuning':tuning,'metrics':metrics,'runtime_sec':time.time()-START}
with open(os.path.join(BASE,'experimental_results.json'),'w') as f:json.dump(obj,f,indent=2)
print('ALL DONE',round(time.time()-START,1),'seconds',flush=True)
