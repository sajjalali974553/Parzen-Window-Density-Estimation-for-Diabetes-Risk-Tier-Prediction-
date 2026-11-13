"""Supplemental, reproducible analyses for CS-871 Deliverable 2.
Uses the same fixed split as run_preliminary_experiments.py.
NO NEW TEST-SET MODEL SELECTION; sensitivity analyses only on validation.
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, brier_score_loss, recall_score, balanced_accuracy_score
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.dirname(os.path.abspath(__file__))
start=time.time()
DF=pd.read_csv(os.path.join(BASE,'diabetes_risk_prediction_dataset.csv'))
y=DF['Diabetes_Risk'].eq('High').astype(int).to_numpy()
excluded=['Patient_ID','Diabetes_Risk_Score','AI_Health_Recommendation','Doctor_Consultation_Needed','Diabetes_Risk']
X=DF.drop(columns=excluded)
num=X.select_dtypes(include='number').columns.tolist()
cat=X.select_dtypes(exclude='number').columns.tolist()
Xtrain,Xtemp,ytrain,ytemp=train_test_split(X,y,test_size=.4,random_state=42,stratify=y)
Xval,Xtest,yval,ytest=train_test_split(Xtemp,ytemp,test_size=.5,random_state=42,stratify=ytemp)
Xk,_,yk,_=train_test_split(Xtrain,ytrain,train_size=8000,stratify=ytrain,random_state=42)
feature_sets={'3d':['Age','HbA1c','Fasting_Blood_Sugar'], '6d':['Age','BMI','HbA1c','Fasting_Blood_Sugar','Blood_Pressure_Systolic','Sleep_Hours']}
O={'protocol':{'split_random_state':42,'diagnostic_date':'2026-11-09','n_train':len(ytrain),'n_val':len(yval),'n_test':len(ytest),'n_kde':len(yk),'test_prevalence':float(ytest.mean()),'n_numeric':len(num),'n_categorical':len(cat)},'matched_logistic':{},'kde':{},'descriptives':{},'bootstrap':{}}

# Full input dimension, inferred directly from preprocessing (avoid category-count assumption).
fp=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),cat)])
fp.fit(Xtrain)
O['protocol']['full_logistic_encoded_dimension']=int(fp.transform(Xtrain.iloc[:2]).shape[1])
O['protocol']['categorical_encoded_columns']=int(O['protocol']['full_logistic_encoded_dimension']-len(num))
O['descriptives']['pcos_male_yes']=int(((DF['Gender'].astype(str).str.strip().str.lower()=='male')&(DF['PCOS'].astype(str).str.strip().str.lower()=='yes')).sum())


def threshold_recall90(y0,p):
    fpr,tpr,th=roc_curve(y0,p)
    ix=np.flatnonzero(tpr>=0.90)
    ix=ix[np.argmin(fpr[ix])]
    return float(th[ix])

def metrics(p,threshold=.5):
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(ytest,pred,labels=[0,1]).ravel()
    return {'auc':float(roc_auc_score(ytest,p)), 'ap':float(average_precision_score(ytest,p)), 'brier':float(brier_score_loss(ytest,p)), 'threshold':float(threshold),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)), 'balanced_accuracy':float(balanced_accuracy_score(ytest,pred)), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp), 'p_min':float(p.min()),'p_max':float(p.max()),'p_median':float(np.median(p))}

# Expanded reference and regularization sensitivity, so all table entries are reproducible.
full_lr=Pipeline([('pre',fp),('model',LogisticRegression(max_iter=1200,C=1,random_state=42))])
full_lr.fit(Xtrain,ytrain)
O['full_logistic_brier_test']=float(brier_score_loss(ytest,full_lr.predict_proba(Xtest)[:,1]))
O['logistic_regularization_validation_auc']={}
for fkey,features in feature_sets.items():
    O['logistic_regularization_validation_auc'][fkey]={}
    for c in [0.01,0.1,1,10,100]:
        model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('logit',LogisticRegression(C=c,max_iter=1500,solver='lbfgs',random_state=42))])
        model.fit(Xk[features],yk)
        pval=model.predict_proba(Xval[features])[:,1]
        O['logistic_regularization_validation_auc'][fkey][str(c)]=float(roc_auc_score(yval,pval))

predictions={}
for fs,features in feature_sets.items():
    O['matched_logistic'][fs]={}
    for tag,xx,yy in [('8000',Xk,yk),('30000',Xtrain,ytrain)]:
        m=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('lr',LogisticRegression(max_iter=1500,C=1.,solver='lbfgs',random_state=42))])
        m.fit(xx[features], yy)
        pv=m.predict_proba(Xval[features])[:,1]
        p=m.predict_proba(Xtest[features])[:,1]
        t=threshold_recall90(yval,pv)
        O['matched_logistic'][fs][tag]={'validation_auc':float(roc_auc_score(yval,pv)),'default':metrics(p),'recall90':metrics(p,t)}
        predictions['logistic_'+fs+'_'+tag]=p
        print('LOG',fs,tag,'AUC',round(roc_auc_score(ytest,p),5),'time',round(time.time()-start,1),flush=True)

for fs,features in feature_sets.items():
    im=SimpleImputer(strategy='median');sc=StandardScaler()
    ztr=sc.fit_transform(im.fit_transform(Xk[features]));zv=sc.transform(im.transform(Xval[features]));zt=sc.transform(im.transform(Xtest[features]));priors=np.bincount(yk)/len(yk)
    O['kde'][fs]={'validation_bandwidth_sensitivity':{},'training_size_sensitivity':{}}
    for h in ([0.20,0.25,0.35] if fs=='3d' else [1.0]):
        models=[]
        for c in [0,1]:
            model=KernelDensity(kernel='gaussian',bandwidth=h,algorithm='ball_tree',atol=.001,rtol=.001).fit(ztr[yk==c]);models.append(model)
        ld=np.column_stack([models[c].score_samples(zv)+np.log(priors[c]) for c in [0,1]])
        pv=np.exp(ld[:,1]-logsumexp(ld,axis=1))
        O['kde'][fs]['validation_bandwidth_sensitivity'][str(h)]={'auc':float(roc_auc_score(yval,pv)),'ap':float(average_precision_score(yval,pv))}
        if h==(.35 if fs=='3d' else 1.0):
            te=np.column_stack([models[c].score_samples(zt)+np.log(priors[c]) for c in [0,1]])
            p=np.exp(te[:,1]-logsumexp(te,axis=1))
            t=threshold_recall90(yval,pv)
            O['kde'][fs]['test_default']=metrics(p)
            O['kde'][fs]['test_recall90']=metrics(p,t)
            predictions['kde_'+fs]=p
        print('KDE',fs,'h',h,'VAL AUC',round(roc_auc_score(yval,pv),5),'t',round(time.time()-start,1),flush=True)
    # isolate sample-size variation on validation with fixed bandwidth; do not inspect test
    for n in ([2000,4000,8000] if fs=='3d' else [4000,8000]):
        if n==8000:
            sx,sy=Xk,yk
        else:
            sx,_,sy,_=train_test_split(Xk,yk,train_size=n,random_state=42,stratify=yk)
        ii=SimpleImputer(strategy='median');ss=StandardScaler()
        zz=ss.fit_transform(ii.fit_transform(sx[features]));v=ss.transform(ii.transform(Xval[features]));pri=np.bincount(sy)/len(sy)
        h=.35 if fs=='3d' else 1.0
        mod=[KernelDensity(kernel='gaussian',bandwidth=h,algorithm='ball_tree',atol=.001,rtol=.001).fit(zz[sy==c]) for c in [0,1]]
        logd=np.column_stack([mod[c].score_samples(v)+np.log(pri[c]) for c in [0,1]])
        pv=np.exp(logd[:,1]-logsumexp(logd,axis=1))
        O['kde'][fs]['training_size_sensitivity'][str(n)]={'validation_auc':float(roc_auc_score(yval,pv)),'validation_ap':float(average_precision_score(yval,pv))}
        print('SIZE',fs,n,round(roc_auc_score(yval,pv),5),'time',round(time.time()-start,1),flush=True)

# Paired bootstrap on a fixed held-out set; descriptive single-split uncertainty only.
rng=np.random.default_rng(20261109)
pos=np.flatnonzero(ytest==1);neg=np.flatnonzero(ytest==0)
diffs=[];a3=[];l3=[]
for i in range(600):
    ids=np.r_[rng.choice(pos,size=len(pos),replace=True),rng.choice(neg,size=len(neg),replace=True)]
    a=roc_auc_score(ytest[ids],predictions['kde_3d'][ids]);b=roc_auc_score(ytest[ids],predictions['logistic_3d_8000'][ids]);diffs.append(a-b);a3.append(a);l3.append(b)
O['bootstrap']={'method':'600 stratified paired test-set bootstrap resamples; descriptive uncertainty conditional on fixed fitting split, not repeated-training CI','n_resamples':600,'kde_3d_auc_95ci':list(map(float,np.quantile(a3,[.025,.975]))),'logistic_3d_8000_auc_95ci':list(map(float,np.quantile(l3,[.025,.975]))),'auc_diff_kde_minus_equal_n_logistic':float(roc_auc_score(ytest,predictions['kde_3d'])-roc_auc_score(ytest,predictions['logistic_3d_8000'])),'auc_diff_95ci':list(map(float,np.quantile(diffs,[.025,.975])))}

fig,ax=plt.subplots(figsize=(7.0,3.35))
for fs,color in [('3d','#176c87'),('6d','#b86e40')]:
    d=O['kde'][fs]['training_size_sensitivity']
    xs=sorted(map(int,d))
    ax.plot(xs,[d[str(n)]['validation_auc'] for n in xs],marker='o',lw=2,label=f'{fs.upper()} KDE')
ax.set(xlabel='KDE training records (stratified samples)',ylabel='Validation ROC-AUC',title='Training-size sensitivity at fixed bandwidth')
ax.grid(alpha=.2);ax.legend(loc='best');ax.set_ylim(.81,.88);fig.tight_layout()
fig.savefig(os.path.join(BASE,'revised_training_size_sensitivity.png'),dpi=185);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.0,3.45))
for key,label in [('kde_3d','3D Parzen KDE'),('logistic_3d_8000','3D logistic, equal n'),('kde_6d','6D Parzen KDE'),('logistic_6d_8000','6D logistic, equal n')]:
    from sklearn.metrics import roc_curve
    fpr,tpr,_=roc_curve(ytest,predictions[key]);ax.plot(fpr,tpr,lw=1.7,label=f'{label}: {roc_auc_score(ytest,predictions[key]):.3f}')
ax.plot([0,1],[0,1],linestyle='--',color='grey',lw=.8)
ax.grid(alpha=.2);ax.set(xlabel='False-positive rate',ylabel='High-class recall',title='ROC curves: equal 8,000-record fitting subsets')
ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(os.path.join(BASE,'revised_equal_sample_roc.png'),dpi=185);plt.close(fig)
from sklearn.metrics import precision_recall_curve
fig,ax=plt.subplots(figsize=(7.0,3.45))
for key,label in [('kde_3d','3D Parzen KDE'),('logistic_3d_8000','3D logistic, equal n'),('kde_6d','6D Parzen KDE'),('logistic_6d_8000','6D logistic, equal n')]:
    precision,recall,_=precision_recall_curve(ytest,predictions[key]);ap=average_precision_score(ytest,predictions[key])
    ax.plot(recall,precision,lw=1.7,label=f'{label}: {ap:.3f}')
ax.axhline(ytest.mean(),linestyle='--',color='grey',lw=.8,label=f'Prevalence: {ytest.mean():.3f}')
ax.grid(alpha=.2);ax.set(xlabel='High-class recall',ylabel='High-class precision',title='Precision–recall: equal 8,000-record subsets')
ax.legend(loc='lower left',fontsize=7.7);fig.tight_layout();fig.savefig(os.path.join(BASE,'revised_equal_sample_pr.png'),dpi=185);plt.close(fig)
O['runtime_s']=time.time()-start
with open(os.path.join(BASE,'revised_diagnostics.json'),'w') as f:json.dump(O,f,indent=2)
print('DONE',round(time.time()-start,1),'s',flush=True)
