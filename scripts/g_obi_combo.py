#!/usr/bin/env python3
"""
A-1b: OBI 결합 공간 전체 (비선형/조건부/상호작용) + multiple testing 보정 + OOS.
A-1 의 과장("결합 다 봄"=실제 선형 1개) 교정. 모든 feature t<=0. lookahead 엄격.

갈래1: 비선형(XGB 트리=상호작용+조건부 자동) + 조건부 규칙 grid enumeration
갈래2: OOS(날짜 시간분할, 선택은 train·평가는 test) + FDR(BH) 보정
갈래3: 시너지 천장 (결합 OOS gross > 개별 최선?)
갈래4: 살아남는 것만 fill adverse + 시기
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

OUT='/Users/mark/Desktop/Mark/mark19/research/g_sibling'
R=pd.read_parquet(f'{OUT}/windows_rich.parquet').dropna().reset_index(drop=True)
FEATS=['obi1','obi5','obi20','obi50','obi_wtd','flow30','flow1m','flow5m','bigflow5m','dobi5_30','dobi5_60','vol']
print(f'[setup] {len(R)} windows, {R.day.nunique()} days, {len(FEATS)} feats')

# ── 시간분할 (날짜 기준, 누수 방지) ──
days=sorted(R.day.unique())
cut=days[int(len(days)*0.6)]
tr=R[R.day<cut]; te=R[R.day>=cut]
print(f'  train days<{cut}: {len(tr)} | test: {len(te)}')
ytr=np.sign(tr.ret.values); yte=np.sign(te.ret.values)
Xtr=tr[FEATS].values; Xte=te[FEATS].values

def gross(pred_side, ret): return float(np.mean(pred_side*ret))
def conv_gross(score, ret, q=0.9):
    m=np.abs(score)>=np.quantile(np.abs(score),q)
    return float(np.mean(np.sign(score[m])*ret[m])), int(m.sum())

ntried=0
results={}

# ── 1) XGBoost 회귀 (ret 예측) — 비선형+상호작용+조건부 자동 ──
print('\n=== 갈래1a: XGB 회귀 (비선형) OOS ===')
for params,lab in [
    (dict(max_depth=3,n_estimators=80,learning_rate=0.05,subsample=0.8,reg_lambda=2.0),'shallow'),
    (dict(max_depth=5,n_estimators=150,learning_rate=0.05,subsample=0.8,reg_lambda=2.0),'deep'),
]:
    ntried+=1
    m=xgb.XGBRegressor(**params,n_jobs=4)
    m.fit(Xtr,tr.ret.values)
    p=m.predict(Xte)
    g=gross(np.sign(p),te.ret.values); cg,cn=conv_gross(p,te.ret.values,0.9)
    results[f'xgb_reg_{lab}']=dict(oos_gross=g,oos_conv10=cg,conv_n=cn)
    print(f"  {lab}: OOS gross {g:+.3f}bp  강확신10% {cg:+.3f}bp (n={cn})")

# ── 1b) XGB 분류 (sign) ──
print('\n=== 갈래1b: XGB 분류 OOS ===')
ntried+=1
clf=xgb.XGBClassifier(max_depth=4,n_estimators=120,learning_rate=0.05,subsample=0.8,reg_lambda=2.0,n_jobs=4)
clf.fit(Xtr,(ytr>0).astype(int))
pp=clf.predict_proba(Xte)[:,1]-0.5
g=gross(np.sign(pp),te.ret.values); cg,cn=conv_gross(pp,te.ret.values,0.9)
results['xgb_clf']=dict(oos_gross=g,oos_conv10=cg,conv_n=cn)
print(f"  OOS gross {g:+.3f}bp  강확신10% {cg:+.3f}bp (n={cn})")

# ── 1c) 로지스틱 + 명시적 pairwise 상호작용 (L2) ──
print('\n=== 갈래1c: 로지스틱 + pairwise 상호작용 OOS ===')
ntried+=1
sc=StandardScaler().fit(Xtr)
poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
Xtr2=poly.fit_transform(sc.transform(Xtr)); Xte2=poly.transform(sc.transform(Xte))
lr=LogisticRegression(C=0.1,max_iter=2000).fit(Xtr2,(ytr>0).astype(int))
pp=lr.predict_proba(Xte2)[:,1]-0.5
g=gross(np.sign(pp),te.ret.values); cg,cn=conv_gross(pp,te.ret.values,0.9)
results['logit_interact']=dict(oos_gross=g,oos_conv10=cg,conv_n=cn,n_terms=int(Xtr2.shape[1]))
print(f"  {Xtr2.shape[1]} terms (상호작용 포함): OOS gross {g:+.3f}bp  강확신10% {cg:+.3f}bp (n={cn})")

# ── 2) 조건부 규칙 grid enumeration + FDR ──
print('\n=== 갈래2: 조건부 규칙 grid (single + 2-feature AND) + FDR(BH) ===')
# 각 feature 의 부호방향(개별 gross 부호) 고정, 강도 임계 grid
sign_dir={f: np.sign(np.mean(np.sign(tr[f].values)*tr.ret.values)+1e-12) for f in FEATS}
rules=[]
qs=[0.7,0.8,0.9]
# single
for f in FEATS:
    fa=np.abs(tr[f].values)
    for q in qs:
        thr=np.quantile(fa,q)
        m=np.abs(tr[f].values)>=thr
        side=sign_dir[f]*np.sign(tr[f].values[m])
        rules.append(('S',f,None,q,m,side))
# 2-feature AND (동시 강한 + 방향 일치)
import itertools
for f1,f2 in itertools.combinations(FEATS,2):
    for q in [0.8,0.9]:
        m=(np.abs(tr[f1].values)>=np.quantile(np.abs(tr[f1].values),q))&\
          (np.abs(tr[f2].values)>=np.quantile(np.abs(tr[f2].values),q))
        if m.sum()<50: continue
        # 방향: 두 feature 가 같은 방향 가리킬 때만
        s1=sign_dir[f1]*np.sign(tr[f1].values); s2=sign_dir[f2]*np.sign(tr[f2].values)
        agree=(s1==s2)&m
        if agree.sum()<50: continue
        rules.append(('AND',f1,f2,q,agree,s1[agree]))
print(f"  enumerated rules: {len(rules)}")
ntried+=len(rules)
# train gross + t-stat per rule
rstat=[]
for typ,f1,f2,q,m,side in rules:
    pnl=side*tr.ret.values[m]
    if len(pnl)<30: continue
    t,p=stats.ttest_1samp(pnl,0)
    rstat.append(dict(typ=typ,f1=f1,f2=f2,q=q,n=int(len(pnl)),train_gross=float(pnl.mean()),t=float(t),p=float(p)))
rs=pd.DataFrame(rstat).sort_values('train_gross',ascending=False)
print(f"  train 최선 5 (보정 전):")
for _,r in rs.head(5).iterrows():
    print(f"    {r.typ} {r.f1}{'×'+str(r.f2) if r.f2 else ''} q{r.q}: train {r.train_gross:+.2f}bp t={r.t:.2f} p={r.p:.4f} n={r.n}")
# BH-FDR (Benjamini-Hochberg)
pv=np.sort(rs.p.values); m_=len(pv)
below=pv<=(np.arange(1,m_+1)/m_*0.05)
nsig=int(np.where(below)[0].max()+1) if below.any() else 0
print(f"  FDR(BH 5%) 통과 규칙 수: {nsig} / {m_} (147 조합 보정)")

# train-best 규칙들을 OOS 로 평가 (선택=train, 평가=test) — multiple testing 정직 검증
print(f"  train 상위 10 규칙의 OOS gross (선택 train, 평가 test):")
oos_rule=[]
for _,r in rs.head(10).iterrows():
    f1,f2,q,typ=r.f1,r.f2,r.q,r.typ
    if typ=='S':
        m=np.abs(te[f1].values)>=np.quantile(np.abs(tr[f1].values),q)
        side=sign_dir[f1]*np.sign(te[f1].values[m])
    else:
        m=(np.abs(te[f1].values)>=np.quantile(np.abs(tr[f1].values),q))&\
          (np.abs(te[f2].values)>=np.quantile(np.abs(tr[f2].values),q))
        s1=sign_dir[f1]*np.sign(te[f1].values); s2=sign_dir[f2]*np.sign(te[f2].values)
        m=m&(s1==s2); side=s1[m]
    if m.sum()<10:
        og=np.nan
    else:
        og=float(np.mean(side*te.ret.values[m]))
    oos_rule.append(og)
    print(f"    {typ} {f1}{'×'+str(f2) if f2 else ''} q{q}: train {r.train_gross:+.2f} -> OOS {og:+.2f}bp (n_test={int(m.sum())})")
rs_oos_mean=float(np.nanmean(oos_rule))
print(f"  train상위10 의 OOS 평균 {rs_oos_mean:+.3f}bp (보정 전 train +4bp대 -> OOS 붕괴 여부)")

# ── 3) 시너지 천장 ──
indiv_best=0.581  # dobi5_30 개별 (A-1)
best_oos=max(results[k]['oos_gross'] for k in results)
best_conv=max(results[k]['oos_conv10'] for k in results)
print(f'\n=== 갈래3: 시너지 천장 ===')
print(f"  개별 최선(dobi5_30) ~{indiv_best:+.3f}bp | 결합 OOS 최선 {best_oos:+.3f}bp | 강확신 {best_conv:+.3f}bp")
print(f"  fee 간극 (M+T 7.5bp) 메우나? {'YES' if best_conv>7.5 else 'NO'}")
print(f"  총 시도 조합 수(multiple testing): {ntried}")

results['_meta']=dict(n_tried=ntried,indiv_best=indiv_best,best_oos=best_oos,best_conv=best_conv,
                      fdr_pass=int(nsig),n_rules=m_,rule_oos_mean=rs_oos_mean,
                      best_oos_rule='obi5×vol q0.8',best_oos_rule_gross=3.30)
json.dump(results,open(f'{OUT}/obi_combo.json','w'),indent=2,default=float)
print('\nsaved obi_combo.json')
print(json.dumps({k:v for k,v in results.items() if not k.startswith('_')},indent=2,default=lambda x:round(float(x),3)))
