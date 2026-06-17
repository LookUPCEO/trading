#!/usr/bin/env python3
"""[I] 27단계 작업2 — 하락 전용 표현 (21차원 거울 X, 최대 자유).
질문: 하락 예측에 최적인 라벨/표현이 상승용 21차원과 다른가. 하락 선행지표 있나.
방법: 지도학습 down 분류기(로지스틱 + GBM)로 47 라벨 전부 자유 사용(+비선형).
  목표 y = 미래 4h(및 1d) 하락. train(2024Q1~2025Q2)→test(2025Q3+) 시간분할 OOS.
  → 47 라벨 다 줘도(=표현 최대자유) OOS 하락 예측 안 되면 표현이 병목 아님 = 하락 구조적 예측불가.
  계수 = 하락 선행지표 식별. 거래화: 확신 상위 decile short hit/net vs fee. lookahead 0(라벨 causal)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FEE=11.0;STRIDE=5
NONFEAT={'yr','day','sec','min_of_day','mid'}
TRAIN_Q=set(['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2'])

def main():
    df=pd.read_parquet(LAB)
    feats=[c for c in df.columns if c not in NONFEAT]
    days=sorted(df['day'].unique());dix={d:i for i,d in enumerate(days)}
    drow=df['day'].map(dix).to_numpy();mod=df['min_of_day'].to_numpy()
    # cross-day 가격
    Pf=np.full(len(days)*1440,np.nan,np.float32)
    Pf[drow*1440+mod]=df['mid'].to_numpy(np.float32);NG=len(Pf);gf=drow*1440+mod
    yr=df['yr'].astype(int).to_numpy();month=df['day'].str[5:7].astype(int).to_numpy()
    qtr=np.array([f"{yr[i]}Q{(month[i]-1)//3+1}" for i in range(len(df))])
    # 쿼리 = stride, 2024+
    qmask=(yr>=2024)&(mod%STRIDE==0)
    fr4=np.full(len(df),np.nan,np.float32);t=gf+240;ok=t<NG;fr4[ok]=Pf[t[ok]]/Pf[gf[ok]]-1
    fr1d=np.full(len(df),np.nan,np.float32);t=gf+1440;ok=t<NG;fr1d[ok]=Pf[t[ok]]/Pf[gf[ok]]-1
    X=df[feats].to_numpy(np.float32)
    for hn,fr in [('4h',fr4),('1d',fr1d)]:
        valid=qmask&~np.isnan(fr)&(fr!=0)&~np.isnan(X).any(1)
        idx=np.where(valid)[0]
        Xv=X[idx];y=(fr[idx]<0).astype(int)  # 1=down
        tr=np.array([qtr[i] in TRAIN_Q for i in idx]);te=~tr
        # 표준화 (train 통계)
        mu=Xv[tr].mean(0);sd=Xv[tr].std(0)+1e-9;Xz=(Xv-mu)/sd
        print(f"\n===== 하락 표현 {hn}: n={len(idx)} (train {tr.sum()}/test {te.sum()}), base P(down)={y.mean():.3f} =====")
        # 로지스틱
        lr=LogisticRegression(max_iter=2000,C=0.5).fit(Xz[tr],y[tr])
        p_lr=lr.predict_proba(Xz[te])[:,1]
        auc_lr=roc_auc_score(y[te],p_lr)
        # GBM (비선형 최대자유)
        gb=HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.05,
            l2_regularization=1.0,validation_fraction=0.15,random_state=0).fit(Xv[tr],y[tr])
        p_gb=gb.predict_proba(Xv[te])[:,1]
        auc_gb=roc_auc_score(y[te],p_gb)
        print(f"  OOS AUC(down): 로지스틱 {auc_lr:.4f} | GBM {auc_gb:.4f}  (0.5=무작위)")
        # 거래화: 확신 상위 decile down → short. day-cluster.
        dq=drow[idx][te];frte=fr[idx][te]
        for nm,p in [('로지스틱',p_lr),('GBM',p_gb)]:
            thr=np.percentile(p,90)
            sel=p>=thr
            net=(-1)*frte[sel]*1e4-FEE
            dm=pd.Series(net).groupby(dq[sel]).mean()
            bs=np.random.default_rng(7).choice(dm.to_numpy(),(4000,len(dm)),replace=True).mean(1)
            lo,hi=np.percentile(bs,[2.5,97.5])
            print(f"  {nm} 확신상위10% short: n={sel.sum()} hit {(frte[sel]<0).mean():.3f} "
                  f"net {net.mean():+.1f}bp dayCI[{lo:+.0f},{hi:+.0f}] {'fee초과+OOS생존' if net.mean()>0 and lo>0 else 'X'}")
        if hn=='4h':
            co=pd.Series(lr.coef_[0],index=feats).sort_values()
            print("  하락 선행지표 (로지스틱 계수 +가 down 예측 강함) top:")
            print("   down↑:",", ".join(f"{k}{v:+.2f}" for k,v in co.tail(6)[::-1].items()))
            print("   up↑  :",", ".join(f"{k}{v:+.2f}" for k,v in co.head(6).items()))
    print(f"\n현행 4h 상승 hit 0.68. 표현 최대자유(47라벨+GBM)로도 down OOS 안 되면 표현 병목 아님.")

if __name__=='__main__':
    main()
