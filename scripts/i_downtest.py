#!/usr/bin/env python3
"""[I] 28단계 작업2~4 — 새 하락 라벨이 하락을 담나 (라벨부족 vs 시장본질 가름).
핵심: 27단계 47라벨+GBM = OOS AUC(down 4h) 0.52. 새 10라벨 추가 시 AUC 오르나.
  안 오르면 = 하락 진짜 예측불가(라벨 문제 아님). 오르면 = 라벨 부족(본인 지적 적중).
작업2: ① 새 라벨 vs 기존 47 상관(다른 정보?) ② 새 라벨 단독 down 예측력 ③ AUC: 47 vs 47+10 vs 10.
작업3+4: 새 라벨로 하락 horizon hit/net, 확신상위 short OOS, fee/Bonferroni.
lookahead 0 (라벨 causal, 미래=쿼리 자기 4h/30m cross-day). train(24Q1~25Q2)→test(25Q3+)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FEE=11.0
NEW=['bid_dep_chg30','bid_dep_chg60','ask_dep_chg30','book_thin_asym','bid_conc',
     'dn_rv_60','dn_rv_300','rv_skew_300','sell_accel','sell_spike']
NONFEAT={'yr','day','sec','min_of_day','mid'}
TRAIN_Q=set(['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2'])

def clipz(X, lo=1, hi=99):
    # robust: percentile clip (heavy tail) then z (train 통계는 호출부에서)
    return X

def main():
    dl=pd.read_parquet(f'{OUT}/downlab.parquet')
    lab=pd.read_parquet(LAB)
    old=[c for c in lab.columns if c not in NONFEAT]
    M=dl.merge(lab,on=['day','min_of_day'],how='inner',suffixes=('','_lab'))
    print(f"[merge] downlab {len(dl)} ∩ labels = {len(M)} rows, {M.day.nunique()} days")
    yr=M['day'].str[:4].astype(int).to_numpy();month=M['day'].str[5:7].astype(int).to_numpy()
    qtr=np.array([f"{yr[i]}Q{(month[i]-1)//3+1}" for i in range(len(M))])
    # cross-day 미래 (day,min 정렬 후 글로벌 가격)
    days=sorted(M['day'].unique());dix={d:i for i,d in enumerate(days)}
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid']);full=full[full.day.isin(days)]
    Pf=np.full(len(days)*1440,np.nan,np.float32)
    Pf[full['day'].map(dix).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    gf=M['day'].map(dix).to_numpy()*1440+M['min_of_day'].to_numpy();NG=len(Pf)
    def fr(H):
        f=np.full(len(M),np.nan,np.float32);t=gf+H;ok=t<NG;f[ok]=Pf[t[ok]]/Pf[gf[ok]]-1;return f
    fr4=fr(240); fr30=fr(30)

    print("\n===== 작업1: 새 라벨이 기존 47과 다른 정보인가 (최대 |corr|) =====")
    Xnew_raw=M[NEW].to_numpy(np.float64)
    Xold=M[old].to_numpy(np.float64)
    for j,nm in enumerate(NEW):
        a=Xnew_raw[:,j];va=~np.isnan(a)
        best=0;bo=''
        for k,om in enumerate(old):
            b=Xold[:,k];vv=va&~np.isnan(b)
            if vv.sum()<1000: continue
            c=abs(np.corrcoef(a[vv],b[vv])[0,1])
            if c>best: best=c;bo=om
        print(f"  {nm:16s}: max|corr| {best:.3f} vs {bo} {'(중복)' if best>0.7 else '(새 정보)'}")

    print("\n===== 작업2: AUC(down) — 47 vs 47+10 vs 10단독 (OOS) =====")
    for hn,frv in [('4h',fr4),('30m',fr30)]:
        valid=~np.isnan(frv)&(frv!=0)&~np.isnan(Xold).any(1)&~np.isnan(Xnew_raw).any(1)
        idx=np.where(valid)[0]
        y=(frv[idx]<0).astype(int)
        tr=np.array([qtr[i] in TRAIN_Q for i in idx]);te=~tr
        Xo=Xold[idx];Xn=Xnew_raw[idx]
        # robust clip (train pct) + z
        def prep(X):
            lo=np.percentile(X[tr],1,0);hi=np.percentile(X[tr],99,0)
            Xc=np.clip(X,lo,hi);mu=Xc[tr].mean(0);sd=Xc[tr].std(0)+1e-9;return (Xc-mu)/sd
        Zo=prep(Xo);Zn=prep(Xn);Zall=np.hstack([Zo,Zn])
        print(f"  -- {hn}: n={len(idx)} (tr {tr.sum()}/te {te.sum()}), base P(down)={y.mean():.3f}")
        res={}
        for nm,Z in [('47old',Zo),('47+10',Zall),('10new',Zn)]:
            lr=LogisticRegression(max_iter=2000,C=0.5).fit(Z[tr],y[tr])
            a_lr=roc_auc_score(y[te],lr.predict_proba(Z[te])[:,1])
            res[nm]=a_lr
        # GBM 47 vs 47+10 (비선형 최대자유)
        Xall=np.hstack([Xo,Xn])
        go=HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.05,l2_regularization=1.0,random_state=0).fit(Xo[tr],y[tr])
        ga=HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.05,l2_regularization=1.0,random_state=0).fit(Xall[tr],y[tr])
        ag_o=roc_auc_score(y[te],go.predict_proba(Xo[te])[:,1])
        ag_a=roc_auc_score(y[te],ga.predict_proba(Xall[te])[:,1])
        print(f"     로지스틱 AUC: 47old {res['47old']:.4f} | 47+10 {res['47+10']:.4f} (Δ{res['47+10']-res['47old']:+.4f}) | 10new단독 {res['10new']:.4f}")
        print(f"     GBM      AUC: 47old {ag_o:.4f} | 47+10 {ag_a:.4f} (Δ{ag_a-ag_o:+.4f})")
        if hn=='4h':
            # 작업3+4: 47+10 GBM 확신상위 10% short OOS
            p=ga.predict_proba(Xall[te])[:,1];thr=np.percentile(p,90);sel=p>=thr
            frte=frv[idx][te];dq=M['day'].map(dix).to_numpy()[idx][te]
            net=(-1)*frte[sel]*1e4-FEE
            dm=pd.Series(net).groupby(dq[sel]).mean()
            bs=np.random.default_rng(7).choice(dm.to_numpy(),(4000,len(dm)),replace=True).mean(1)
            lo,hi=np.percentile(bs,[2.5,97.5])
            print(f"     [거래화] 47+10 GBM 확신상위10% short: n={sel.sum()} hit {(frte[sel]<0).mean():.3f} "
                  f"net {net.mean():+.1f} dayCI[{lo:+.0f},{hi:+.0f}] {'fee초과+OOS' if net.mean()>0 and lo>0 else 'X'}")

    print("\n  판정: AUC Δ(47→47+10) ≈ 0 이면 하락=시장 본질(라벨 문제 아님). 유의상승+거래화 생존이면 라벨부족.")

if __name__=='__main__':
    main()
