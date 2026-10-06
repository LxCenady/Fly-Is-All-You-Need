import numpy as np
exec(open('measure.py').read().split("r=np.random.default_rng(0);perm")[0])   # reuse network, patches, run(), feats(), decode()
r=np.random.default_rng(0);perm=r.permutation(N);ro=perm[:200]
NT=1650
for g in np.arange(1.5,7.01,.5):
    sym=seq('iid',NT,np.random.default_rng(1));SP=run(g,1,sym)
    act=SP.mean()
    # coverage: fraction of readout neurons that spike at least once in a token window
    W=SP[:NT*5].reshape(NT,5,N)[:,:,ro].any(1).mean()
    # input-free persistence: keep simulating 40 steps with no drive, from the final state
    v=np.full(N,.25);s=np.zeros(N,bool);we=.5/12;w=np.where(rod,-we*5,we)
    for t in range(NT*5):   # replay to final state cheaply: reuse SP last step as state proxy
        pass
    last=SP[-1].astype(bool)
    # rerun 60 free steps starting from the recorded last spikes (v reset proxy)
    s=last.copy();v=np.full(N,.5);free=[]
    for t in range(60):
        I=np.bincount(tgt[s].ravel(),weights=np.repeat(w[s],12),minlength=N)
        v=.8*v+g*I+.05;v=np.maximum(v,-1);s=v>=1;v[s]=0;free.append(s.mean())
    # branching: spikes(t+1)/spikes(t) averaged where spikes(t)>5
    n=SP.sum(1);m=n[:-1]>5;sig=(n[1:][m]/n[:-1][m]).mean()
    acc=decode(feats(SP,ro,'counts',0,NT),sym)
    print(f"g={g:.1f} act={act*100:5.1f}% cover={W:.3f} sigma={sig:.2f} free40-60={np.mean(free[40:])*100:5.1f}% k1={acc[1]:.2f} k2={acc[2]:.2f}")
