import numpy as np,sys,time
G=48
cells=np.array([(x,y) for y in range(G) for x in range(G) if (x-23.5)**2+(y-23.5)**2<576]);N=len(cells)
idx=-np.ones((G,G),int);idx[cells[:,1],cells[:,0]]=np.arange(N)
rod=(cells[:,0]%3==1)&(cells[:,1]%3==1)
rng=np.random.default_rng(3);K=12;tgt=np.zeros((N,K),int)
for i,(x,y) in enumerate(cells):
    for k in range(K):
        j=-1
        while j<0 or j==i:
            r=np.sqrt(-2*np.log(rng.random()+1e-9))*2.2;a=rng.random()*6.2832
            tx,ty=round(x+r*np.cos(a)),round(y+r*np.sin(a))
            if 0<=tx<G and 0<=ty<G:j=idx[ty,tx]
        tgt[i,k]=j
S=8
ang=np.arange(S)/S*2*np.pi;loc=np.stack([23.5+12*np.cos(ang),23.5+12*np.sin(ang)],1)
patch=np.array([((cells-l)**2).sum(1)<=9 for l in loc]).astype(float)  # S x N
P=np.full((S,S),.1/6);
for a in range(S):P[a,(a+1)%S]=.45;P[a,(a+3)%S]=.45;P[a,a]=0
P/=P.sum(1,keepdims=True)
def seq(kind,T,r):
    if kind=='iid':return r.integers(S,size=T)
    s=[0]
    for t in range(T-1):s.append(r.choice(S,p=P[s[-1]]))
    s=np.array(s)
    if kind=='shuffled':r.shuffle(s)
    return s
def run(gain,ins,sym,steps_per=5):
    T=len(sym)*steps_per;v=np.full(N,.25);s=np.zeros(N,bool);we=.5/12;wi=-we*5*ins
    w=np.where(rod,wi,we);SP=np.zeros((T,N),np.uint8)
    for t in range(T):
        inp=.5*patch[sym[t//steps_per]]
        I=np.bincount(tgt[s].ravel(),weights=np.repeat(w[s],K),minlength=N)
        v=.8*v+gain*I+.05+inp;v=np.maximum(v,-1);s=v>=1;v[s]=0;SP[t]=s
    return SP
def feats(SP,ro,kind,shift,nt):
    sp=SP[:,ro].astype(float);T=len(sp)
    c=np.zeros((nt,len(ro)));tr=np.zeros((nt,len(ro)));x=np.zeros(len(ro))
    for t in range(T):
        x=.82*x+sp[t];w=(t-shift)//5
        if 0<=w<nt:
            c[w]+=sp[t]
            if (t-shift)%5==4:tr[w]=x
    if kind=='counts':return c
    if kind=='trace':return tr
    if kind=='both':return np.hstack([c,tr])
    if kind=='delay':return np.hstack([c,np.vstack([c[:1]*0,c[:-1]])])
def decode(F,sym,ks=range(6),ntr=1200,lam=1.0):
    F=(F-F[:ntr].mean(0))/(F[:ntr].std(0)+1e-6);F=np.hstack([F,np.ones((len(F),1))])
    out=[]
    for k in ks:
        X=F[k:];y=sym[:len(sym)-k];Y=np.eye(S)[y]
        A=X[:ntr].T@X[:ntr]+lam*np.eye(X.shape[1]);W=np.linalg.solve(A,X[:ntr].T@Y[:ntr])
        out.append((np.argmax(X[ntr:]@W,1)==y[ntr:]).mean())
    return np.array(out)
def baseline(sym,ks=range(6),ntr=1200):
    out=[]
    for k in ks:
        cur=sym[k:];past=sym[:len(sym)-k];C=np.zeros((S,S))
        for a,b in zip(cur[:ntr],past[:ntr]):C[a,b]+=1
        out.append((C.argmax(1)[cur[ntr:]]==past[ntr:]).mean())
    return np.array(out)
r=np.random.default_rng(0);perm=r.permutation(N)
NT=1650
def show(name,gain=4,ins=1,kind='iid',M=200,feat='counts',shift=0):
    sym=seq(kind,NT,np.random.default_rng(1));SP=run(gain,ins,sym)
    acc=decode(feats(SP,perm[:M],feat,shift,NT),sym);b=baseline(sym)
    print(f"{name:28s} act={SP.mean()*100:5.1f}%  acc={np.round(acc,2)}  base={np.round(b,2)}")
t0=time.time()
for M in [10,40,200,400]:show(f"M={M}",M=M)
for f in ['counts','trace','both','delay']:show(f"feat={f}",feat=f)
show("shift=1",shift=1);show("shift=2",shift=2)
for k in ['markov','shuffled']:show(f"seq={k}",kind=k)
for g in [2,3,5,6]:show(f"gain={g}",gain=g)
print(time.time()-t0)
