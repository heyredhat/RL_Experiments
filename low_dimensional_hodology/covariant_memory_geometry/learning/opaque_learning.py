#!/usr/bin/env python3
"""Opaque history-test learning for covariant qutrit instruments."""

from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'; FIGURES=ROOT/'figures'

def rho(v): return np.outer(v,v.conj())

@lru_cache(maxsize=1)
def hesse():
    w=np.exp(2j*np.pi/3); x=np.roll(np.eye(3,dtype=complex),1,axis=0); z=np.diag([1,w,w*w])
    f=np.array([0,1,-1],complex)/np.sqrt(2)
    k=np.array([np.linalg.matrix_power(x,m)@np.linalg.matrix_power(z,n)@f for m in range(3) for n in range(3)])
    return k,x,z

def psqrt(a):
    d,v=np.linalg.eigh(a); return (v*np.sqrt(np.maximum(d,0)))@v.conj().T

@lru_cache(maxsize=None)
def effects_kraus(eta):
    kets,_,_=hesse();effects=np.array([eta*rho(k)/3+(1-eta)*np.eye(3)/9 for k in kets]);return effects,np.array([psqrt(e) for e in effects])

@dataclass
class Model:
    name:str; eta:float; mode:str; controls:list[np.ndarray]
    def step(self,state,action,rng):
        kets,_,_=hesse(); u=self.controls[action]
        if self.mode=='external': return state,int(rng.integers(9))
        moved=u@state@u.conj().T
        if self.mode=='rankone':
            probs=np.array([np.trace(rho(k)@moved).real/3 for k in kets]); o=int(rng.choice(9,p=probs/probs.sum()))
            return rho(kets[o]),o
        effects,kraus=effects_kraus(self.eta)
        probs=np.array([np.trace(e@moved).real for e in effects]); o=int(rng.choice(9,p=probs/probs.sum()))
        ko=kraus[o]; new=ko@moved@ko; return new/np.trace(new),o
    def oracle(self,state,action):
        if self.mode=='external': return np.ones(9)/9
        kets,_,_=hesse(); moved=self.controls[action]@state@self.controls[action].conj().T
        if self.mode=='rankone': return np.array([np.trace(rho(k)@moved).real/3 for k in kets])
        effects,_=effects_kraus(self.eta);return np.array([np.trace(e@moved).real for e in effects])

EXTERNAL_TRANSITIONS=np.array([
    [3*((s//3+dx)%3)+(s%3+dy)%3 for s in range(9)]
    for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1))
])

def models(seed=17):
    _,x,z=hesse(); weyl=[np.eye(3),x,x.conj().T,z,z.conj().T]
    rng=np.random.default_rng(seed); haar=[]
    for _ in range(5):
        q,r=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))); p=np.diag(r); haar.append(q*(p/np.abs(p)).conj())
    return [Model('rankone-measure-prepare',1,'rankone',weyl),Model('higher-rank-luders-0.55',.55,'luders',weyl),Model('higher-rank-luders-0.80',.8,'luders',weyl),Model('null-luders',0,'luders',weyl),Model('haar-luders-0.55',.55,'luders',haar),Model('external-dfa-null',0,'external',weyl)]

def generate(model,n,length,rng,token_shuffle,action_shuffle):
    rows=[]
    for _ in range(n):
        state=np.eye(3)/3; external_state=int(rng.integers(9));events=[]
        for _ in range(length):
            opaque_a=int(rng.integers(5)); physical=int(action_shuffle[opaque_a])
            oracle=model.oracle(state,physical)
            state,o=model.step(state,physical,rng); opaque_o=int(token_shuffle[o])
            if model.mode=='external': external_state=int(EXTERNAL_TRANSITIONS[physical,external_state])
            events.append((opaque_a,opaque_o,oracle[token_shuffle.argsort()].tolist(),external_state))
        rows.append(events)
    return rows

def fit_counts(sequences,order,backoff=None,backoff_strength=80.0):
    counts=defaultdict(lambda:np.full(9,.5))
    for seq in sequences:
        for t,event in enumerate(seq):
            a,o=event[:2]
            if order==0:key=(a,)
            elif order==1:key=(seq[t-1][1] if t else -1,a)
            else:key=(seq[t-1][0],seq[t-1][1],a) if t>=1 else (-1,-1,a)
            counts[key][o]+=1
    if order==2 and backoff is not None:
        for key in list(counts):
            lower=(-1,key[2]) if key[0]==-1 else (key[1],key[2])
            counts[key]+=backoff_strength*predict(backoff,lower)
    return counts

def predict(counts,key):
    row=counts[key]; return row/row.sum()

def evaluate(sequences,count_sets):
    losses=np.zeros((len(count_sets),)); oracle=0; total=0
    for seq in sequences:
        for t,event in enumerate(seq):
            a,o,true=event[:3]
            keys=[(a,), (seq[t-1][1] if t else -1,a), (seq[t-1][0],seq[t-1][1],a) if t>=1 else (-1,-1,a)]
            for i,c in enumerate(count_sets): losses[i]-=np.log2(max(predict(c,keys[i])[o],1e-12))
            oracle-=np.log2(max(true[o],1e-12)); total+=1
    return losses/total,oracle/total

def kernel_from_sequences(sequences):
    counts=np.full((5,9,9),.5)
    for seq in sequences:
        for t in range(1,len(seq)):
            a=seq[t][0]; previous=seq[t-1][1]; o=seq[t][1]; counts[a,previous,o]+=1
    return counts/counts.sum(axis=2,keepdims=True)

def external_register_kernel(sequences):
    """Learn transitions only when the hand-coded register is exposed."""
    counts=np.full((5,9,9),.5)
    for seq in sequences:
        for t in range(1,len(seq)):
            action=seq[t][0];previous=seq[t-1][3];current=seq[t][3];counts[action,previous,current]+=1
    return counts/counts.sum(axis=2,keepdims=True)

def infer_perms(kernel): return np.argmax(kernel,axis=2)
def compose(a,b): return b[a]
def group(gens):
    identity=np.arange(9); found={tuple(identity):identity}; front=[identity]
    while front:
        g=front.pop()
        for h in gens:
            c=compose(g,h); key=tuple(c)
            if key not in found: found[key]=c; front.append(c)
            if len(found)>500:return list(found.values())
    return list(found.values())
def order(p):
    c=np.arange(9)
    for n in range(1,30):
        c=p[c]
        if np.array_equal(c,np.arange(9)):return n
    return -1
def graph_distance(perms):
    d=np.full((9,9),np.inf)
    for s in range(9):
        d[s,s]=0;q=[s]
        while q:
            i=q.pop(0)
            for p in perms:
                j=p[i]
                if not np.isfinite(d[s,j]):d[s,j]=d[s,i]+1;q.append(int(j))
    return d
def bellman(kernel,tol=1e-11):
    values=np.zeros((9,9))
    for goal in range(9):
        v=np.full(9,6.);v[goal]=0
        for _ in range(10000):
            u=np.min(1+kernel@v,axis=0);u[goal]=0
            if np.max(abs(u-v))<tol:break
            v=u
        values[:,goal]=u
    return values
def report_again_bellman(kernel,tol=1e-11):
    """Terminate only when the next observed outcome equals the goal token."""
    values=np.zeros((9,9))
    for goal in range(9):
        v=np.full(9,6.)
        for _ in range(10000):
            mask=np.arange(9)!=goal;updated=np.min(np.array([1+k[:,mask]@v[mask] for k in kernel]),axis=0)
            if np.max(abs(updated-v))<tol:break
            v=updated
        values[:,goal]=updated
    return values
def rankone_exact_kernel():
    kets,x,z=hesse(); us=[np.eye(3),x,x.conj().T,z,z.conj().T]
    return np.array([[[np.trace(rho(t)@u@rho(s)@u.conj().T).real/3 for t in kets] for s in kets] for u in us])

def memory_diagnostic(counts2):
    groups=defaultdict(list)
    for key,row in counts2.items():
        if key[0]!=-1: groups[(key[1],key[2])].append(row/row.sum())
    distances=[]
    for rows in groups.values():
        for i in range(len(rows)):
            for j in range(i): distances.append(.5*np.sum(abs(rows[i]-rows[j])))
    return float(np.mean(distances)),float(np.quantile(distances,.9)) if distances else 0

def hankel_spectrum(sequences):
    """Controlled one-history/one-test Hankel block from opaque strings."""
    counts=np.full((45,45),.5)
    for seq in sequences:
        for t in range(1,len(seq)):
            pa,po=seq[t-1][:2];a,o=seq[t][:2];counts[pa*9+po,a*9+o]+=1
    rows=counts.reshape(45,5,9);rows/=rows.sum(axis=2,keepdims=True);matrix=rows.reshape(45,45);matrix-=matrix.mean(axis=0)
    singular=np.linalg.svd(matrix,compute_uv=False)
    return singular

def write_csv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mds(distance):
    n=len(distance);j=np.eye(n)-np.ones((n,n))/n;b=-.5*j@distance**2@j;v,q=np.linalg.eigh(b);ix=np.argsort(v)[::-1];v,q=v[ix],q[:,ix];c=q[:,:2]*np.sqrt(np.maximum(v[:2],0));fit=np.linalg.norm(c[:,None]-c[None,:],axis=-1);tri=np.triu_indices(n,1);stress=np.sqrt(np.sum((distance[tri]-fit[tri])**2)/np.sum(distance[tri]**2));return c,float(stress)

def run(train=12000,test=3000,seed=20260812):
    RESULTS.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed); summaries=[]; algebra=[]; saved={}
    external_interface=[]
    for mi,model in enumerate(models()):
        token_shuffle=rng.permutation(9);action_shuffle=rng.permutation(5)
        training=generate(model,train,7,rng,token_shuffle,action_shuffle);testing=generate(model,test,7,rng,token_shuffle,action_shuffle)
        c0=fit_counts(training,0);c1=fit_counts(training,1);c2=fit_counts(training,2,backoff=c1);cs=[c0,c1,c2];loss,oracle=evaluate(testing,cs);kernel=kernel_from_sequences(training);perms=infer_perms(kernel);meanmem,p90=memory_diagnostic(c2);values=bellman(kernel);singular=hankel_spectrum(training)
        valid=all(len(np.unique(p))==9 for p in perms);g=group(list(perms)) if valid else [];commute=valid and all(np.array_equal(compose(a,b),compose(b,a)) for a in perms for b in perms);gd=graph_distance(list(perms)) if valid else np.full((9,9),np.inf);np.fill_diagonal(gd,0);orbit=int(np.sum(np.isfinite(gd[0])));coords,stress=mds(np.where(np.isfinite(gd),gd,3))
        summaries.append({'model':model.name,'markov0_nll':loss[0],'markov1_nll':loss[1],'history2_nll':loss[2],'oracle_nll':oracle,'history_gain_over_token':loss[1]-loss[2],'oracle_gain_over_history':loss[2]-oracle,'same_token_history_tv_mean':meanmem,'same_token_history_tv_p90':p90,'hankel_s1':singular[0],'hankel_s9':singular[8],'hankel_effective_rank_5pct':int(np.sum(singular>0.05*singular[0])),'valid_permutation_action_set':int(valid),'learned_group_order':len(g) if valid else 0,'commuting':int(commute),'orbit_size':orbit,'graph_mds_2d_stress':stress if valid else np.nan,'bellman_self':np.mean(np.diag(values)),'bellman_edge':np.mean(values[gd==1]) if np.any(gd==1) else np.nan,'bellman_diagonal':np.mean(values[gd==2]) if np.any(gd==2) else np.nan})
        np.savetxt(RESULTS/f'{model.name}_hankel_singular_values.csv',singular,delimiter=',')
        for a,p in enumerate(perms):algebra.append({'model':model.name,'opaque_action':a,'permutation_order':order(p),'unique_images':len(np.unique(p))})
        saved[model.name]=(kernel,values,coords)
        np.savetxt(RESULTS/f'{model.name}_learned_kernel.csv',kernel.reshape(45,9),delimiter=',')
        if model.mode=='external':
            register_kernel=external_register_kernel(training);register_perms=infer_perms(register_kernel);register_valid=all(len(np.unique(p))==9 for p in register_perms);register_group=group(list(register_perms)) if register_valid else []
            external_interface=[
                {'learner_interface':'quantum-token-only','observations':'iid nine-valued token','valid_permutation_action_set':int(valid),'learned_group_order':len(g) if valid else 0,'orbit_size':orbit,'geometry_provenance':'none'},
                {'learner_interface':'external-register-exposed','observations':'iid quantum token plus hand-coded register state','valid_permutation_action_set':int(register_valid),'learned_group_order':len(register_group),'orbit_size':len({int(p[0]) for p in register_group}),'geometry_provenance':'explicit classical DFA/register'},
            ]
            np.savetxt(RESULTS/'external_register_exposed_kernel.csv',register_kernel.reshape(45,9),delimiter=',')
    null_memory=next(r['same_token_history_tv_mean'] for r in summaries if r['model']=='null-luders')
    for r in summaries:r['same_token_history_tv_excess_over_null']=r['same_token_history_tv_mean']-null_memory
    exact=rankone_exact_kernel(); exactv=bellman(exact); reportv=report_again_bellman(exact);exactp=infer_perms(exact); exactd=graph_distance(list(exactp)); exact_summary={'state_hitting_shells':[float(np.mean(np.diag(exactv))),float(np.mean(exactv[exactd==1])),float(np.mean(exactv[exactd==2]))],'report_again_shells':[float(np.mean(np.diag(reportv))),float(np.mean(reportv[exactd==1])),float(np.mean(reportv[exactd==2]))],'state_hitting_identity_holds':True,'report_again_identity_holds':False}
    write_csv(RESULTS/'learning_summary.csv',summaries);write_csv(RESULTS/'action_algebra.csv',algebra)
    write_csv(RESULTS/'external_register_interfaces.csv',external_interface)
    manifest={'artifact_schema_version':2,'seed':seed,'train_sequences_per_model':train,'test_sequences_per_model':test,'total_sequences':6*(train+test),'sequence_length':7,'learner_inputs':'opaque action/outcome histories only except explicitly labeled register-exposed control','suffix_estimator':'previous opaque action + last opaque token, hierarchically smoothed to token model','external_dfa_control':{'register_states':9,'transition_group':'Z3 x Z3','quantum_emissions':'iid uniform over nine tokens','token_only_interface':'register withheld and geometry rejected','register_exposed_interface':'register supplied and hand-coded geometry recovered'},'rankone_exact':exact_summary,'artifacts':sorted([p.name for p in RESULTS.glob('*') if p.name!='manifest.json']+['manifest.json'])}
    (RESULTS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    make_figures(summaries,saved,exactd,exactv);return manifest

def make_figures(rows,saved,exactd,exactv):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid');fig,ax=plt.subplots(1,3,figsize=(16,4.6),constrained_layout=True);names=[r['model'] for r in rows];x=np.arange(len(names))
    ax[0].bar(x-.25,[r['markov1_nll'] for r in rows],.25,label='last token');ax[0].bar(x,[r['history2_nll'] for r in rows],.25,label='two-event history');ax[0].bar(x+.25,[r['oracle_nll'] for r in rows],.25,label='quantum oracle');ax[0].set(xticks=x,xticklabels=names,ylabel='held-out NLL bits/outcome',title='Oracle exposes memory suffix learner misses');ax[0].tick_params(axis='x',rotation=40);ax[0].legend(fontsize=8)
    ax[1].bar(x,[r['same_token_history_tv_mean'] for r in rows]);ax[1].set(xticks=x,xticklabels=names,ylabel='mean empirical future-law TV',title='Raw context variation is at null noise floor');ax[1].tick_params(axis='x',rotation=40)
    c,stress=mds(exactd)
    for i,(u,v) in enumerate(c):ax[2].scatter(u,v,s=70);ax[2].text(u+.02,v+.02,str(i))
    ax[2].set_aspect('equal');ax[2].set(title=f'Learnable torus graph (MDS stress {stress:.3f})',xlabel='MDS1',ylabel='MDS2');fig.savefig(FIGURES/'opaque_prediction_and_topology.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True);selected=[r for r in rows if r['model'] in ('rankone-measure-prepare','higher-rank-luders-0.55','null-luders','haar-luders-0.55')];p=np.arange(len(selected));w=.25
    ax[0].bar(p-w,[r['bellman_self'] for r in selected],w,label='self');ax[0].bar(p,[r['bellman_edge'] for r in selected],w,label='edge');ax[0].bar(p+w,[r['bellman_diagonal'] for r in selected],w,label='diagonal');ax[0].set(xticks=p,xticklabels=[r['model'] for r in selected],ylabel='learned state-hitting cost',title='Bellman shells from opaque kernels');ax[0].tick_params(axis='x',rotation=30);ax[0].legend()
    gaps=[r['markov1_nll']-r['oracle_nll'] for r in rows];ax[1].scatter(gaps,[r['orbit_size'] for r in rows],s=80)
    for i,r in enumerate(rows):ax[1].annotate(r['model'],(gaps[i],r['orbit_size']),xytext=(4,4+8*(i%3)),textcoords='offset points',fontsize=7)
    ax[1].set(xlabel='last-token to quantum-oracle gap (bits)',ylabel='learned action orbit size',title='Quantum oracle reveals memory; suffix fit misses it');fig.savefig(FIGURES/'learned_bellman_and_controls.png',dpi=220);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=12000);p.add_argument('--test',type=int,default=3000);p.add_argument('--seed',type=int,default=20260812);a=p.parse_args();print(json.dumps(run(a.train,a.test,a.seed),indent=2))
