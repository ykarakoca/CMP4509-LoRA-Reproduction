"""Audit saved predictions and regenerate tables/figures without a GPU."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
METHODS=['full','head','lora2','lora8'];NAMES=['Full FT','Head-only','LoRA r=2','LoRA r=8'];SEEDS=[17,42,123]
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();assert out!=R and R not in out.parents,'Write generated analysis outside repository'
 out.mkdir(parents=True,exist_ok=True)
 if not a.runs:
  integrity=read(R/'PUBLIC_SHA256SUMS.json')
  for name,digest in integrity.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
 records=[];histories={};reference=None;max_error=0
 manifest=read(R/'configs/sst2_split_manifest.json');ids=manifest['splits']['final_evaluation']['ids_in_order']
 for m in METHODS:
  for seed in SEEDS:
   tag=f'{m}_seed{seed}'
   if a.runs:
    matches=list((a.runs/'official').glob(tag+'_*/result.json'));assert len(matches)==1
    train=read(matches[0]);base=a.runs/'final_evaluation'
   else:train=read(R/'results/frozen'/f'{tag}_training.json');base=R/'results/frozen'
   final=read(base/f'{tag}.json');rows=read(base/f'{tag}_predictions.json')
   assert hashlib.sha256((base/f'{tag}_predictions.json').read_bytes()).hexdigest()==final['predictions_sha256']
   assert train['status']=='completed' and train['reload']['verified']
   assert len(rows)==872 and [r['idx'] for r in rows]==ids
   labels=[r['label'] for r in rows]
   if reference is None:reference=labels
   assert labels==reference
   logits=np.array([r['logits'] for r in rows],dtype=np.float64);assert np.isfinite(logits).all()
   pred=logits.argmax(1);assert pred.tolist()==[r['prediction'] for r in rows]
   assert set(labels)=={0,1}
   cm=np.zeros((2,2),dtype=int)
   for y,yhat in zip(labels,pred):cm[y,yhat]+=1
   acc=float(np.mean(pred==labels));mx=logits.max(1)
   loss=float(np.mean(mx+np.log(np.exp(logits-mx[:,None]).sum(1))-logits[np.arange(872),labels]))
   metrics=final['metrics'];assert acc==metrics['accuracy'] and cm.tolist()==metrics['confusion_matrix']
   assert math.isclose(loss,metrics['loss'],rel_tol=1e-6,abs_tol=1e-7)
   max_error=max(max_error,abs(loss-metrics['loss']))
   best=min(train['epochs'],key=lambda e:(-e['tuning']['accuracy'],e['tuning']['loss'],e['epoch']))
   assert best['epoch']==train['best']['epoch']==final['checkpoint']['selected_epoch']
   assert len(train['epochs'])==5 and train['planned_optimizer_steps']==1565
   histories[tag]=train['epochs']
   records.append(dict(method=m,seed=seed,accuracy=acc,loss=metrics['loss'],trainable_parameters=train['trainable_parameters'],
    training_minutes=sum(e['training']['training_seconds'] for e in train['epochs'])/60,
    peak_allocated_gib=max(e['training']['peak_allocated_bytes'] for e in train['epochs'])/2**30))
 aggregates=[]
 for m in METHODS:
  group=[r for r in records if r['method']==m];v={'method':m,'n_seeds':3,'trainable_parameters':group[0]['trainable_parameters']}
  for k in ['accuracy','loss','training_minutes','peak_allocated_gib']:
   v[k+'_mean']=statistics.mean(r[k] for r in group);v[k+'_sample_std']=statistics.stdev(r[k] for r in group)
  aggregates.append(v)
 if not a.runs:
  with (R/'results/original_analysis/summary.csv').open() as f:original={r['method']:r for r in csv.DictReader(f)}
  for row in aggregates:
   for key,value in row.items():
    if key!='method':assert math.isclose(float(original[row['method']][key]),value,rel_tol=1e-12,abs_tol=1e-12),(row['method'],key)
 for name,data in [('per_run',records),('summary',aggregates)]:
  (out/f'{name}.json').write_text(json.dumps(data,indent=2))
  with (out/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 x=np.arange(4);fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
 for ax,metric,label,factor in zip(axes,['accuracy','training_minutes','peak_allocated_gib'],['Final accuracy (%)','Training time (min)','Peak allocated memory (GiB)'],[100,1,1]):
  means=[v[metric+'_mean']*factor for v in aggregates];std=[v[metric+'_sample_std']*factor for v in aggregates]
  ax.errorbar(x,means,yerr=std,fmt='D',capsize=5,label='Mean ± sample SD')
  for j,m in enumerate(METHODS):ax.scatter(j+np.linspace(-.12,.12,3),[r[metric]*factor for r in records if r['method']==m],s=16,c='black',label='Individual seeds' if j==0 else None)
  ax.set_xticks(x,NAMES,rotation=20);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('Independent repetition' if a.runs else 'Frozen Stage 5 results — 3 seeds')
 fig.savefig(out/'accuracy_efficiency.pdf');fig.savefig(out/'accuracy_efficiency.png');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for ax,m,name in zip(axes.flat,METHODS,NAMES):
  for seed in SEEDS:ax.plot(range(1,6),[e['tuning']['accuracy']*100 for e in histories[f'{m}_seed{seed}']],marker='o',label=f'Seed {seed}')
  ax.set(title=name,xlabel='Epoch',ylabel='Tuning accuracy (%)',xticks=range(1,6));ax.legend();ax.grid(alpha=.2)
 fig.savefig(out/'tuning_curves.png');plt.close(fig)
 report={'status':'PASS','runs':12,'predictions':10464,'maximum_loss_recalculation_error':max_error,'scope':'Saved evidence audit; does not load weights or independently verify labels against upstream data'}
 (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
