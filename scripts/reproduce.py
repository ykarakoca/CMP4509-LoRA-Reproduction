"""Run independent repetitions in fresh processes; never modify frozen evidence."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['full','smoke'],default='full')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--strict-env',action='store_true',help='Require original package versions and Tesla T4')
 args=p.parse_args(); root=args.output.resolve()
 assert root != R and R not in root.parents, 'Keep generated runs outside the repository.'
 root.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,LORA_RUN_ROOT=str(root),LORA_MODE=args.mode,LORA_STRICT_ENV=str(int(args.strict_env)))
 proto=json.loads((R/'configs/protocol.json').read_text())
 pairs=[(m,s) for m in proto['methods'] for s in (proto['seeds'] if args.mode=='full' else [42])]
 for method,seed in pairs:
  subprocess.run([sys.executable,'-u',str(R/'src/train_worker.py'),method,str(seed)],env=env,check=True)
 if args.mode=='smoke':
  print('SMOKE COMPLETE: four 64-example training checks; no final evaluation or scientific results.'); return
 entries=[]
 for method,seed in pairs:
  candidates=[]
  for path in (root/'official').glob('*/result.json'):
   r=json.loads(path.read_text())
   if (r['config']['method'],r['config']['seed'])==(method,seed):candidates.append((path,r))
  assert len(candidates)==1,(method,seed,'ambiguous or missing run')
  path,r=candidates[0];assert r['status']=='completed' and r['reload']['verified']
  cp=r['best']['checkpoint'];assert sha(cp['path'])==cp['sha256']
  entries.append(dict(method=method,seed=seed,selected_epoch=r['best']['epoch'],
      checkpoint_path=cp['path'],checkpoint_sha256=cp['sha256'],checkpoint_size_bytes=cp['size_bytes'],
      result_path=str(path),result_sha256=sha(path)))
 index={'protocol_sha256':sha(R/'configs/protocol.json'),'checkpoint_count':12,'checkpoints':entries}
 dest=root/'official_selected_checkpoints.json';encoded=json.dumps(index,indent=2)
 if dest.exists(): assert dest.read_text()==encoded,'Existing fixed checkpoint index differs'
 else: dest.write_text(encoded)
 subprocess.run([sys.executable,'-u',str(R/'src/evaluate_runs.py')],env=env,check=True)
 subprocess.run([sys.executable,str(R/'scripts/analyze.py'),'--runs',str(root),'--output',str(root/'analysis')],check=True)
if __name__=='__main__':main()
