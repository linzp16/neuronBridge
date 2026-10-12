"""Plot measured tuning results; do not interpolate or replace actual trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/rokae_xy_tune'))
    p.add_argument('--selected',required=True)
    args = p.parse_args()
    summaries=[]
    for path in sorted(args.root.glob('*/summary.json')):
        data=json.loads(path.read_text())
        summaries.append(dict(experiment=path.parent.name,
            training_last_rmse_mm=data['last']['rmse_mm'],
            training_last5_rmse_mm=data['last5_rmse_mm'],
            frozen_rmse_mm=data['frozen'][0]['rmse_mm'],
            frozen_max_mm=data['frozen'][0]['max_mm'],
            frozen_gap_mm=data['frozen'][0]['gap_mm'],
            untrained_rmse_mm=data['untrained'][0]['rmse_mm']))
    with (args.root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    selected=args.root/args.selected
    lineage=[]; current=selected
    while True:
        lineage.append(current)
        parent=json.loads((current/'summary.json').read_text()).get('resumed_from')
        if parent is None:
            break
        current=Path(parent)
    lineage.reverse()
    fig, axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    offset=0
    for path in lineage:
        rows=read_rows(path/'train'/'endpoint_epochs.csv')
        x=np.array([int(r['epoch']) for r in rows])+offset
        axes[0,0].plot(x,[float(r['rmse_mm']) for r in rows],label=path.name)
        if offset:
            axes[0,0].axvline(offset+.5,color='.7',ls=':',lw=1)
        offset=int(x[-1])
    stats=json.loads((selected/'summary.json').read_text())
    axes[0,0].axhline(stats['frozen'][0]['rmse_mm'],color='k',ls='--',label='Final frozen replay')
    axes[0,0].set(xlabel='Cumulative training epoch',ylabel='Endpoint RMSE (mm)',
                  title='Online training (stage changes marked)',yscale='log')
    axes[0,0].legend(fontsize=8);axes[0,0].grid(alpha=.2)
    rows=read_rows(selected/'frozen'/'end_effector'/'positions.csv')
    actual=np.array([[float(r[f'actual_{a}_m']) for a in 'xyz'] for r in rows])
    target=np.array([[float(r[f'target_{a}_m']) for a in 'xyz'] for r in rows])
    sample=[int(r['sample']) for r in rows]
    ax=axes[0,1]
    for points,color,ls,label in [(target,'k','--','Target circle'),(actual,'#0072B2','-','Frozen learned controller')]:
        ax.plot(points[:,0]*1000,points[:,1]*1000,color=color,ls=ls,label=label)
    ax.scatter(*actual[0,:2]*1000,c='#009E73',marker='o',label='First sample',zorder=3)
    ax.scatter(*actual[-1,:2]*1000,c='#D55E00',marker='x',label='Last sample',zorder=3)
    ax.set(xlabel='World x (mm)',ylabel='World y (mm)',title='Actual XY path (equal scale)')
    ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for i,a in enumerate('xyz'):
        axes[1,0].plot(sample,(actual[:,i]-target[:,i])*1000,label=a)
    axes[1,0].axhline(0,color='k',lw=.7)
    axes[1,0].set(xlabel='Control sample',ylabel='Signed position error (mm)',title='Coordinate errors, including height')
    axes[1,0].legend();axes[1,0].grid(alpha=.2)
    error=np.linalg.norm(actual-target,axis=1)*1000
    axes[1,1].plot(sample,error,color='#D55E00')
    axes[1,1].set(xlabel='Control sample',ylabel='Euclidean error (mm)',
                  title=f"Frozen RMSE {stats['frozen'][0]['rmse_mm']:.2f} mm; gap {stats['frozen'][0]['gap_mm']:.2f} mm")
    axes[1,1].grid(alpha=.2)
    cfg=json.loads((selected/'frozen'/'config.json').read_text())
    mode='PD-assisted' if cfg['pd_assist'] else 'Neural only (PD OFF)'
    fig.suptitle(f'ROKAE XY circle | {offset} training epochs | {mode} | {args.selected}')
    for ext in ('png','svg'):
        fig.savefig(args.root/f'convergence_{args.selected}.{ext}',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    names=[r['experiment'] for r in summaries]; x=np.arange(len(names))
    ax.bar(x-.18,[r['frozen_rmse_mm'] for r in summaries],width=.36,label='Frozen RMSE')
    ax.bar(x+.18,[r['frozen_gap_mm'] for r in summaries],width=.36,label='Frozen start/end gap')
    ax.set(xticks=x,xticklabels=names,ylabel='Distance (mm)',yscale='log',title='All tested candidates (different training lengths; see configs)')
    ax.tick_params(axis='x',rotation=35);ax.legend();ax.grid(axis='y',alpha=.2)
    for ext in ('png','svg'):
        fig.savefig(args.root/f'ablation_comparison.{ext}',dpi=200)
    plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
