"""Reproducible circle-learning ablations with endpoint metrics and frozen replay."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from rokae_cerebellum_6dof import Config, default_urdf, train
from train_rokae_xy_circle import make_circle
from plot_rokae_end_effector import load_chain, forward


def endpoint_metrics(directory, urdf):
    chain, _ = load_chain(urdf, 'xMateSR3C_link6')
    groups = {}
    with (directory/'joint_tracking.csv').open() as f:
        for row in csv.DictReader(f):
            groups.setdefault(int(row['epoch']), {}).setdefault(int(row['sample']), []).append(row)
    metrics = []
    for epoch, samples in sorted(groups.items()):
        actual, target = [], []
        for _, rows in sorted(samples.items()):
            rows.sort(key=lambda r: int(r['joint']))
            if [int(r['joint']) for r in rows] != list(range(1,7)):
                raise ValueError('Incomplete joint trace')
            actual.append(forward(chain, [float(r['q_rad']) for r in rows]))
            target.append(forward(chain, [float(r['q_des_rad']) for r in rows]))
        actual, target = np.array(actual), np.array(target)
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite trajectory')
        error = np.linalg.norm(actual-target, axis=1)*1000
        metrics.append(dict(epoch=epoch, rmse_mm=float(np.sqrt(np.mean(error**2))),
                            max_mm=float(error.max()), final_error_mm=float(error[-1]),
                            gap_mm=float(np.linalg.norm(actual[-1]-actual[0])*1000)))
    with (directory/'endpoint_epochs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['baseline', 'feedback', 'slow', 'local', 'local_slow'], required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--urdf', type=Path, default=default_urdf())
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', type=Path, help='Previous experiment directory; retain topology and load final weights')
    p.add_argument('--ltp-scale', type=float, default=1.)
    p.add_argument('--ltd-scale', type=float, default=1.)
    p.add_argument('--cf-angle-norm', type=float)
    p.add_argument('--nn', type=int)
    p.add_argument('--nv', type=int)
    p.add_argument('--population', type=int)
    p.add_argument('--arm-kp-scale', type=float, default=1.)
    p.add_argument('--cf-position-mix', type=float)
    p.add_argument('--pd-assist', action='store_true', help='Explicit opt-in to PD, including when resuming old PD weights')
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory to preserve previous experiments')
    trajectory, _, _, metadata = make_circle(args.urdf, .05, 4., .02)
    cfg = Config(epochs=args.epochs, samples_per_epoch=len(trajectory))
    if args.case != 'baseline':
        cfg = replace(cfg, feedback_sample_count=50, cf_angle_norm_deg=5.,
                      cf_velocity_norm_deg_s=45., cf_mix_position=.8)
    if args.case in ('slow', 'local_slow'):
        cfg = replace(cfg, alpha=cfg.alpha*.1, beta=cfg.beta*.1)
    if args.case.startswith('local'):
        q = np.rad2deg(np.array([v[0] for v in trajectory]))
        qd = np.rad2deg(np.array([v[1] for v in trajectory]))
        vmax = np.maximum(np.max(np.abs(qd),axis=0)*1.5, 15.)
        cfg = replace(cfg, nn=6, nv=3, angle_min_deg=tuple(q.min(0)-10.),
                      angle_max_deg=tuple(q.max(0)+10.), velocity_min_deg_s=tuple(-vmax),
                      velocity_range_deg_s=tuple(2*vmax))
    if args.resume:
        if args.nn or args.nv or args.population:
            p.error('Do not change topology while restoring weights')
        cfg = Config(**json.loads((args.resume/'train'/'config.json').read_text()))
        cfg = replace(cfg, epochs=args.epochs)
    cfg = replace(cfg, pd_assist=args.pd_assist,
                  alpha=cfg.alpha*args.ltp_scale, beta=cfg.beta*args.ltd_scale,
                  cf_angle_norm_deg=args.cf_angle_norm or cfg.cf_angle_norm_deg,
                  cf_mix_position=cfg.cf_mix_position if args.cf_position_mix is None else args.cf_position_mix,
                  pd_kp=tuple(k*(args.arm_kp_scale if i < 3 else 1.) for i,k in enumerate(cfg.pd_kp)))
    if args.nn:
        cfg = replace(cfg, nn=args.nn)
    if args.nv:
        cfg = replace(cfg, nv=args.nv)
    if args.population:
        # Keep the maximum decoded torque and total CF count comparable.
        cfg = replace(cfg, n_pc=args.population, n_cf=args.population, n_dcn=args.population,
                      torque_gain=tuple(g*cfg.n_dcn/args.population for g in cfg.torque_gain),
                      spike_cf_max=cfg.spike_cf_max*cfg.n_cf/args.population)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'trajectory.json').write_text(json.dumps(metadata, indent=2))
    train(cfg, args.urdf, args.output_dir/'train', trajectory=trajectory,
          initial_weights=args.resume/'train'/'rokae_cerebellum_weights.txt' if args.resume else None)
    trained = endpoint_metrics(args.output_dir/'train', args.urdf)
    frozen_cfg = replace(cfg, epochs=2, alpha=0., beta=0.)
    train(frozen_cfg, args.urdf, args.output_dir/'frozen', trajectory=trajectory,
          initial_weights=args.output_dir/'train'/'rokae_cerebellum_weights.txt')
    frozen = endpoint_metrics(args.output_dir/'frozen', args.urdf)
    train(frozen_cfg, args.urdf, args.output_dir/'untrained', trajectory=trajectory)
    untrained = endpoint_metrics(args.output_dir/'untrained', args.urdf)
    result = dict(case=args.case, resumed_from=str(args.resume) if args.resume else None,
                  first=trained[0], last=trained[-1],
                  last5_rmse_mm=float(np.mean([r['rmse_mm'] for r in trained[-5:]])),
                  frozen=frozen, untrained=untrained)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
