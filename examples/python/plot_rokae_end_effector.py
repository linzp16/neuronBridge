"""Reconstruct end-effector positions from measured joint CSV and URDF."""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


def load_chain(path, frame):
    root = ET.parse(path).getroot()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    chain = []
    link = frame
    while link in by_child:
        joint = by_child[link]
        chain.append(joint)
        link = joint.find('parent').get('link')
    if not chain:
        raise ValueError(f'No chain to {frame}')
    chain.reverse()
    active = [j for j in chain if j.get('type') != 'fixed']
    if [j.get('name') for j in active] != [f'xmate_joint_{i}' for i in range(1, 7)]:
        raise ValueError('Unexpected joint order: verify CSV-to-URDF mapping')
    return chain, link


def pose(chain, q):
    transform = np.eye(4)
    index = 0
    for joint in chain:
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        local = np.eye(4)
        local[:3, 3] = xyz
        local[:3, :3] = rotation([0,0,1], rpy[2]) @ rotation([0,1,0], rpy[1]) @ rotation([1,0,0], rpy[0])
        transform = transform @ local
        if joint.get('type') != 'fixed':
            if joint.get('type') not in ('revolute', 'continuous'):
                raise ValueError('Only fixed/revolute joints supported')
            element = joint.find('axis')
            axis = np.fromstring(element.get('xyz'), sep=' ') if element is not None else [1,0,0]
            motion = np.eye(4)
            motion[:3,:3] = rotation(axis, q[index])
            transform = transform @ motion
            index += 1
    return transform


def forward(chain, q):
    return pose(chain, q)[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/rokae_joint_gains'))
    parser.add_argument('--urdf', type=Path, default=Path(r'D:\Pinocchio\Model\xMateSR3C_description\urdf\xMateSR3C.urdf'))
    parser.add_argument('--frame', default='xMateSR3C_link6')
    args = parser.parse_args()
    chain, reference = load_chain(args.urdf, args.frame)
    # Independent zero-pose check for the reference xMate URDF.
    if args.frame == 'xMateSR3C_link6':
        np.testing.assert_allclose(forward(chain, np.zeros(6)), [0, .136, .9965], atol=1e-8)
    with (args.run_dir/'joint_tracking.csv').open() as handle:
        records = list(csv.DictReader(handle))
    epoch = max(int(row['epoch']) for row in records)
    grouped = {}
    for row in records:
        if int(row['epoch']) == epoch:
            grouped.setdefault(int(row['sample']), []).append(row)
    actual, target = [], []
    samples = sorted(grouped)
    for sample in samples:
        rows = sorted(grouped[sample], key=lambda r: int(r['joint']))
        if [int(r['joint']) for r in rows] != list(range(1,7)):
            raise ValueError('Missing or duplicated joint records')
        actual.append(forward(chain, [float(r['q_rad']) for r in rows]))
        target.append(forward(chain, [float(r['q_des_rad']) for r in rows]))
    actual, target = np.array(actual), np.array(target)
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite positions')
    error = np.linalg.norm(actual-target, axis=1)*1000
    out = args.run_dir/'end_effector'
    out.mkdir(exist_ok=True)
    stats = dict(epoch=epoch, samples=len(samples), frame=args.frame, reference=reference,
                 position_rmse_mm=float(np.sqrt(np.mean(error**2))),
                 mean_position_error_mm=float(error.mean()), max_position_error_mm=float(error.max()),
                 final_position_error_mm=float(error[-1]),
                 actual_endpoint_gap_mm=float(np.linalg.norm(actual[-1]-actual[0])*1000),
                 target_endpoint_gap_mm=float(np.linalg.norm(target[-1]-target[0])*1000))
    (out/'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    with (out/'positions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample','actual_x_m','actual_y_m','actual_z_m','target_x_m','target_y_m','target_z_m','error_mm'])
        writer.writerows([s,*a,*d,e] for s,a,d,e in zip(samples,actual,target,error))
    fig = plt.figure(figsize=(12,9), layout='constrained')
    ax = fig.add_subplot(2,2,1, projection='3d')
    for points, label, color, style in [(target,'Desired','#333333','--'),(actual,'Actual','#0072B2','-')]:
        ax.plot(*points.T, label=label, color=color, linestyle=style, linewidth=2)
        ax.scatter(*points[0], color=color, marker='o', s=35)
        ax.scatter(*points[-1], color=color, marker='x', s=45)
    combined = np.vstack([actual,target]); center = (combined.max(0)+combined.min(0))/2
    radius = max(np.ptp(combined,axis=0))/2*1.1
    ax.set(xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius),
           zlim=(center[2]-radius,center[2]+radius), xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',
           title='3D path (circle: start; cross: last sample)')
    ax.set_box_aspect([1,1,1]); ax.legend()
    for slot, dims, name in [(2,(0,1),'XY'),(3,(0,2),'XZ')]:
        axis = fig.add_subplot(2,2,slot)
        for points, label, color, style in [(target,'Desired','#333333','--'),(actual,'Actual','#0072B2','-')]:
            axis.plot(points[:,dims[0]],points[:,dims[1]],color=color,linestyle=style,label=label)
        axis.set(xlabel='x (m)',ylabel=('y (m)' if dims[1]==1 else 'z (m)'),title=name+' projection')
        axis.set_aspect('equal', adjustable='datalim'); axis.grid(alpha=.2); axis.legend()
    axis = fig.add_subplot(2,2,4)
    axis.plot(samples,error,color='#D55E00')
    axis.set(xlabel='Control sample',ylabel='Position error (mm)',title=f"RMSE {stats['position_rmse_mm']:.1f} mm; max {error.max():.1f} mm")
    axis.grid(alpha=.2)
    config_path = args.run_dir/'config.json'
    cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    mode = ('PD assistance enabled' if cfg['pd_assist'] else 'Neural-only control (PD OFF)') if 'pd_assist' in cfg else 'Control mode unspecified'
    fig.suptitle(f'ROKAE end-effector: epoch {epoch} | {args.frame} | {mode}')
    for ext in ('png','svg'):
        fig.savefig(out/f'trajectory_comparison.{ext}', dpi=200)
    plt.close(fig)
    print(json.dumps(stats, indent=2))
    print(out.resolve())


if __name__ == '__main__':
    main()
