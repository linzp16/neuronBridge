"""Continuous, fixed-orientation XY-circle IK followed by cerebellar training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_rokae_end_effector import load_chain, pose
from rokae_cerebellum_6dof import Config, default_urdf, train


def rotation_vector(matrix):
    theta = np.arccos(np.clip((np.trace(matrix)-1)/2, -1., 1.))
    vee = np.array([matrix[2,1]-matrix[1,2], matrix[0,2]-matrix[2,0],
                    matrix[1,0]-matrix[0,1]])
    if theta > np.pi-1e-5:
        raise RuntimeError('IK orientation residual approached pi')
    return vee * (.5 if theta < 1e-7 else theta/(2*np.sin(theta)))


def solve_ik(residual, seed, lower, upper):
    q = seed.copy()
    for _ in range(150):
        error = residual(q)
        if np.linalg.norm(error) < 1e-9:
            break
        h = 1e-6
        jac = np.column_stack([(residual(q + np.eye(6)[i]*h)-error)/h
                               for i in range(6)])
        step = -jac.T @ np.linalg.solve(jac @ jac.T + 1e-8*np.eye(6), error)
        step *= min(1., .2/max(np.linalg.norm(step), 1e-15))
        for scale in (1., .5, .25, .125, .0625, .03125):
            candidate = np.clip(q+scale*step, lower+1e-8, upper-1e-8)
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                q = candidate
                break
        else:
            break
    return q


def make_circle(urdf, radius, period, dt):
    chain, reference = load_chain(urdf, 'xMateSR3C_link6')
    seed = np.deg2rad([0, 20, -70, 0, 45, 0])
    initial = pose(chain, seed)
    center = initial[:3,3].copy()
    center[0] -= radius  # The known pose lies at phase zero.
    orientation = initial[:3,:3]
    limits = [j.find('limit') for j in chain if j.get('type') != 'fixed']
    lower = np.array([float(l.get('lower')) for l in limits])
    upper = np.array([float(l.get('upper')) for l in limits])
    speed_limits = np.array([float(l.get('velocity')) for l in limits])
    count = round(period/dt)
    if count < 3 or not np.isclose(count*dt, period):
        raise ValueError('Period must be an integer multiple of control interval')
    times = np.arange(count+1)*dt
    targets = center + np.column_stack([radius*np.cos(2*np.pi*times/period),
                                       radius*np.sin(2*np.pi*times/period), np.zeros(count+1)])
    angles, position_errors, orientation_errors = [], [], []
    for target in targets:
        def residual(q):
            transform = pose(chain,q)
            rot_error = rotation_vector(orientation.T @ transform[:3,:3])
            return np.r_[transform[:3,3]-target, .2*rot_error]
        solution = solve_ik(residual, seed, lower, upper)
        error = residual(solution)
        if np.linalg.norm(error[:3]) > 1e-6 or np.linalg.norm(error[3:])/.2 > 1e-5:
            raise RuntimeError(f'Circle IK infeasible at sample {len(angles)}: {error}')
        seed = solution
        angles.append(seed.copy())
        position_errors.append(np.linalg.norm(error[:3]))
        orientation_errors.append(np.linalg.norm(error[3:])/.2)
    angles = np.array(angles)
    if np.max(np.abs(np.diff(angles,axis=0))) > .2:
        raise RuntimeError('Discontinuous IK branch')
    if not np.allclose(angles[0],angles[-1],atol=1e-5):
        raise RuntimeError('IK joint trajectory is not closed')
    velocity = np.gradient(angles, dt, axis=0, edge_order=2)
    velocity[0] = velocity[-1] = (angles[1]-angles[-2])/(2*dt)
    if (np.abs(velocity)>speed_limits).any():
        raise RuntimeError('Target exceeds URDF velocity limits')
    metadata = dict(kind='xy_circle',radius_m=radius,period_s=period,center_m=center.tolist(),
                    frame='xMateSR3C_link6',reference=reference,
                    fixed_rotation_matrix=orientation.tolist(),
                    max_ik_position_error_m=max(position_errors),
                    max_ik_orientation_error_rad=max(orientation_errors),
                    samples=len(times), includes_duplicate_endpoint=True,
                    max_abs_velocity_rad_s=np.max(np.abs(velocity),axis=0).tolist())
    return list(zip(angles.tolist(),velocity.tolist())), targets, times, metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,default=default_urdf())
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--radius',type=float,default=.05)
    p.add_argument('--period',type=float,default=4.)
    p.add_argument('--disable-learning',action='store_true')
    p.add_argument('--pd-assist',action='store_true',help='Explicitly opt in to direct PD torque (off by default)')
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/rokae_neural_only/standalone'))
    args=p.parse_args()
    if args.radius<=0 or args.period<=0:
        p.error('Radius and period must be positive')
    if args.output_dir.exists():
        p.error('Choose a new output directory to preserve previous experiments')
    trajectory,targets,times,meta=make_circle(args.urdf,args.radius,args.period,.02)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'trajectory.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    with (args.output_dir/'desired_circle.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['target_time_s','x_m','y_m','z_m',*[f'q{i}_rad' for i in range(1,7)]])
        w.writerows([t,*xyz,*state[0]] for t,xyz,state in zip(times,targets,trajectory))
    print(json.dumps(meta,indent=2),flush=True)
    cfg=Config(epochs=args.epochs,samples_per_epoch=len(trajectory),
               pd_assist=args.pd_assist,
               alpha=0 if args.disable_learning else Config.alpha,
               beta=0 if args.disable_learning else Config.beta)
    train(cfg,args.urdf,args.output_dir,trajectory=trajectory)


if __name__=='__main__':
    main()
