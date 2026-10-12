import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def circle(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples'/'python'))
    return importlib.import_module('train_rokae_xy_circle')


def test_circle_geometry_and_continuity(circle):
    urdf = circle.default_urdf()
    if not urdf.exists():
        pytest.skip('xMate URDF unavailable')
    trajectory, targets, times, metadata = circle.make_circle(urdf,.05,4.,.02)
    assert len(trajectory) == 201
    assert times[-1] == 4.
    np.testing.assert_allclose(targets[0],targets[-1],atol=1e-10)
    np.testing.assert_allclose(targets[:,2],targets[0,2],atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(targets[:,:2]-np.array(metadata['center_m'])[:2],axis=1),.05,atol=1e-10)
    assert metadata['max_ik_position_error_m'] < 1e-6
    assert metadata['max_ik_orientation_error_rad'] < 1e-5
    np.testing.assert_allclose(trajectory[0][0],trajectory[-1][0],atol=1e-5)


@pytest.mark.parametrize('kwargs', [dict(cf_angle_norm_deg=0),dict(alpha=-1),
    dict(beta=1),dict(pd_kp=(1,)),dict(feedback_sample_count=-1),
    dict(angle_min_deg=(0,)*6),dict(velocity_range_deg_s=(0,)*6)])
def test_invalid_training_parameters(circle,kwargs):
    with pytest.raises(ValueError):
        circle.Config(**kwargs)


def test_training_defaults_to_neural_only(circle):
    assert circle.Config().pd_assist is False
