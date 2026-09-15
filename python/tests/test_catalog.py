import pytest

import neuronbridge as nb


pytestmark = pytest.mark.skipif(
    nb._core is None,
    reason="neuronbridge native extension is not built",
)


def test_catalog_lists_public_models_and_filters_backends():
    models = nb.catalog.list_models()
    assert "TimeDrivenLIF_Exponential_double" in models
    assert "CustomLifConductanceV1" not in models

    maintained_models = nb.catalog.list_models(public_only=False)
    assert "CustomLifConductanceV1" in maintained_models

    dense_models = nb.catalog.list_models(backend="dense_gpu")
    assert "TimeDrivenLIF_Exponential_double" in dense_models
    assert "InputSpikeNeuronModel" not in dense_models


def test_catalog_describe_uses_native_get_parameters():
    info = nb.catalog.describe_model(
        "TimeDrivenLIF_Exponential_double",
        base_timestep=0.1,
    )
    assert info["name"] == "TimeDrivenLIF_Exponential_double"
    assert info["backend"] == "legacy_cpu"
    assert info["implementation"] == "TimeDrivenLIF_Exponential_double"
    assert info["parameters"]["tau"] == pytest.approx(20.0)
    assert info["parameters"]["int_method"]["name"] == "ForwardEulerMethod"
    assert info["parameters"]["int_method"]["parameters"]["step"] == pytest.approx(0.1)


def test_catalog_alias_selects_legacy_gpu_by_default():
    info = nb.catalog.describe_model(
        "TimeDrivenLIF_Exponential_double_GPU",
        base_timestep=0.1,
    )
    assert info["name"] == "TimeDrivenLIF_Exponential_double"
    assert info["backend"] == "legacy_gpu"
    assert info["implementation"] == "TimeDrivenLIF_Exponential_double_GPU"
    assert "V_rest" in info["parameters"]


def test_catalog_can_query_a_model_with_required_configuration():
    info = nb.catalog.describe_model(
        "HandwritingTimeDrivenModel",
        parameters={"role": "CM"},
        base_timestep=0.1,
    )
    assert info["parameters"]["role"] == "CM"
    assert info["parameters"]["int_method"]["parameters"]["step"] == pytest.approx(0.1)


def test_catalog_rejects_unknown_model_and_backend():
    with pytest.raises(KeyError):
        nb.catalog.describe_model("MissingNeuronModel")
    with pytest.raises(ValueError):
        nb.catalog.list_models(backend="missing")


def test_catalog_converts_existing_fixed_size_parameter_arrays():
    voltage_jump = nb.catalog.describe_model("TimeDrivenLIF_Voltage_jump")
    izhikevich = nb.catalog.describe_model("TimeDrivenIzhikevic_Exponential_Decay")
    assert voltage_jump["parameters"]["random_mu"] == [0.0, 0.0]
    assert izhikevich["parameters"]["random_mu"] == [0.0] * 6
