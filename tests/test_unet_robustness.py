"""Shape, serialization and training-only normalization contracts for U-Net."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.exp_unet_robustness import encode_coefficients, fit_normalization, predict_physical, train_member
from models.tiny_unet import UNet2d


def data():
    rng = np.random.default_rng(91)
    return np.exp(rng.normal(size=(3, 8, 8))), rng.normal(size=(3, 8, 8))*.02+.1


@pytest.mark.parametrize("width", [4, 10, 16])
def test_unet_shape_and_gradient_flow_through_encoder_and_decoder(width):
    torch.set_num_threads(1)
    model = UNet2d(width=width)
    output = model(torch.randn(2, 8, 12, 3))
    assert output.shape == (2, 8, 12)
    output.square().mean().backward()
    for name in ["enc1", "enc2", "bott", "up2", "dec2", "up1", "dec1", "out"]:
        gradients = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.any(g != 0) for g in gradients)


@pytest.mark.parametrize("shape", [(2,8,8), (2,8,8,2), (2,7,8,3), (2,8,2,3)])
def test_unet_rejects_invalid_spatial_and_channel_contract(shape):
    with pytest.raises(ValueError):
        UNet2d(width=4)(torch.randn(*shape))


def test_training_normalization_is_reused_for_each_target_batch():
    a, u = data()
    normalization = fit_normalization(a,u)
    target = a[:1]*1000
    encoded = encode_coefficients(target,normalization)
    expected = (np.log(target)-np.log(a).mean())/np.log(a).std()
    np.testing.assert_allclose(encoded[...,0],expected,rtol=1e-7)
    np.testing.assert_array_equal(encoded,encode_coefficients(np.concatenate([target,a]),normalization)[:1])
    assert not np.isclose(encoded[...,0].mean(),0)


def test_prediction_physical_units_and_batch_invariance():
    a,u = data(); normalization = fit_normalization(a,u)
    class Constant(torch.nn.Module):
        def forward(self,x):
            return torch.full(x.shape[:3],2.0)
    expected = 2*normalization["ustd"]+normalization["umean"]
    prediction = predict_physical(Constant(),a,normalization,batch_size=1)
    np.testing.assert_allclose(prediction,expected,rtol=0,atol=1e-15)
    torch.manual_seed(4)
    model = UNet2d(width=4)
    first = predict_physical(model,a,normalization,batch_size=1)
    together = predict_physical(model,a,normalization,batch_size=3)
    np.testing.assert_allclose(first,together,rtol=1e-5,atol=1e-8)
    assert not model.training


def test_checkpoint_roundtrip_and_fixed_seed_training_are_reproducible(tmp_path):
    a,u = data(); normalization = fit_normalization(a,u)
    args = SimpleNamespace(width=4,lr=.001,epochs=2,batch_size=2)
    first,history = train_member(19,a,u,normalization,args)
    second,_ = train_member(19,a,u,normalization,args)
    assert len(history) == 2 and history[-1]["epoch"] == 2
    for left,right in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    path = tmp_path/"model.pt"
    torch.save({"model_state_dict":first.state_dict(),"normalization":normalization},path)
    saved = torch.load(path,weights_only=True)
    restored = UNet2d(width=4)
    restored.load_state_dict(saved["model_state_dict"])
    np.testing.assert_array_equal(predict_physical(first,a,normalization),
                                  predict_physical(restored,a,saved["normalization"]))


def test_normalization_rejects_corrupt_or_degenerate_training_data():
    a,u = data()
    for corrupt in [np.zeros_like(a), np.full_like(a,np.nan), -a]:
        with pytest.raises(ValueError):
            fit_normalization(corrupt,u)
    with pytest.raises(ValueError):
        fit_normalization(a,np.zeros_like(u))
    with pytest.raises(ValueError):
        fit_normalization(a,u[:1])
