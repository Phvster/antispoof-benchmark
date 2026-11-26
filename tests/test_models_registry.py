"""Test chung cho MỌI model trong models.REGISTRY (cả của A và B):
logits [B,2] và forward = logits[:,1] - logits[:,0] (điểm cao = thật)."""
import pytest
import torch

from common.audio import NB_SAMP
from common.utils import build_model
from models import REGISTRY


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_model_interface(name):
    torch.manual_seed(0)
    m = build_model({"name": name}).eval()
    x = torch.randn(2, NB_SAMP) * 0.1
    with torch.no_grad():
        lg = m.logits(x)
        s = m(x)
    assert lg.shape == (2, 2)
    assert s.shape == (2,)
    torch.testing.assert_close(s, lg[:, 1] - lg[:, 0], atol=1e-5, rtol=0)
