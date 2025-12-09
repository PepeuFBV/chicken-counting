import torch
from src.model.chicken_model import ChickenCountingModel


def test_forward_shape():
    model = ChickenCountingModel(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256)


def test_predict_count():
    model = ChickenCountingModel(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        counts = model.predict_count(x)
    assert counts.shape == (2,)
