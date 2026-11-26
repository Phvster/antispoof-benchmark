"""Registry tên model -> "module:Class". Config chỉ cần `model.name` (hoặc ghi rõ `model.class`).

Mọi model phải có logits(x) -> [B,2] (cột 0 = giả, cột 1 = thật) và
forward(x) = logits[:,1] - logits[:,0] (điểm cao = thật); tests/test_models_registry.py kiểm tra.
"""

REGISTRY = {
    "aasist_l": "models.aasist_l.model:AASISTL",
}
