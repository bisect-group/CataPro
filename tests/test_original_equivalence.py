"""Equivalence of the emulator_bench wrapper with the original CataPro code (CPU only).

Run from baselines/CataPro:
    CUDA_VISIBLE_DEVICES= python -m pytest tests/test_original_equivalence.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    """Import a file under a unique name; the original and wrapper modules share file names."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


og_inf_model = load("og_inf_model", "inference/model.py")
og_kcat_model = load("og_kcat_model", "training/kcat/model.py")
og_kcat_utils = load("og_kcat_utils", "training/kcat/utils.py")
og_km_model = load("og_km_model", "training/Km/model.py")
og_km_utils = load("og_km_utils", "training/Km/utils.py")
wr_model = load("wr_model", "emulator_bench/model.py")
wr_utils = load("wr_utils", "emulator_bench/utils.py")

ORIGINAL_MODELS = {
    "inference_kcat": og_inf_model.KcatModel,
    "inference_km": og_inf_model.KmModel,
    "training_kcat": og_kcat_model.KcatModel,
    "training_km": og_km_model.KcatModel,
}
FEATS = 1024 + 768 + 167


def make_data(n, seed=0):
    g = th.Generator().manual_seed(seed)
    feats = th.randn(n, 1024 + 768, generator=g)
    maccs = (th.rand(n, 167, generator=g) > 0.5).float()
    label = th.randn(n, 1, generator=g)
    return th.cat([feats, maccs, label], dim=1)


def copy_weights(src, dst):
    """Copy through an explicit key map; every parameter and buffer must be mapped on both sides."""
    keys = [
        "prot_norm.weight", "prot_norm.bias", "prot_norm.running_mean", "prot_norm.running_var",
        "prot_norm.num_batches_tracked",
        "molt5_norm.weight", "molt5_norm.bias", "molt5_norm.running_mean", "molt5_norm.running_var",
        "molt5_norm.num_batches_tracked",
        "decoder.0.weight", "decoder.0.bias",
        "decoder.1.weight", "decoder.1.bias", "decoder.1.running_mean", "decoder.1.running_var",
        "decoder.1.num_batches_tracked",
        "out.0.weight", "out.0.bias",
    ]
    src_sd = src.state_dict()
    assert set(src_sd) == set(keys), set(src_sd) ^ set(keys)
    assert set(dst.state_dict()) == set(keys), set(dst.state_dict()) ^ set(keys)
    dst.load_state_dict({k: src_sd[k].clone() for k in keys})


@pytest.mark.parametrize("name", ORIGINAL_MODELS)
@pytest.mark.parametrize("drop_rate", [0.0, 0.3])
def test_model_output_loss_gradients(name, drop_rate):
    """Same weights: forward output, original RMSE loss and every gradient match in fp64."""
    og = ORIGINAL_MODELS[name](drop_rate, device="cpu").double()
    wr = wr_model.SingleTaskRegressor(drop_rate, device="cpu").double()
    th.manual_seed(1)
    for p in og.parameters():
        p.data.normal_(0, 0.1)
    copy_weights(og, wr)

    data = make_data(16).double()
    ezy, sbt, label = data[:, :1024], data[:, 1024:-1], data[:, -1]
    for model, loss_fn in ((og, og_kcat_utils.rmse_loss), (wr, wr_utils.rmse_loss)):
        model.train()
        th.manual_seed(7)  # same dropout mask
        model.out_ = model(ezy, sbt)
        model.loss_ = loss_fn(label, model.out_.ravel())
        model.loss_.backward()

    th.testing.assert_close(wr.out_, og.out_, rtol=1e-12, atol=1e-12)
    th.testing.assert_close(wr.loss_, og.loss_, rtol=1e-12, atol=1e-12)
    og_params, wr_params = dict(og.named_parameters()), dict(wr.named_parameters())
    assert set(og_params) == set(wr_params)
    for k, p in og_params.items():
        th.testing.assert_close(wr_params[k].grad, p.grad, rtol=1e-10, atol=1e-12, msg=k)
    for k, b in og.state_dict().items():  # running statistics after one training step
        th.testing.assert_close(wr.state_dict()[k], b, rtol=1e-12, atol=1e-12, msg=k)

    og.eval(), wr.eval()
    th.testing.assert_close(wr(ezy, sbt), og(ezy, sbt), rtol=1e-12, atol=1e-12)


class OriginalDataset(th.utils.data.Dataset):
    """The original training script's Mydatasets."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def wrapper_batches(values, batch_size):
    """The wrapper's ShuffledBatches, copied from train_single_target_tvt.py without its heavy imports."""
    src = (ROOT / "emulator_bench/train_single_target_tvt.py").read_text()
    start = src.index("class ShuffledBatches")
    end = src.index("def build_loader")
    ns = {"DataLoader": DataLoader, "th": th}
    exec(src[start:end], ns)
    return ns["ShuffledBatches"](values, batch_size)


@pytest.mark.parametrize("n", [128, 130, 129, 10])
@pytest.mark.parametrize("workers", [0, 2])
def test_batch_order_matches_dataloader(n, workers):
    """Several epochs of shuffled batches equal DataLoader(shuffle=True), including the last partial and singleton batch."""
    values = make_data(n)
    th.manual_seed(42)
    og = DataLoader(OriginalDataset(values), batch_size=64, shuffle=True, num_workers=workers)
    og_epochs = [list(og) for _ in range(4)]
    th.manual_seed(42)
    wr = wrapper_batches(values, 64)
    wr_epochs = [list(wr) for _ in range(4)]
    assert len(og) == len(wr)
    for og_epoch, wr_epoch in zip(og_epochs, wr_epochs):
        assert [len(b) for b in og_epoch] == [len(b) for b in wr_epoch]
        assert all(th.equal(a, b) for a, b in zip(og_epoch, wr_epoch))
    if n == 129:
        assert len(wr_epochs[0][-1]) == 1


def end_to_end(og_model_cls, og_utils, n, epochs=2):
    """Train the original and the wrapper loops from identical weights and data; return both models."""
    values = make_data(n, seed=3)
    og = og_model_cls(0.0, device="cpu")
    wr = wr_model.SingleTaskRegressor(0.0, device="cpu")
    copy_weights(og, wr)
    og_opt = th.optim.Adam(og.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    wr_opt = th.optim.Adam(wr.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    th.manual_seed(42)
    og_loader = DataLoader(OriginalDataset(values), batch_size=64, shuffle=True, num_workers=2)
    og_out = [og_utils.run_a_training_epoch(og, og_loader, og_opt, "cpu") for _ in range(epochs)]
    th.manual_seed(42)
    wr_loader = wrapper_batches(values, 64)
    wr_out = [wr_utils.run_a_training_epoch(wr, wr_loader, wr_opt, "cpu") for _ in range(epochs)]
    return og, wr, og_out, wr_out


def test_training_epochs_match_original_kcat():
    """Whole training epochs (loader, loss, Adam) give the same weights and epoch loss as the original kcat loop."""
    og, wr, og_out, wr_out = end_to_end(og_kcat_model.KcatModel, og_kcat_utils, 130)
    for k, v in og.state_dict().items():
        th.testing.assert_close(wr.state_dict()[k], v, rtol=1e-4, atol=1e-5, msg=k)
    for a, b in zip(og_out, wr_out):
        np.testing.assert_allclose(b[-1], a[-1], rtol=1e-4)  # loss is the last original entry
        np.testing.assert_allclose(b[:2], a[:2], rtol=1e-4, atol=1e-5)  # PCC, SCC


def test_singleton_batch_original_handling():
    """The original kcat loop crashes on a size-1 batch; the original Km loop and the wrapper skip it and agree."""
    values = make_data(65)
    kcat = og_kcat_model.KcatModel(0.0, device="cpu")
    opt = th.optim.Adam(kcat.parameters())
    loader = DataLoader(OriginalDataset(values), batch_size=64, shuffle=False)
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        og_kcat_utils.run_a_training_epoch(kcat, loader, opt, "cpu")

    og, wr, og_out, wr_out = end_to_end(og_km_model.KcatModel, og_km_utils, 129)
    for k, v in og.state_dict().items():
        th.testing.assert_close(wr.state_dict()[k], v, rtol=1e-4, atol=1e-5, msg=k)
    for a, b in zip(og_out, wr_out):
        np.testing.assert_allclose(b[-1], a[-1], rtol=1e-4)  # mean loss divides by len(loader) in both


@pytest.fixture(scope="module")
def t5_models(monkeypatch_module):
    """Load the cached ProtT5 and MolT5 once and share them between the original and wrapper code."""
    pytest.importorskip("transformers")
    from transformers import T5EncoderModel, T5Tokenizer

    fu = load("wr_feature_utils", "emulator_bench/feature_utils.py")
    og = load("og_inference_utils", "inference/utils.py")
    cache = {}

    def memo(cls):
        orig = cls.from_pretrained

        def get(name, *a, **kw):
            if (cls, name) not in cache:
                try:
                    cache[cls, name] = orig(name, *a, local_files_only=True, **kw)
                except OSError:
                    pytest.skip(f"{name} is not in the local Hugging Face cache")
            return cache[cls, name]

        return get

    monkeypatch_module.setattr(T5EncoderModel, "from_pretrained", memo(T5EncoderModel))
    monkeypatch_module.setattr(T5Tokenizer, "from_pretrained", memo(T5Tokenizer))
    # The original code calls batch_encode_plus, which transformers 5 removed; __call__ is its replacement.
    monkeypatch_module.setattr(
        T5Tokenizer, "batch_encode_plus", lambda self, texts, **kw: self(texts, **kw), raising=False
    )
    return fu, og


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_molt5_batched_masked_equals_per_item(t5_models):
    """Batched padded MolT5 features equal the original per-item features (fp32, CPU)."""
    fu, og = t5_models
    smiles = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "C", "CCO", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"]
    expected = og.get_molT5_embed(list(smiles), fu.MOL_MODEL_ID)
    got = fu.get_molT5_embed(smiles, batch_size=3)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_prott5_batched_masked_equals_per_item(t5_models):
    """Batched padded ProtT5 features equal the original per-item features (fp32, CPU)."""
    fu, og = t5_models
    seqs = ["MKTAYIAKQR", "MKVLAAGIVGLLLAQPSEA", "MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAKQR", "UZOBMKT"]
    expected = og.Seq_to_vec(list(seqs), fu.PROT_MODEL_ID)
    got = fu.Seq_to_vec(seqs, batch_size=3)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_maccs_equals_original():
    """Cached-unique MACCS keys equal the original per-item keys exactly."""
    fu = load("wr_feature_utils2", "emulator_bench/feature_utils.py")
    og = load("og_inference_utils2", "inference/utils.py")
    smiles = ["CCO", "CC(=O)O", "CCO", "c1ccccc1"]
    np.testing.assert_array_equal(fu.GetMACCSKeys(smiles), og.GetMACCSKeys(list(smiles)))
