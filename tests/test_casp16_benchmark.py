import torch

from src.advanced_model import AdvancedProteinFoldingModel
from src.benchmarks.casp16_benchmark import CASP16Benchmark
from src.data.casp16_loader import CASP16DataLoader, CASP16Target


class DummyEmbedder:
    embed_dim = 1280

    def __call__(self, sequences):
        l = len(sequences[0])
        return {"embeddings": torch.randn(1, l, self.embed_dim)}


def test_casp16_loader(tmp_path):
    loader = CASP16DataLoader(cache_dir=str(tmp_path / "cache"))
    targets = loader.download_targets(force_refresh=True)
    assert len(targets) > 0
    assert all(hasattr(t, "sequence") for t in targets)
    assert all(len(t.sequence) > 0 for t in targets)


def test_casp16_prediction(tmp_path):
    target = CASP16Target(
        target_id="T1204",
        sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
        native_pdb_path=None,
        category="Regular",
        length=55,
        release_date=None,
        has_domains=False,
        domains=[],
    )

    model = AdvancedProteinFoldingModel(input_dim=1280, use_quantum=False)
    benchmark = CASP16Benchmark(model, model, DummyEmbedder(), device="cpu")
    result = benchmark.predict_target(target, model)

    assert "coordinates" in result
    assert result["coordinates"].shape == (len(target.sequence), 3)
    assert "plddt" in result
    assert torch.all((result["plddt"] >= 0) & (result["plddt"] <= 100))
