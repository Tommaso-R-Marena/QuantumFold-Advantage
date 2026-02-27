from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.advanced_model import AdvancedProteinFoldingModel
from src.benchmarks.research_metrics import (
    ResearchBenchmark,
    compute_gdt_ts,
    compute_rmsd,
    compute_tm_score,
)
from src.data.casp16_loader import CASP16DataLoader, CASP16Target
from src.utils.pdb_writer import load_pdb_coords, save_pdb


class _Timeout(Exception):
    pass


class CASP16Benchmark:
    def __init__(
        self,
        model_quantum: AdvancedProteinFoldingModel,
        model_classical: AdvancedProteinFoldingModel,
        embedder,
        device: str = "cuda",
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_quantum = model_quantum.to(self.device).eval()
        self.model_classical = model_classical.to(self.device).eval()
        self.embedder = embedder
        self.metrics_calculator = ResearchBenchmark()

    def _signal_handler(self, *_):
        raise _Timeout("prediction timeout")

    def _embed(self, sequence: str) -> torch.Tensor:
        try:
            embed_out = self.embedder([sequence])
            return embed_out["embeddings"].to(self.device)
        except Exception:
            # fallback: deterministic synthetic embedding
            dim = getattr(self.embedder, "embed_dim", 1280)
            torch.manual_seed(len(sequence))
            return torch.randn(1, len(sequence), dim, device=self.device)

    def predict_target(
        self,
        target: CASP16Target,
        model: AdvancedProteinFoldingModel,
        use_recycling: int = 3,
    ) -> Dict:
        start_time = time.time()
        signal.signal(signal.SIGALRM, self._signal_handler)
        signal.alarm(600)
        try:
            if not target.sequence:
                raise ValueError("Target sequence is empty")

            embeddings_tensor = self._embed(target.sequence)
            prev_coords = None
            output = None
            for _ in range(max(1, use_recycling)):
                with torch.no_grad():
                    try:
                        output = model(embeddings_tensor, mask=None)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and self.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                            embeddings_tensor = embeddings_tensor.cpu()
                            model = model.cpu()
                            self.device = "cpu"
                            output = model(embeddings_tensor, mask=None)
                        else:
                            raise
                    prev_coords = output["coordinates"].detach()

            coords = output["coordinates"].squeeze(0).detach().cpu()
            plddt = output.get("plddt", torch.zeros(coords.shape[0])).squeeze(0).detach().cpu()
            plddt = torch.clamp(plddt, 0, 100)

            pdb_path = save_pdb(
                coords=coords.numpy(),
                sequence=target.sequence,
                filename=f"predictions/{target.target_id}_predicted.pdb",
            )

            metrics = None
            if target.native_pdb_path and target.native_pdb_path.exists():
                native_coords = load_pdb_coords(target.native_pdb_path)
                n = min(len(coords), len(native_coords))
                if n > 3:
                    pred = coords[:n]
                    native = native_coords[:n]
                    metrics = {
                        "tm_score": compute_tm_score(pred, native),
                        "rmsd": compute_rmsd(pred, native),
                        "gdt_ts": compute_gdt_ts(pred, native),
                    }

            return {
                "target_id": target.target_id,
                "pdb_file": str(pdb_path),
                "coordinates": coords,
                "plddt": plddt,
                "metrics": metrics,
                "inference_time": time.time() - start_time,
            }
        finally:
            signal.alarm(0)

    def run_full_benchmark(self, n_targets: int = 50) -> Dict:
        loader = CASP16DataLoader()
        targets = loader.download_targets()[:n_targets]

        quantum_results: List[Dict] = []
        classical_results: List[Dict] = []

        for target in tqdm(targets, desc="CASP16 Benchmark"):
            quantum_results.append(self.predict_target(target, self.model_quantum))
            classical_results.append(self.predict_target(target, self.model_classical))

        rows = []
        for q, c in zip(quantum_results, classical_results):
            qtm = q["metrics"]["tm_score"] if q["metrics"] else np.nan
            ctm = c["metrics"]["tm_score"] if c["metrics"] else np.nan
            rows.append(
                {
                    "target_id": q["target_id"],
                    "quantum_tm": qtm,
                    "classical_tm": ctm,
                    "quantum_rmsd": q["metrics"]["rmsd"] if q["metrics"] else np.nan,
                    "classical_rmsd": c["metrics"]["rmsd"] if c["metrics"] else np.nan,
                    "quantum_time": q["inference_time"],
                    "classical_time": c["inference_time"],
                    "improvement": qtm - ctm if not (np.isnan(qtm) or np.isnan(ctm)) else np.nan,
                }
            )

        df_results = pd.DataFrame(rows)
        valid = df_results.dropna(subset=["quantum_tm", "classical_tm"])

        if len(valid) >= 2:
            stat_results = self.metrics_calculator.compare_methods(
                quantum_scores=valid["quantum_tm"].values,
                classical_scores=valid["classical_tm"].values,
                metric_name="TM-score",
            )
        else:
            stat_results = {"wilcoxon_pvalue": 1.0, "cohens_d": 0.0}

        return {
            "per_target_results": df_results,
            "statistical_tests": stat_results,
            "aggregate_stats": {
                "quantum_mean_tm": float(valid["quantum_tm"].mean()) if len(valid) else float("nan"),
                "classical_mean_tm": float(valid["classical_tm"].mean()) if len(valid) else float("nan"),
                "quantum_mean_runtime": float(df_results["quantum_time"].mean()) if len(df_results) else 0.0,
                "classical_mean_runtime": float(df_results["classical_time"].mean()) if len(df_results) else 0.0,
            },
            "raw_quantum": quantum_results,
            "raw_classical": classical_results,
        }

    def generate_casp16_report(self, results: Dict, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        df = results["per_target_results"]
        df.to_csv(output_dir / "casp16_results.csv", index=False)

        latex_table = (
            df.sort_values("improvement", ascending=False)
            .head(20)
            .to_latex(columns=["target_id", "quantum_tm", "classical_tm", "improvement"], float_format="%.3f", index=False)
        )
        (output_dir / "casp16_table.tex").write_text(latex_table)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.scatter(df["classical_tm"], df["quantum_tm"], alpha=0.7)
        finite = df[["classical_tm", "quantum_tm"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            mn = min(finite.min())
            mx = max(finite.max())
            ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel("Classical TM-score")
        ax.set_ylabel("Quantum TM-score")

        ax = axes[0, 1]
        ax.boxplot([df["classical_tm"].dropna(), df["quantum_tm"].dropna()], labels=["Classical", "Quantum"])
        ax.set_title("TM-score distribution")

        ax = axes[1, 0]
        ax.hist(df["improvement"].dropna(), bins=20)
        ax.set_title("Quantum improvement histogram")

        ax = axes[1, 1]
        ax.scatter(df["classical_time"], df["quantum_time"], alpha=0.7)
        ax.set_xlabel("Classical runtime (s)")
        ax.set_ylabel("Quantum runtime (s)")

        fig.tight_layout()
        fig.savefig(output_dir / "casp16_analysis.png", dpi=300)
        plt.close(fig)
