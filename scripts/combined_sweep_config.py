"""Identifiers and coefficients for the matched action-prediction runs."""
from dataclasses import dataclass
from experiment_paths import ROOT

EXPERIMENTS = ROOT / "experiments"
REFERENCE = EXPERIMENTS / "episode50-30-20261106-03"
COMBINED = EXPERIMENTS / "episode50-combined-20261107-01"
MSE_ONLY = EXPERIMENTS / "episode50-mse-20261107-01"
ACTION_WEIGHT = 0.09


@dataclass(frozen=True)
class SweepVariant:
    tag: str
    sigreg_weight: float
    action_weight: float = ACTION_WEIGHT
    track_cross_example_latent_spread: bool = False
    objective: str = "combined"
    run_id: str = "20261107-01"

    @property
    def family(self):
        return {"combined": "combined", "action_prediction": "action"}[self.objective]

    @property
    def label(self):
        return f"{self.family}_{self.tag}"

    @property
    def config(self):
        return f"lewm_{self.label}"

    @property
    def experiment(self):
        return EXPERIMENTS / f"episode50-{self.family}-{self.tag}-{self.run_id}"

    @property
    def model(self):
        return f"tworoom-{self.family}-{self.tag}-episode50-{self.run_id}"


VARIANTS = {variant.label: variant for variant in (
    SweepVariant("s06", 0.06),
    SweepVariant("s03", 0.03),
    SweepVariant("a100", 0.09, action_weight=1.0, track_cross_example_latent_spread=True),
    SweepVariant("w50", 0.0, action_weight=50.0, track_cross_example_latent_spread=True,
                 objective="action_prediction", run_id="20261110-01"),
)}

