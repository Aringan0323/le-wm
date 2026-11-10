"""Two-Room ablations of the observed-displacement Delta-JEPA objective."""
from dataclasses import dataclass
from combined_sweep_config import EXPERIMENTS


@dataclass(frozen=True)
class DeltaVariant:
    label: str
    ldad_weight: float
    sigreg_weight: float
    objective = "delta_jepa"
    track_cross_example_latent_spread = True

    @property
    def config(self):
        return f"lewm_{self.label}"

    @property
    def experiment(self):
        return EXPERIMENTS / f"episode50-{self.label.replace('_', '-')}-20261110-01"

    @property
    def model(self):
        return f"tworoom-{self.label.replace('_', '-')}-episode50-20261110-01"

    @property
    def display_label(self):
        return f"LDAD {self.ldad_weight:g} + SIGReg {self.sigreg_weight:g}"


VARIANTS = {variant.label: variant for variant in (
    DeltaVariant("delta_l0_s09", 0.0, 0.09),
    DeltaVariant("delta_l50_s09", 50.0, 0.09),
    DeltaVariant("delta_l50_s0", 50.0, 0.0),
)}
