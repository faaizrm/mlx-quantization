"""Serializable quantization rules; later matches override earlier rules"""

import hashlib
import json
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Rule:
    pattern: str
    bits: int
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.bits not in (4, 8) or self.group_size not in (32, 64, 128):
            raise ValueError("This study supports 4/8 bits and groups of 32/64/128")


@dataclass(frozen=True)
class QuantConfig:
    name: str
    rules: tuple[Rule, ...] = ()

    def options(self, path: str) -> dict | bool:
        # limit every rule to transformer projections; tied embeddings stay fp16
        if not path.startswith("model.layers.") or path.rsplit(".", 1)[-1] not in MODULES:
            return False
        for rule in reversed(self.rules):
            if fnmatchcase(path, rule.pattern):
                return {"bits": rule.bits, "group_size": rule.group_size, "mode": "affine"}
        return False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, value: dict) -> "QuantConfig":
        return cls(value["name"], tuple(Rule(**r) for r in value["rules"]))


def uniform(bits: int, group_size: int = 64) -> QuantConfig:
    return QuantConfig(f"int{bits}-g{group_size}", (Rule("model.layers.*", bits, group_size),))


def baselines() -> list[QuantConfig]:
    return [QuantConfig("fp16")] + [
        uniform(b, g) for b, g in ((8, 32), (8, 64), (4, 32), (4, 64), (4, 128))
    ]


def layer_only(index: int) -> QuantConfig:
    return QuantConfig(f"layer-{index:02d}-int4", (Rule(f"model.layers.{index}.*", 4),))


def module_only(module: str) -> QuantConfig:
    if module not in MODULES:
        raise ValueError(module)
    return QuantConfig(f"module-{module}-int4", (Rule(f"model.layers.*.{module}", 4),))


def mixed(blocks: list[int], name: str) -> QuantConfig:
    return QuantConfig(
        name,
        (Rule("model.layers.*", 4),)
        + tuple(Rule(f"model.layers.{i}.*", 8) for i in sorted(blocks)),
    )
