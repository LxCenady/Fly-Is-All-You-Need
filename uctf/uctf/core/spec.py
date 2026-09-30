"""The experiment spec: every part of a connectome model, named and explicit.

Version 2 (JSON). Each section names a plugin and passes it parameters; no
part reads another part's settings.

{
 "uctf": 2,
 "name": "my-connectome",
 "connectome": {"source": "folder", "params": {"path": "data"}},
 "transforms": [{"transform": "cut_inputs",
                 "params": {"population": "SENSORY", "layers": ["chemical"]}}],
 "populations": {"PN": {"col": "cell_type", "contains": "PN"}, ...},
 "neuron": {"model": "lif", "params": {"dt": 0.02, "tau": 0.1, ...},
            "overrides": [{"population": "KC", "params": {"threshold": 1.2}}]},
 "synapses": [{"layer": "chemical", "model": "chemical",
               "params": {"gain": 1.5}}],
 "input": {"encoder": "random_subset", "population": "PN", "steps": 6,
           "params": {"active": 160, "drive": 1.5, "sustain": 0.5, "seed": 3}},
 "readout": [{"name": "kc", "feature": "counts", "population": "KC"}],
 "controls": {"rewire-full": [{"transform": "rewire", "params": {...}}]},
 "activity_match": {"param": "synapses.chemical.gain", "readouts": ["kc"]},
 "view": {...}
}

Paths in "connectome.params.path" are relative to the spec file. Version-1
specs (the first UCTF release) are migrated by `migrate_v1`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

VERSION = 2
SPECS = Path(__file__).resolve().parents[1] / "specs"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
REQUIRED = ("connectome", "neuron", "synapses", "input", "readout")


def builtin_specs() -> list:
    return sorted(p.stem for p in SPECS.glob("*.json"))


def find_spec_file(name: str) -> Path | None:
    """A spec of this name among UCTF's built-in specs and examples."""
    cands = [SPECS / f"{name}.json", *sorted(EXAMPLES.glob(f"*/{name}.json"))]
    return next((p for p in cands if name and p.exists()), None)


def load_spec(ref) -> dict:
    """A spec dict, a built-in name or a JSON path -> a validated v2 spec."""
    if isinstance(ref, dict):
        spec, base = copy.deepcopy(ref), None
    else:
        path = Path(ref)
        if not path.suffix:
            path = find_spec_file(str(ref)) or path
        if not path.exists():
            known = ", ".join(builtin_specs())
            raise FileNotFoundError(f"no spec {ref!r} (built-in: {known})")
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec.setdefault("name", path.stem)
        base = path.parent
    if spec.get("uctf", 1) == 1:
        spec = migrate_v1(spec)
    _resolve_paths(spec, base)
    return validate(spec)


TOP_KEYS = {"uctf", "name", "description", "connectome", "transforms",
            "populations", "neuron", "synapses", "input", "readout",
            "controls", "activity_match", "view"}
SECTION_KEYS = {"connectome": {"source", "params"},
                "neuron": {"model", "params", "overrides"},
                "input": {"encoder", "population", "steps", "params"},
                "activity_match": {"param", "readouts"}}
ITEM_KEYS = {"synapses": {"layer", "model", "params"},
             "readout": {"name", "feature", "population", "params"},
             "transforms": {"transform", "params"}}


def validate(spec: dict) -> dict:
    """Check the structure; an unknown key is an error, never ignored."""
    if spec.get("uctf") != VERSION:
        raise ValueError(f"spec version {spec.get('uctf')!r}; expected {VERSION}")
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"spec {spec.get('name')!r} lacks {', '.join(missing)}")
    _known(spec, TOP_KEYS, "spec")
    for key, allowed in SECTION_KEYS.items():
        if key in spec:
            _known(spec[key], allowed, key)
    for key, allowed in ITEM_KEYS.items():
        for item in spec.get(key, []):
            _known(item, allowed, key)
    for name, steps in spec.get("controls", {}).items():
        for step in steps:
            _known(step, ITEM_KEYS["transforms"], f"controls.{name}")
    if not isinstance(spec["synapses"], list) or not spec["synapses"]:
        raise ValueError("spec 'synapses' must be a non-empty list")
    names = [r["name"] for r in spec["readout"]]
    if len(set(names)) != len(names):
        raise ValueError(f"readout names repeat: {names}")
    spec.setdefault("populations", {})
    spec.setdefault("transforms", [])
    spec.setdefault("controls", {})
    match = spec.get("activity_match")
    if match:
        try:
            get_path(spec, match["param"])
        except KeyError:
            raise ValueError(f"activity_match.param {match['param']!r} must "
                             "be set explicitly in the spec") from None
        unknown = set(match["readouts"]) - set(names)
        if unknown:
            raise ValueError(f"activity_match names no readout {sorted(unknown)}")
    return spec


def _known(section: dict, allowed: set, where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} "
                         f"(allowed: {', '.join(sorted(allowed))})")


def _resolve_paths(spec: dict, base: Path | None) -> None:
    params = spec.get("connectome", {}).get("params", {})
    path = params.get("path")
    if base is not None and path and not Path(path).is_absolute():
        params["path"] = str((base / path).resolve())


def get_path(spec: dict, dotted: str):
    """Value at 'a.b.c'; list items are addressed by their 'layer' or 'name'."""
    obj = spec
    for key in dotted.split("."):
        obj = _child(obj, key)
    return obj


def set_path(spec: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    obj = spec
    for key in keys[:-1]:
        obj = _child(obj, key, create=True)
    obj[keys[-1]] = value


# Override paths of version-1 specs, still accepted (e.g. --spec-set).
V1_PATHS = {"input.active": "input.params.active",
            "input.drive": "input.params.drive",
            "input.sustain": "input.params.sustain",
            "input.code_seed": "input.params.seed",
            "neuron.gain": "synapses.chemical.params.gain",
            "neuron.gap_gain": "synapses.electrical.params.gain"}
V1_NEURON = ("dt", "tau", "threshold", "reset", "tonic", "noise_hz",
             "noise_amp", "seed")


def v2_path(dotted: str) -> str:
    """A version-1 override path in version-2 terms (others unchanged)."""
    if dotted in V1_PATHS:
        return V1_PATHS[dotted]
    head, _, rest = dotted.partition(".")
    if head == "neuron" and rest in V1_NEURON:
        return f"neuron.params.{rest}"
    return dotted


def override(spec: dict, dotted: str, value) -> str:
    """Set one existing entry, or a new plugin param (a key directly under
    "params"). Anything else is an error, so a typo is never ignored.
    Returns the (version-2) path that was set."""
    path = v2_path(dotted)
    keys = path.split(".")
    obj = spec
    for key in keys[:-1]:
        is_params = key == "params" and isinstance(obj, dict)
        if is_params and key not in obj:
            obj[key] = {}
        try:
            obj = _child(obj, key)
        except (KeyError, TypeError):
            raise KeyError(f"spec has no {path!r} (at {key!r})") from None
    last = keys[-1]
    if last not in obj and keys[-2:-1] != ["params"]:
        raise KeyError(f"spec has no {path!r}; only plugin params "
                       "(a key under 'params') can be added")
    obj[last] = value
    return path


def _child(obj, key, create=False):
    if isinstance(obj, list):
        for item in obj:
            if key in (item.get("layer"), item.get("name")):
                return item
        raise KeyError(f"no list item with layer/name {key!r}")
    if create and key not in obj:
        obj[key] = {}
    return obj[key]


# ---------------------------------------------------------------- migration
def migrate_v1(old: dict) -> dict:
    """Version-1 spec (first UCTF release) -> version 2, same behaviour."""
    old = copy.deepcopy(old)
    nr, inp = old.get("neuron", {}), old["input"]
    src = old["connectome"]
    loader = src.get("loader", "folder")
    params = {k: v for k, v in src.items() if k not in ("loader", "cut_inputs_to")}
    transforms = []
    if src.get("cut_inputs_to"):
        transforms.append({"transform": "cut_inputs", "params": {
            "population": src["cut_inputs_to"], "layers": ["chemical"]}})
    neuron_keys = ("dt", "tau", "threshold", "reset", "tonic",
                   "noise_hz", "noise_amp", "seed")
    neuron = {k: nr[k] for k in neuron_keys if k in nr}
    synapses = [{"layer": "chemical", "model": "chemical",
                 "params": {"gain": nr.get("gain", 1.5)}}]
    if nr.get("gap_gain"):
        synapses.append({"layer": "electrical", "model": "electrical",
                         "params": {"gain": nr["gap_gain"], "optional": True}})
    classes = old.get("classes", {})
    rewire_layers = {"chemical": "permute_presynaptic",
                     "electrical": "double_edge_swap"}
    counts = [r["name"] for r in old["readout"] if r["feature"] == "counts"]
    new = {
        "uctf": 2,
        "name": old.get("name", "custom"),
        "description": old.get("description", ""),
        "connectome": {"source": loader, "params": params},
        "transforms": transforms,
        "populations": old.get("populations", {}),
        "neuron": {"model": "lif", "params": neuron},
        "synapses": synapses,
        "input": {"encoder": "random_subset", "population": inp["population"],
                  "steps": int(inp.get("steps", 6)),
                  "params": {"active": int(inp["active"]),
                             "drive": float(inp.get("drive", 1.5)),
                             "sustain": float(inp.get("sustain", 0.5)),
                             "seed": int(inp.get("code_seed", 3))}},
        "readout": [{"name": r["name"], "feature": r["feature"],
                     "population": r["population"],
                     "params": ({"tau": r["tau"]} if "tau" in r else {})}
                    for r in old["readout"]],
        "controls": {
            "rewire-full": [{"transform": "rewire", "params": {
                "layers": rewire_layers, "classes": None,
                "skip_missing": True}}],
            "rewire-class": [{"transform": "rewire", "params": {
                "layers": rewire_layers, "classes": classes,
                "skip_missing": True}}],
        },
        "activity_match": {"param": "synapses.chemical.params.gain",
                           "readouts": counts},
        "view": old.get("view", {}),
    }
    return new


def migrate_control_v1(control: dict | None, spec: dict) -> list:
    """{"rewire": "full"|"class", "seed": s} (version 1) -> a transform list."""
    if not control or not control.get("rewire"):
        return []
    recipe = copy.deepcopy(spec["controls"][f"rewire-{control['rewire']}"])
    for step in recipe:
        step.setdefault("params", {})["seed"] = int(control.get("seed", 0))
    return recipe
