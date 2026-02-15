import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

sensationalism = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "sensationalism")
evidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "evidence")
hedge = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "hedge")
noise = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "noise")

# Consequent
fake_score = ctrl.Consequent(np.arange(0, 1.01, 0.01), "fake_score")

sensationalism["low"] = fuzz.trimf(sensationalism.universe, [0, 0, 0.7])
sensationalism["high"] = fuzz.trimf(sensationalism.universe, [0.3, 1, 1])

evidence["low"] = fuzz.trimf(evidence.universe, [0, 0, 0.5])
evidence["high"] = fuzz.trimf(evidence.universe, [0.4, 1, 1])

hedge["low"] = fuzz.trimf(hedge.universe, [0, 0, 0.7])
hedge["high"] = fuzz.trimf(hedge.universe, [0.4, 1, 1])

noise["low"] = fuzz.trimf(noise.universe, [0, 0, 0.7])
noise["high"] = fuzz.trimf(noise.universe, [0.4, 1, 1])

# output memberships (slightly balanced)
fake_score["real_like"] = fuzz.trimf(fake_score.universe, [0, 0, 0.45])
fake_score["maybe"] = fuzz.trimf(fake_score.universe, [0.3, 0.5, 0.7])
fake_score["fake_like"] = fuzz.trimf(fake_score.universe, [0.55, 1, 1])

# rules
rules = [
    # strong FAKE when everything suspicious
    ctrl.Rule(
        sensationalism["high"] & evidence["low"] & hedge["high"] & noise["high"],
        fake_score["fake_like"],
    ),
    # FAKE when sensational + weak evidence + suspicious tone
    ctrl.Rule(
        sensationalism["high"] & evidence["low"] & (hedge["high"] | noise["high"]),
        fake_score["fake_like"],
    ),
    # mild FAKE when any suspicious signal appears
    ctrl.Rule(
        sensationalism["high"] | hedge["high"] | noise["high"],
        fake_score["fake_like"],
    ),
    # REAL when strong evidence + low sensationalism (simplified, important)
    ctrl.Rule(
        evidence["high"] & sensationalism["low"],
        fake_score["real_like"],
    ),
    # neutral / uncertain
    ctrl.Rule(
        sensationalism["low"] & evidence["low"],
        fake_score["maybe"],
    ),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)


# normalization (NO sigmoid)


def normalize01(x, clip_low=0.0, clip_high=1.0):
    return float(np.clip(x, clip_low, clip_high))


# inference
def compute_fuzzy_score(example_feature_dict):

    sim.reset()  # critical to avoid state carryover

    s = normalize01(example_feature_dict.get("sensationalism", 0))
    e = normalize01(example_feature_dict.get("evidence", 0))
    h = normalize01(example_feature_dict.get("hedge", 0))
    n = normalize01(example_feature_dict.get("noise", 0))

    sim.input["sensationalism"] = s
    sim.input["evidence"] = e
    sim.input["hedge"] = h
    sim.input["noise"] = n

    try:
        sim.compute()
        score = float(sim.output["fake_score"])
    except Exception:
        # fb heuristic
        score = float(0.6 * s + 0.3 * (1 - e) + 0.2 * n)

    return float(np.clip(score, 0.0, 1.0))
