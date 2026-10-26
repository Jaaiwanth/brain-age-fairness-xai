"""Phase 10l - Generate the model fairness passport from saved results and pre-specified criteria.

Reads the criteria fixed in release/passport_criteria.json (committed before the cross-validation
results existed) and the Phase 5-10 result files, applies the rules mechanically and writes
release/passport.json, a machine-readable passport. Re-running it on another model's result files
(same layout) produces that model's passport.

Usage: python scripts/29_fairness_passport.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["PASS", "INCONCLUSIVE", "CONCERN", "FAIL"]


def worst(verdicts):
    return max(verdicts, key=ORDER.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--criteria", type=Path, default=HERE.parent / "release" / "passport_criteria.json")
    p.add_argument("--out", type=Path, default=HERE.parent / "release" / "passport.json")
    args = p.parse_args()
    R = args.project_dir / "results"
    crit = json.loads(args.criteria.read_text())
    margin = crit["equivalence_margin_years"]
    checks = {}

    acc = json.loads((R / "phase5" / "summary.json").read_text())["raw"]
    mae_hi, r_lo = acc["ci95"]["mae"][1], acc["ci95"]["r"][0]
    checks["accuracy"] = {"verdict": "PASS" if mae_hi <= 5 and r_lo >= 0.90 else "CONCERN",
                          "evidence": {"test_mae": acc["metrics"]["mae"], "mae_ci_upper": mae_hi,
                                       "r": acc["metrics"]["r"], "r_ci_lower": r_lo}}

    v = pd.read_csv(R / "phase10d_verdicts" / "graded_verdicts.csv")
    primary = v[v.analysis.str.startswith("5-fold CV")]
    if primary.empty:
        primary = v[v.analysis.str.startswith("main model")]
    for name, comps in [("fairness_sex", ["Male vs Female"]), ("fairness_site", ["HH vs Guys", "IOP vs Guys"])]:
        rows = primary[primary.comparison.isin(comps)]
        checks[name] = {"verdict": worst(rows.verdict.tolist()),
                        "evidence": rows[["comparison", "outcome", "est", "ci95", "verdict"]].to_dict("records")}

    swap = pd.read_csv(R / "phase9b_stress_test" / "site_swap_effects.csv")
    swap = swap[~swap.own_site_control]
    own = pd.read_csv(R / "phase9b_stress_test" / "site_swap_effects.csv").set_index("source_site")
    swap["vs_own"] = swap.mean_delta - swap.source_site.map(own[own.own_site_control].mean_delta)
    cv = json.loads((R / "phase10k_cv_stress_test" / "summary.json").read_text())
    rep = {(m["site"], m["look"]): m["mean_delta"] for m in cv["matrix"]}
    worst_pair = swap.loc[swap.vs_own.abs().idxmax()]
    replicated = (rep.get((worst_pair.source_site, worst_pair.target_site), 0) * worst_pair.vs_own) > 0
    checks["scanner_invariance"] = {
        "verdict": "FAIL" if abs(worst_pair.vs_own) >= margin and replicated else "CONCERN",
        "evidence": {"worst_pair": f"{worst_pair.source_site} -> {worst_pair.target_site}",
                     "change_vs_own_site": float(worst_pair.vs_own), "replicated_in_cv_models": bool(replicated),
                     "cv_change": rep.get((worst_pair.source_site, worst_pair.target_site))}}

    erase = json.loads((R / "phase10f_site_erasure" / "summary.json").read_text())
    probe = next(x for x in erase["versions"] if x["features"] == "original")["probe_bal_acc"]
    checks["site_encoding"] = {"verdict": "INFORMATIONAL",
                               "evidence": {"probe_balanced_accuracy": probe,
                                            "permutation_null_p95": erase["probe_permutation_null"]["p95"],
                                            "encoded": probe > erase["probe_permutation_null"]["p95"]}}

    rho = json.loads((R / "phase7" / "summary.json").read_text())["sanity"]["median"]
    dele = json.loads((R / "phase10g_deletion_baselines" / "summary.json").read_text())["tests"]
    ok = rho < 0.5 and all(t["p_top_vs_random"] < 0.05 for t in dele.values())
    checks["faithful_explanations"] = {"verdict": "PASS" if ok else "CONCERN",
                                       "evidence": {"randomisation_spearman": rho, "deletion": dele}}

    rel = pd.read_csv(R / "phase10i_uncertainty" / "reliability_by_site.csv")
    own90 = rel[(rel.method == "model's own distribution") & (rel.level == 0.9)].set_index("site").coverage
    if own90.between(0.85, 0.95).all():
        u = "PASS"
    elif own90.between(0.80, 0.97).all():
        u = "CONCERN"
    else:
        u = "FAIL"
    mond = rel[(rel.method.str.contains("Mondrian")) & (rel.level == 0.9)].set_index("site").coverage
    checks["honest_uncertainty"] = {"verdict": u, "evidence": {"own_90_coverage": own90.round(3).to_dict(),
                                                              "per_site_conformal_90_coverage": mond.round(3).to_dict()}}

    passport = {"model": "SFCN (UK Biobank-pretrained), fine-tuned on IXI", "criteria_file": str(args.criteria.name),
                "criteria_fixed_on": crit["fixed_on"], "checks": checks,
                "usage": {"suitable_for": ["research within a single acquisition site", "teaching and methods development"],
                          "not_suitable_for": ["clinical or individual decisions",
                                               "comparing brain-age gaps across scanners without site-aware validation",
                                               "new sites without per-site calibration of uncertainty"]}}
    args.out.write_text(json.dumps(passport, indent=2, default=float))
    for k, c in checks.items():
        print(f"{k:24s} {c['verdict']}")


if __name__ == "__main__":
    main()
