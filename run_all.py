"""Integration runner for QuDrugGuard V2."""

from __future__ import annotations

from pathlib import Path

import auth
import drug_db
import train


def main() -> None:
    auth_result = auth.self_test(verbose=False)
    if not auth_result["passed"]:
        raise SystemExit("Auth self-test failed.")
    print("Auth module check: OK")

    drug_result = drug_db.self_test(verbose=False)
    if not drug_result["passed"]:
        raise SystemExit("Drug database self-test failed.")
    print("Drug database check: OK")

    model_bundle = train.load_model()
    if not Path("qudrug_model.pkl").exists():
        raise SystemExit("Model file was not created.")
    print("Model check: OK")

    high_risk = train.predict_interaction("Warfarin", "Clarithromycin", model_bundle)
    low_risk = train.predict_interaction("Amoxicillin", "Acetaminophen", model_bundle)
    if high_risk["risk_score"] <= low_risk["risk_score"]:
        raise SystemExit("Prediction ordering sanity check failed.")
    print("Inference sanity check: OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
