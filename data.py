"""Training examples for QuDrugGuard V2."""

from __future__ import annotations

from collections import Counter


TRAINING_PAIRS = [
    {"drug_a": "Warfarin", "drug_b": "Clarithromycin", "label": 1, "rationale": "CYP3A4 inhibition and bleeding amplification."},
    {"drug_a": "Warfarin", "drug_b": "Metronidazole", "label": 1, "rationale": "CYP2C9 inhibition markedly elevates warfarin exposure."},
    {"drug_a": "Simvastatin", "drug_b": "Clarithromycin", "label": 1, "rationale": "Severe CYP3A4 interaction raises statin toxicity risk."},
    {"drug_a": "Citalopram", "drug_b": "Ondansetron", "label": 1, "rationale": "Additive QT prolongation risk."},
    {"drug_a": "Tramadol", "drug_b": "Sertraline", "label": 1, "rationale": "Serotonergic toxicity and seizure burden."},
    {"drug_a": "Theophylline", "drug_b": "Ciprofloxacin", "label": 1, "rationale": "CYP1A2 inhibition increases theophylline levels."},
    {"drug_a": "Carbamazepine", "drug_b": "Clarithromycin", "label": 1, "rationale": "Metabolic inhibition with CNS toxicity."},
    {"drug_a": "Apixaban", "drug_b": "Diltiazem", "label": 1, "rationale": "P-gp and CYP3A4 interaction boosts anticoagulant exposure."},
    {"drug_a": "Linezolid", "drug_b": "Fluoxetine", "label": 1, "rationale": "High-risk serotonergic interaction."},
    {"drug_a": "Clopidogrel", "drug_b": "Omeprazole", "label": 1, "rationale": "CYP2C19 inhibition reduces antiplatelet activation."},
    {"drug_a": "Aspirin", "drug_b": "Warfarin", "label": 1, "rationale": "Combined bleeding pathway risk."},
    {"drug_a": "Phenytoin", "drug_b": "Metronidazole", "label": 1, "rationale": "Metabolic inhibition raises narrow-index drug levels."},
    {"drug_a": "Ziprasidone", "drug_b": "Ondansetron", "label": 1, "rationale": "Pronounced QT stacking."},
    {"drug_a": "Gemfibrozil", "drug_b": "Simvastatin", "label": 1, "rationale": "Transporter and metabolic interaction increases myopathy risk."},
    {"drug_a": "Amoxicillin", "drug_b": "Acetaminophen", "label": 0, "rationale": "Low mechanistic interaction burden."},
    {"drug_a": "Metformin", "drug_b": "Lisinopril", "label": 0, "rationale": "Common co-administration with manageable monitoring."},
    {"drug_a": "Cetirizine", "drug_b": "Amoxicillin", "label": 0, "rationale": "Minimal shared pathway overlap."},
    {"drug_a": "Rosuvastatin", "drug_b": "Metformin", "label": 0, "rationale": "Limited CYP involvement and low transporter conflict."},
    {"drug_a": "Levetiracetam", "drug_b": "Gabapentin", "label": 0, "rationale": "Predominantly renal handling with limited CYP burden."},
    {"drug_a": "Pantoprazole", "drug_b": "Amoxicillin", "label": 0, "rationale": "Low-risk regimen in routine practice."},
    {"drug_a": "Losartan", "drug_b": "Metformin", "label": 0, "rationale": "Low direct metabolic conflict."},
    {"drug_a": "Semaglutide", "drug_b": "Empagliflozin", "label": 0, "rationale": "Complementary antidiabetic therapy without direct enzyme conflict."},
    {"drug_a": "Pravastatin", "drug_b": "Lisinopril", "label": 0, "rationale": "Low shared toxicity burden."},
    {"drug_a": "Budesonide", "drug_b": "Albuterol", "label": 0, "rationale": "Standard respiratory combination therapy."},
    {"drug_a": "Famotidine", "drug_b": "Cetirizine", "label": 0, "rationale": "Minimal PK and PD overlap."},
    {"drug_a": "Memantine", "drug_b": "Donepezil", "label": 0, "rationale": "Common cognitive-support combination with manageable risk."},
    {"drug_a": "Amlodipine", "drug_b": "Hydrochlorothiazide", "label": 0, "rationale": "Low direct interaction burden."},
    {"drug_a": "Ezetimibe", "drug_b": "Rosuvastatin", "label": 0, "rationale": "Common lipid-lowering combination with limited enzyme overlap."},
]


def dataset_summary() -> dict:
    labels = Counter(item["label"] for item in TRAINING_PAIRS)
    return {
        "total_pairs": len(TRAINING_PAIRS),
        "dangerous_pairs": labels.get(1, 0),
        "safe_pairs": labels.get(0, 0),
    }


if __name__ == "__main__":
    summary = dataset_summary()
    print("QuDrugGuard training pairs:", summary["total_pairs"])
    print("Dangerous pairs:", summary["dangerous_pairs"])
    print("Safe pairs:", summary["safe_pairs"])
