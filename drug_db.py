"""Drug catalog and interaction feature engineering for QuDrugGuard V2."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any


DRUG_CATEGORIES = {
    "Analgesics": ["Acetaminophen", "Ibuprofen", "Naproxen", "Diclofenac", "Celecoxib", "Tramadol", "Morphine", "Oxycodone", "Codeine", "Aspirin"],
    "Antibiotics": ["Amoxicillin", "Azithromycin", "Ciprofloxacin", "Levofloxacin", "Doxycycline", "Clarithromycin", "Metronidazole", "Linezolid", "Trimethoprim", "Clindamycin"],
    "Antihypertensives": ["Amlodipine", "Losartan", "Valsartan", "Lisinopril", "Enalapril", "Metoprolol", "Carvedilol", "Diltiazem", "Verapamil", "Hydrochlorothiazide"],
    "Antidiabetics": ["Metformin", "Glipizide", "Glyburide", "Pioglitazone", "Sitagliptin", "Empagliflozin", "Dapagliflozin", "Insulin Glargine", "Semaglutide", "Linagliptin"],
    "Anticoagulants": ["Warfarin", "Apixaban", "Rivaroxaban", "Dabigatran", "Heparin", "Enoxaparin", "Clopidogrel", "Ticagrelor", "Prasugrel", "Edoxaban"],
    "Antidepressants": ["Sertraline", "Fluoxetine", "Escitalopram", "Citalopram", "Venlafaxine", "Duloxetine", "Bupropion", "Amitriptyline", "Paroxetine", "Mirtazapine"],
    "Antipsychotics": ["Quetiapine", "Risperidone", "Olanzapine", "Haloperidol", "Aripiprazole", "Clozapine", "Ziprasidone", "Lurasidone", "Chlorpromazine", "Paliperidone"],
    "Antihistamines": ["Cetirizine", "Levocetirizine", "Loratadine", "Fexofenadine", "Diphenhydramine", "Hydroxyzine", "Chlorpheniramine", "Desloratadine", "Promethazine", "Meclizine"],
    "Respiratory": ["Albuterol", "Salmeterol", "Fluticasone", "Budesonide", "Prednisone", "Tiotropium", "Ipratropium", "Theophylline", "Benzonatate", "Guaifenesin"],
    "Gastrointestinal": ["Omeprazole", "Pantoprazole", "Famotidine", "Ondansetron", "Metoclopramide", "Loperamide", "Sucralfate", "Mesalamine", "Domperidone", "Simethicone"],
    "LipidLowering": ["Atorvastatin", "Rosuvastatin", "Simvastatin", "Pravastatin", "Fenofibrate", "Ezetimibe", "Niacin", "Gemfibrozil", "Alirocumab", "Evolocumab"],
    "Neurology": ["Levetiracetam", "Carbamazepine", "Lamotrigine", "Phenytoin", "Valproate", "Gabapentin", "Pregabalin", "Donepezil", "Memantine", "Topiramate"],
}


CATEGORY_PROFILES = {
    "Analgesics": {"primary": "CYP2C9", "secondary": ["UGT", "CYP2D6"], "transporter": "P-gp", "solubility": 2.4, "logp": 2.3, "binding": 80, "hepatic": 62, "renal": 36},
    "Antibiotics": {"primary": "CYP3A4", "secondary": ["CYP1A2", "UGT"], "transporter": "P-gp", "solubility": 6.5, "logp": 1.7, "binding": 55, "hepatic": 48, "renal": 50},
    "Antihypertensives": {"primary": "CYP3A4", "secondary": ["CYP2D6", "CYP2C9"], "transporter": "P-gp", "solubility": 3.2, "logp": 2.1, "binding": 72, "hepatic": 58, "renal": 40},
    "Antidiabetics": {"primary": "CYP2C9", "secondary": ["CYP3A4", "UGT"], "transporter": "OCT2", "solubility": 4.6, "logp": 1.4, "binding": 62, "hepatic": 44, "renal": 54},
    "Anticoagulants": {"primary": "CYP2C9", "secondary": ["CYP3A4", "P-gp"], "transporter": "P-gp", "solubility": 2.0, "logp": 2.6, "binding": 86, "hepatic": 60, "renal": 38},
    "Antidepressants": {"primary": "CYP2D6", "secondary": ["CYP2C19", "CYP3A4"], "transporter": "P-gp", "solubility": 2.8, "logp": 2.5, "binding": 78, "hepatic": 64, "renal": 30},
    "Antipsychotics": {"primary": "CYP3A4", "secondary": ["CYP2D6", "CYP1A2"], "transporter": "P-gp", "solubility": 1.8, "logp": 3.0, "binding": 84, "hepatic": 68, "renal": 28},
    "Antihistamines": {"primary": "CYP3A4", "secondary": ["CYP2D6", "P-gp"], "transporter": "P-gp", "solubility": 3.5, "logp": 2.1, "binding": 70, "hepatic": 52, "renal": 42},
    "Respiratory": {"primary": "CYP3A4", "secondary": ["CYP1A2", "UGT"], "transporter": "P-gp", "solubility": 4.8, "logp": 1.9, "binding": 58, "hepatic": 46, "renal": 50},
    "Gastrointestinal": {"primary": "CYP2C19", "secondary": ["CYP3A4", "P-gp"], "transporter": "P-gp", "solubility": 5.6, "logp": 1.6, "binding": 60, "hepatic": 54, "renal": 44},
    "LipidLowering": {"primary": "CYP3A4", "secondary": ["CYP2C9", "OATP1B1"], "transporter": "OATP1B1", "solubility": 1.6, "logp": 3.4, "binding": 88, "hepatic": 70, "renal": 24},
    "Neurology": {"primary": "CYP2C19", "secondary": ["CYP3A4", "UGT"], "transporter": "P-gp", "solubility": 3.1, "logp": 2.0, "binding": 64, "hepatic": 50, "renal": 48},
}


INTERACTION_FEATURES = [
    "shared_enzyme_count", "inhibitor_overlap_score", "inducer_overlap_score", "transporter_conflict",
    "solubility_gap", "protein_binding_shift", "hepatic_burden", "renal_burden", "qt_stack", "cns_stack",
    "serotonin_stack", "bleeding_stack", "nephro_stack", "hepatotoxic_stack", "narrow_therapeutic_penalty",
    "category_distance",
]


ENZYME_WEIGHT = {"CYP3A4": 1.0, "CYP2D6": 0.9, "CYP2C9": 0.95, "CYP2C19": 0.85, "CYP1A2": 0.8, "UGT": 0.7, "P-gp": 0.75, "OCT2": 0.55, "OATP1B1": 0.7}


OVERRIDES = {
    "Warfarin": {"primary_enzyme": "CYP2C9", "secondary_enzymes": ["CYP3A4", "CYP1A2"], "narrow_therapeutic_index": 1, "bleeding_risk": 0.98, "protein_binding_pct": 99, "half_life_hr": 36, "hepatic_clearance_pct": 82, "renal_clearance_pct": 14},
    "Clarithromycin": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["P-gp", "CYP1A2"], "inhibitor_strength": 0.95, "qt_risk": 0.55},
    "Metronidazole": {"primary_enzyme": "CYP2C9", "secondary_enzymes": ["UGT"], "inhibitor_strength": 0.82},
    "Simvastatin": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["OATP1B1"], "transporter": "OATP1B1", "hepatotoxicity": 0.52, "protein_binding_pct": 95},
    "Citalopram": {"primary_enzyme": "CYP2C19", "secondary_enzymes": ["CYP3A4", "CYP2D6"], "qt_risk": 0.72, "serotonergic": 0.88},
    "Ondansetron": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["CYP2D6", "CYP1A2"], "qt_risk": 0.78},
    "Tramadol": {"primary_enzyme": "CYP2D6", "secondary_enzymes": ["CYP3A4"], "cns_depressant": 0.68, "serotonergic": 0.54},
    "Sertraline": {"primary_enzyme": "CYP2D6", "secondary_enzymes": ["CYP2C19", "CYP3A4"], "serotonergic": 0.9, "inhibitor_strength": 0.48},
    "Theophylline": {"primary_enzyme": "CYP1A2", "secondary_enzymes": ["CYP3A4"], "narrow_therapeutic_index": 1, "qt_risk": 0.25},
    "Ciprofloxacin": {"primary_enzyme": "CYP1A2", "secondary_enzymes": ["P-gp"], "inhibitor_strength": 0.74, "qt_risk": 0.42},
    "Carbamazepine": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["UGT"], "inducer_strength": 0.92, "narrow_therapeutic_index": 1, "cns_depressant": 0.42},
    "Apixaban": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["P-gp"], "bleeding_risk": 0.84, "protein_binding_pct": 87},
    "Diltiazem": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["P-gp"], "inhibitor_strength": 0.67},
    "Linezolid": {"primary_enzyme": "UGT", "secondary_enzymes": ["MAO"], "serotonergic": 0.76, "hepatotoxicity": 0.2},
    "Fluoxetine": {"primary_enzyme": "CYP2D6", "secondary_enzymes": ["CYP2C19", "CYP3A4"], "serotonergic": 0.95, "inhibitor_strength": 0.86, "half_life_hr": 72},
    "Clopidogrel": {"primary_enzyme": "CYP2C19", "secondary_enzymes": ["CYP3A4"], "bleeding_risk": 0.76},
    "Omeprazole": {"primary_enzyme": "CYP2C19", "secondary_enzymes": ["CYP3A4"], "inhibitor_strength": 0.7},
    "Phenytoin": {"primary_enzyme": "CYP2C19", "secondary_enzymes": ["CYP2C9"], "narrow_therapeutic_index": 1, "inducer_strength": 0.62},
    "Ziprasidone": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["CYP1A2"], "qt_risk": 0.86},
    "Gemfibrozil": {"primary_enzyme": "UGT", "secondary_enzymes": ["OATP1B1"], "inhibitor_strength": 0.7, "transporter": "OATP1B1"},
    "Acetaminophen": {"primary_enzyme": "UGT", "secondary_enzymes": ["CYP2E1"], "hepatotoxicity": 0.46, "solubility_mg_ml": 14.0, "protein_binding_pct": 15},
    "Amoxicillin": {"primary_enzyme": "None", "secondary_enzymes": [], "solubility_mg_ml": 10.2, "renal_clearance_pct": 68, "hepatic_clearance_pct": 12, "transporter": "None"},
    "Metformin": {"primary_enzyme": "None", "secondary_enzymes": ["OCT2"], "solubility_mg_ml": 95.0, "renal_clearance_pct": 88, "hepatic_clearance_pct": 5, "transporter": "OCT2"},
    "Lisinopril": {"primary_enzyme": "None", "secondary_enzymes": [], "renal_clearance_pct": 76, "hepatic_clearance_pct": 8, "transporter": "None"},
    "Levetiracetam": {"primary_enzyme": "None", "secondary_enzymes": [], "renal_clearance_pct": 72, "hepatic_clearance_pct": 12, "transporter": "None"},
    "Gabapentin": {"primary_enzyme": "None", "secondary_enzymes": [], "renal_clearance_pct": 84, "hepatic_clearance_pct": 4, "transporter": "LAT1"},
    "Semaglutide": {"primary_enzyme": "Proteolysis", "secondary_enzymes": [], "transporter": "None", "protein_binding_pct": 99, "half_life_hr": 168},
    "Empagliflozin": {"primary_enzyme": "UGT", "secondary_enzymes": ["P-gp"], "renal_clearance_pct": 54},
    "Rosuvastatin": {"primary_enzyme": "None", "secondary_enzymes": ["OATP1B1"], "transporter": "OATP1B1", "hepatic_clearance_pct": 38, "renal_clearance_pct": 30},
    "Budesonide": {"primary_enzyme": "CYP3A4", "secondary_enzymes": ["UGT"], "hepatic_clearance_pct": 70},
    "Albuterol": {"primary_enzyme": "SULT", "secondary_enzymes": [], "transporter": "None"},
    "Famotidine": {"primary_enzyme": "None", "secondary_enzymes": [], "renal_clearance_pct": 72, "hepatic_clearance_pct": 8},
    "Donepezil": {"primary_enzyme": "CYP2D6", "secondary_enzymes": ["CYP3A4"], "qt_risk": 0.4},
    "Hydrochlorothiazide": {"primary_enzyme": "None", "secondary_enzymes": [], "renal_clearance_pct": 82, "hepatic_clearance_pct": 4},
    "Pravastatin": {"primary_enzyme": "None", "secondary_enzymes": ["OATP1B1"], "transporter": "OATP1B1"},
    "Ezetimibe": {"primary_enzyme": "UGT", "secondary_enzymes": [], "transporter": "OATP1B1"},
    "Aspirin": {"primary_enzyme": "UGT", "secondary_enzymes": ["CYP2C9"], "bleeding_risk": 0.8, "protein_binding_pct": 90},
}


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _scale(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _base_properties(category: str, drug_name: str, index: int) -> dict[str, Any]:
    profile = CATEGORY_PROFILES[category]
    name_score = sum(ord(char) for char in drug_name)
    modulation = ((name_score % 13) - 6) / 10.0
    solubility = max(0.1, profile["solubility"] + modulation + (index % 3) * 0.35)
    logp = max(0.1, profile["logp"] + ((index % 5) - 2) * 0.18)
    protein_binding = _scale(profile["binding"] + ((name_score % 17) - 8), 5, 99)
    hepatic = _scale(profile["hepatic"] + ((index % 4) - 1.5) * 6, 2, 90)
    renal = _scale(profile["renal"] + ((index % 5) - 2) * 5, 2, 95)
    toxicity_bias = max(0.05, (abs(modulation) + 0.25) / 2)
    return {
        "name": drug_name,
        "category": category,
        "primary_enzyme": profile["primary"],
        "secondary_enzymes": list(profile["secondary"]),
        "transporter": profile["transporter"],
        "solubility_mg_ml": round(solubility, 2),
        "logp": round(logp, 2),
        "protein_binding_pct": round(protein_binding, 2),
        "half_life_hr": round(4 + (name_score % 19) * 1.8, 2),
        "hepatic_clearance_pct": round(hepatic, 2),
        "renal_clearance_pct": round(renal, 2),
        "inhibitor_strength": round(_scale(0.12 + (name_score % 11) / 20, 0, 0.95), 2),
        "inducer_strength": round(_scale(0.08 + (name_score % 7) / 25, 0, 0.95), 2),
        "qt_risk": round(_scale(0.06 + toxicity_bias * 0.35, 0, 0.95), 2),
        "cns_depressant": round(_scale(0.03 + toxicity_bias * 0.4, 0, 0.95), 2),
        "serotonergic": round(_scale(0.02 + toxicity_bias * 0.25, 0, 0.95), 2),
        "bleeding_risk": round(_scale(0.04 + toxicity_bias * 0.28, 0, 0.98), 2),
        "nephrotoxicity": round(_scale(0.03 + (renal / 100) * 0.3, 0, 0.9), 2),
        "hepatotoxicity": round(_scale(0.03 + (hepatic / 100) * 0.28, 0, 0.9), 2),
        "narrow_therapeutic_index": 1 if drug_name in {"Warfarin", "Phenytoin", "Theophylline", "Carbamazepine"} else 0,
    }


def _apply_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    merged.update(override)
    return merged


def _build_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    catalog: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category, drugs in DRUG_CATEGORIES.items():
        for index, drug_name in enumerate(drugs):
            item = _base_properties(category, drug_name, index)
            if drug_name in OVERRIDES:
                item = _apply_override(item, OVERRIDES[drug_name])
            item["enzyme_set"] = sorted(enzyme for enzyme in [item["primary_enzyme"], *item.get("secondary_enzymes", [])] if enzyme and enzyme != "None")
            catalog[_normalize_name(drug_name)] = item
            grouped[category].append(item)
    return catalog, dict(grouped)


DRUG_LOOKUP, DRUG_DATA = _build_catalog()


def list_all_drugs() -> list[str]:
    return sorted(item["name"] for item in DRUG_LOOKUP.values())


def get_drug_properties(drug_name: str) -> dict[str, Any]:
    key = _normalize_name(drug_name)
    if key not in DRUG_LOOKUP:
        raise KeyError(f"Unknown drug: {drug_name}")
    return deepcopy(DRUG_LOOKUP[key])


def search_drugs(query: str, limit: int = 20) -> list[dict[str, Any]]:
    query_norm = _normalize_name(query)
    if not query_norm:
        return [deepcopy(item) for item in sorted(DRUG_LOOKUP.values(), key=lambda item: item["name"])[:limit]]
    matches = [deepcopy(item) for key, item in DRUG_LOOKUP.items() if query_norm in key or query_norm in item["category"].lower()]
    return sorted(matches, key=lambda item: (item["category"], item["name"]))[:limit]


def check_shared_enzyme(drug_a: str, drug_b: str) -> dict[str, Any]:
    left = get_drug_properties(drug_a)
    right = get_drug_properties(drug_b)
    shared = sorted(set(left["enzyme_set"]) & set(right["enzyme_set"]))
    enzyme_weight = round(sum(ENZYME_WEIGHT.get(item, 0.5) for item in shared), 2)
    return {
        "shared": shared,
        "count": len(shared),
        "weighted_overlap": enzyme_weight,
        "primary_conflict": left["primary_enzyme"] == right["primary_enzyme"] and left["primary_enzyme"] not in {"None", "Proteolysis", "SULT"},
    }


def _category_distance(category_a: str, category_b: str) -> float:
    if category_a == category_b:
        return 0.0
    categories = list(DRUG_CATEGORIES.keys())
    return abs(categories.index(category_a) - categories.index(category_b)) / (len(categories) - 1)


def build_pair_feature_vector(drug_a: str, drug_b: str) -> dict[str, Any]:
    left = get_drug_properties(drug_a)
    right = get_drug_properties(drug_b)
    overlap = check_shared_enzyme(drug_a, drug_b)
    inhibitor_overlap = 0.0
    inducer_overlap = 0.0
    left_targets = set(right["enzyme_set"])
    right_targets = set(left["enzyme_set"])
    if left["primary_enzyme"] in left_targets:
        inhibitor_overlap += left["inhibitor_strength"]
        inducer_overlap += left["inducer_strength"]
    if right["primary_enzyme"] in right_targets:
        inhibitor_overlap += right["inhibitor_strength"]
        inducer_overlap += right["inducer_strength"]
    transporter_conflict = 1.0 if left["transporter"] == right["transporter"] and left["transporter"] not in {"None", ""} else 0.0
    solubility_gap = abs(math.log1p(left["solubility_mg_ml"]) - math.log1p(right["solubility_mg_ml"]))
    protein_binding_shift = abs(left["protein_binding_pct"] - right["protein_binding_pct"]) / 100
    vector = {
        "shared_enzyme_count": float(overlap["count"]),
        "inhibitor_overlap_score": round(inhibitor_overlap, 4),
        "inducer_overlap_score": round(inducer_overlap, 4),
        "transporter_conflict": transporter_conflict,
        "solubility_gap": round(solubility_gap, 4),
        "protein_binding_shift": round(protein_binding_shift, 4),
        "hepatic_burden": round((left["hepatic_clearance_pct"] + right["hepatic_clearance_pct"]) / 200, 4),
        "renal_burden": round((left["renal_clearance_pct"] + right["renal_clearance_pct"]) / 200, 4),
        "qt_stack": round(min(1.0, left["qt_risk"] + right["qt_risk"]), 4),
        "cns_stack": round(min(1.0, left["cns_depressant"] + right["cns_depressant"]), 4),
        "serotonin_stack": round(min(1.0, left["serotonergic"] + right["serotonergic"]), 4),
        "bleeding_stack": round(min(1.0, left["bleeding_risk"] + right["bleeding_risk"]), 4),
        "nephro_stack": round(min(1.0, left["nephrotoxicity"] + right["nephrotoxicity"]), 4),
        "hepatotoxic_stack": round(min(1.0, left["hepatotoxicity"] + right["hepatotoxicity"]), 4),
        "narrow_therapeutic_penalty": float(left["narrow_therapeutic_index"] or right["narrow_therapeutic_index"]),
        "category_distance": round(_category_distance(left["category"], right["category"]), 4),
    }
    drivers = []
    if overlap["shared"]:
        drivers.append(f"Shared metabolic pathways: {', '.join(overlap['shared'])}")
    if vector["bleeding_stack"] > 0.75:
        drivers.append("High combined bleeding liability")
    if vector["qt_stack"] > 0.75:
        drivers.append("High combined QT prolongation burden")
    if vector["serotonin_stack"] > 0.75:
        drivers.append("High combined serotonergic burden")
    if vector["transporter_conflict"]:
        drivers.append(f"Shared transporter pressure on {left['transporter']}")
    if vector["solubility_gap"] > 1:
        drivers.append("Marked solubility mismatch may alter absorption profile")
    if vector["narrow_therapeutic_penalty"]:
        drivers.append("One of the drugs has a narrow therapeutic index")
    return {
        "drug_a": left,
        "drug_b": right,
        "shared_enzyme": overlap,
        "feature_vector": vector,
        "feature_list": [vector[name] for name in INTERACTION_FEATURES],
        "drivers": drivers or ["No dominant mechanistic conflicts detected"],
    }


def summarize_catalog() -> dict[str, Any]:
    return {"total_drugs": len(DRUG_LOOKUP), "total_categories": len(DRUG_CATEGORIES), "categories": {category: len(drugs) for category, drugs in DRUG_CATEGORIES.items()}}


def self_test(verbose: bool = True) -> dict[str, Any]:
    summary = summarize_catalog()
    enzyme_check = check_shared_enzyme("Warfarin", "Metronidazole")
    search_check = search_drugs("statin", limit=5)
    result = {"summary": summary, "enzyme_check": enzyme_check, "search_results": [item["name"] for item in search_check], "passed": summary["total_drugs"] == 120 and summary["total_categories"] == 12 and enzyme_check["count"] >= 1}
    if verbose:
        print("Total drugs:", summary["total_drugs"])
        print("Total categories:", summary["total_categories"])
        print("Warfarin/Metronidazole shared enzymes:", ", ".join(enzyme_check["shared"]) or "None")
        print("Search sample:", ", ".join(result["search_results"]))
    return result


if __name__ == "__main__":
    self_test(verbose=True)
