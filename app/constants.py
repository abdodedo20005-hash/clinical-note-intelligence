"""Label -> group mapping and the class order of the trained classifier (copied from the notebook)."""

LABEL_TO_GROUP = {
    "Surgery": "Surgery (Operative Notes)",
    "Cosmetic / Plastic Surgery": "Surgery (Operative Notes)",
    "Radiology": "Radiology",
    "Consult - History and Phy.": "General Clinical Notes",
    "General Medicine": "General Clinical Notes",
    "SOAP / Chart / Progress Notes": "General Clinical Notes",
    "Office Notes": "General Clinical Notes",
    "Discharge Summary": "General Clinical Notes",
    "Emergency Room Reports": "General Clinical Notes",
    "Letters": "General Clinical Notes",
    "IME-QME-Work Comp etc.": "General Clinical Notes",
    "Orthopedic": "Musculoskeletal & Rehab",
    "Podiatry": "Musculoskeletal & Rehab",
    "Chiropractic": "Musculoskeletal & Rehab",
    "Physical Medicine - Rehab": "Musculoskeletal & Rehab",
    "Rheumatology": "Musculoskeletal & Rehab",
    "Pain Management": "Musculoskeletal & Rehab",
    "Cardiovascular / Pulmonary": "Cardiopulmonary & Sleep",
    "Sleep Medicine": "Cardiopulmonary & Sleep",
    "Neurology": "Neurology & Mental Health",
    "Neurosurgery": "Neurology & Mental Health",
    "Psychiatry / Psychology": "Neurology & Mental Health",
    "Speech - Language": "Neurology & Mental Health",
    "Gastroenterology": "Digestive & Metabolic",
    "Endocrinology": "Digestive & Metabolic",
    "Bariatrics": "Digestive & Metabolic",
    "Diets and Nutritions": "Digestive & Metabolic",
    "Urology": "Genitourinary & Renal",
    "Nephrology": "Genitourinary & Renal",
    "Obstetrics / Gynecology": "Women's & Children's Health",
    "Pediatrics - Neonatal": "Women's & Children's Health",
    "ENT - Otolaryngology": "Head, Eye, Ear, Skin & Dental",
    "Ophthalmology": "Head, Eye, Ear, Skin & Dental",
    "Dentistry": "Head, Eye, Ear, Skin & Dental",
    "Dermatology": "Head, Eye, Ear, Skin & Dental",
    "Allergy / Immunology": "Head, Eye, Ear, Skin & Dental",
    "Hematology - Oncology": "Oncology & Pathology",
    "Lab Medicine - Pathology": "Oncology & Pathology",
    "Autopsy": "Oncology & Pathology",
    "Hospice - Palliative Care": "Oncology & Pathology",
}

# Column order of the classifier's output = order of the "Notes per group" table in the notebook
# (groups sorted by how many label occurrences they hold). Do NOT reorder.
GROUPS = [
    "General Clinical Notes",
    "Surgery (Operative Notes)",
    "Musculoskeletal & Rehab",
    "Cardiopulmonary & Sleep",
    "Neurology & Mental Health",
    "Radiology",
    "Digestive & Metabolic",
    "Head, Eye, Ear, Skin & Dental",
    "Genitourinary & Renal",
    "Women's & Children's Health",
    "Oncology & Pathology",
]

# Decision threshold tuned on the validation set in the notebook (the top-1 group is always returned).
THRESHOLD = 0.35

assert set(LABEL_TO_GROUP.values()) == set(GROUPS)
