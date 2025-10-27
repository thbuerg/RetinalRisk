COVARIATE_SETS = dict(
    AGESEX=[
        "age_at_recruitment",
        "sex",
    ],
    PANEL=[
        # basics
        "age_at_recruitment",
        "sex",
        "smoking_status",
        "alcohol_intake_frequency",
        # "physical_activity",
        # physical
        "body_mass_index_bmi",
        # 'waist_hip_ratio',
        "weight",
        "standing_height",
        "systolic_blood_pressure",
        # lipids
        "cholesterol",
        "ldl_direct",
        "hdl_cholesterol",
        "triglycerides",
        # diabetes
        "glucose",
        "glycated_haemoglobin_hba1c",
        # kidney
        "creatinine",
        "cystatin_c",
        "urea",
        "urate",
        # liver
        "aspartate_aminotransferase",
        "alanine_aminotransferase",
        "alkaline_phosphatase",
        "albumin",
        # inflammation
        "creactive_protein",
        # Blood counts
        "red_blood_cell_erythrocyte_count",
        "white_blood_cell_leukocyte_count",
        "platelet_count",
        "haemoglobin_concentration",
        "haematocrit_percentage",
        "mean_corpuscular_haemoglobin",
        "mean_corpuscular_volume",
        "mean_corpuscular_haemoglobin_concentration",
    ],
)
