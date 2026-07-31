USE bone_tumor;
GO

/*==============================================================================
STAGE 01 - DATA CLEANING.
==============================================================================*/

IF COL_LENGTH('dbo.Bone_Tumor1', 'age_int') IS NULL
BEGIN
    ALTER TABLE dbo.Bone_Tumor1
    ADD age_int INT;
END
GO

IF COL_LENGTH('dbo.Bone_Tumor1', 'age_group') IS NULL
BEGIN
    ALTER TABLE dbo.Bone_Tumor1
    ADD age_group VARCHAR(20);
END
GO

UPDATE dbo.Bone_Tumor1
SET
    age_int = TRY_CAST(age AS INT),
    age_group =
        CASE
            WHEN TRY_CAST(age AS INT) IS NULL THEN 'Unknown'
            WHEN TRY_CAST(age AS INT) < 20 THEN 'Pediatric'
            WHEN TRY_CAST(age AS INT) BETWEEN 20 AND 40 THEN 'YoungAdult'
            WHEN TRY_CAST(age AS INT) BETWEEN 41 AND 60 THEN 'Adult'
            ELSE 'Elderly'
        END;
GO

/*==============================================================================
STAGE 02 - DATA QUALITY CHECKS
==============================================================================*/

SELECT
    COUNT(*) AS TotalRows,
    COUNT(DISTINCT patient_id) AS UniquePatients
FROM dbo.Bone_Tumor1;
GO

SELECT
    patient_id,
    COUNT(*) AS DuplicateCount
FROM dbo.Bone_Tumor1
GROUP BY patient_id
HAVING COUNT(*) > 1;
GO

SELECT
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS age_null,
    SUM(CASE WHEN age_int IS NULL THEN 1 ELSE 0 END) AS age_int_null,
    SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) AS sex_null,
    SUM(CASE WHEN grade IS NULL THEN 1 ELSE 0 END) AS grade_null,
    SUM(CASE WHEN histology IS NULL THEN 1 ELSE 0 END) AS histology_null,
    SUM(CASE WHEN treatment IS NULL THEN 1 ELSE 0 END) AS treatment_null,
    SUM(CASE WHEN [primary_site ] IS NULL THEN 1 ELSE 0 END) AS primary_site_null,
    SUM(CASE WHEN mskcc_type IS NULL THEN 1 ELSE 0 END) AS mskcc_type_null,
    SUM(CASE WHEN [outcome_status ] IS NULL THEN 1 ELSE 0 END) AS outcome_status_null
FROM dbo.Bone_Tumor1;
GO

SELECT DISTINCT sex
FROM dbo.Bone_Tumor1
ORDER BY sex;
GO

SELECT DISTINCT grade
FROM dbo.Bone_Tumor1
ORDER BY grade;
GO

SELECT DISTINCT histology
FROM dbo.Bone_Tumor1
ORDER BY histology;
GO

SELECT DISTINCT treatment
FROM dbo.Bone_Tumor1
ORDER BY treatment;
GO

SELECT DISTINCT [primary_site ]
FROM dbo.Bone_Tumor1
ORDER BY [primary_site ];
GO

SELECT DISTINCT mskcc_type
FROM dbo.Bone_Tumor1
ORDER BY mskcc_type;
GO

SELECT DISTINCT [outcome_status ]
FROM dbo.Bone_Tumor1
ORDER BY [outcome_status ];
GO

/*==============================================================================
STAGE 03 - EXPLORATORY DATA ANALYSIS
==============================================================================*/

SELECT sex, COUNT(*) AS cnt
FROM dbo.Bone_Tumor1
GROUP BY sex
ORDER BY cnt DESC;
GO

SELECT grade, COUNT(*) AS cnt
FROM dbo.Bone_Tumor1
GROUP BY grade
ORDER BY cnt DESC;
GO

SELECT [outcome_status ], COUNT(*) AS cnt
FROM dbo.Bone_Tumor1
GROUP BY [outcome_status ]
ORDER BY cnt DESC;
GO

SELECT age_group, COUNT(*) AS cnt
FROM dbo.Bone_Tumor1
GROUP BY age_group
ORDER BY cnt DESC;
GO

SELECT
    AVG(CAST(age_int AS FLOAT)) AS mean_age,
    MIN(age_int) AS min_age,
    MAX(age_int) AS max_age
FROM dbo.Bone_Tumor1
WHERE age_int IS NOT NULL;
GO

/*==============================================================================
STAGE 04 - FEATURE ENGINEERING
==============================================================================*/

CREATE OR ALTER VIEW dbo.vw_feature_matrix
AS
SELECT
    patient_id,
    age_int,

    /*------------------------------
      SEX FEATURES
    ------------------------------*/
    CASE WHEN sex = 'Male' THEN 1 ELSE 0 END AS sex_male,
    CASE WHEN sex = 'Female' THEN 1 ELSE 0 END AS sex_female,

    /*------------------------------
      GRADE FEATURES
    ------------------------------*/
    CASE WHEN grade = 'Low' THEN 1 ELSE 0 END AS grade_low,
    CASE WHEN grade = 'Intermediate' THEN 1 ELSE 0 END AS grade_intermediate,
    CASE WHEN grade = 'High' THEN 1 ELSE 0 END AS grade_high,

    /*------------------------------
      AGE GROUP FEATURES
    ------------------------------*/
    CASE WHEN age_group = 'Pediatric' THEN 1 ELSE 0 END AS age_pediatric,
    CASE WHEN age_group = 'YoungAdult' THEN 1 ELSE 0 END AS age_youngadult,
    CASE WHEN age_group = 'Adult' THEN 1 ELSE 0 END AS age_adult,
    CASE WHEN age_group = 'Elderly' THEN 1 ELSE 0 END AS age_elderly,

    /*------------------------------
      HISTOLOGY FEATURES
      Replace values if DISTINCT differs
    ------------------------------*/
    CASE WHEN histology = 'Osteosarcoma' THEN 1 ELSE 0 END AS histology_osteosarcoma,
    CASE WHEN histology = 'Chondrosarcoma' THEN 1 ELSE 0 END AS histology_chondrosarcoma,
    CASE WHEN histology = 'Ewing Sarcoma' THEN 1 ELSE 0 END AS histology_ewing_sarcoma,

    /*------------------------------
      TREATMENT FEATURES
      Replace values if DISTINCT differs
    ------------------------------*/
    CASE WHEN treatment = 'Surgery' THEN 1 ELSE 0 END AS treatment_surgery,
    CASE WHEN treatment = 'Chemotherapy' THEN 1 ELSE 0 END AS treatment_chemotherapy,
    CASE WHEN treatment = 'Radiotherapy' THEN 1 ELSE 0 END AS treatment_radiotherapy,

    /*------------------------------
      PRIMARY SITE FEATURES
      Replace values if DISTINCT differs
    ------------------------------*/
    CASE WHEN [primary_site ] = 'Femur' THEN 1 ELSE 0 END AS primary_site_femur,
    CASE WHEN [primary_site ] = 'Tibia' THEN 1 ELSE 0 END AS primary_site_tibia,
    CASE WHEN [primary_site ] = 'Pelvis' THEN 1 ELSE 0 END AS primary_site_pelvis,

    /*------------------------------
      MSKCC TYPE FEATURES
      Replace values if DISTINCT differs
    ------------------------------*/
    CASE WHEN mskcc_type = 'Type I' THEN 1 ELSE 0 END AS mskcc_type_1,
    CASE WHEN mskcc_type = 'Type II' THEN 1 ELSE 0 END AS mskcc_type_2,
    CASE WHEN mskcc_type = 'Type III' THEN 1 ELSE 0 END AS mskcc_type_3

FROM dbo.Bone_Tumor1;
GO

/*==============================================================================
STAGE 05 - DASHBOARD VIEW
==============================================================================*/

CREATE OR ALTER VIEW dbo.vw_dashboard_dataset
AS
SELECT
    patient_id,
    age,
    age_int,
    age_group,
    sex,
    grade,
    histology,
    treatment,
    [primary_site ],
    mskcc_type,
    [outcome_status ]
FROM dbo.Bone_Tumor1;
GO

/*==============================================================================
STAGE 06 - MACHINE LEARNING DATASET
==============================================================================*/

CREATE OR ALTER VIEW dbo.vw_ml_dataset
AS
SELECT
    f.*,
    b.[outcome_status ],

    CASE
        WHEN b.[outcome_status ] = 'NED' THEN 0
        WHEN b.[outcome_status ] = 'AWD' THEN 1
        WHEN b.[outcome_status ] = 'D' THEN 2
        ELSE NULL
    END AS outcome_label

FROM dbo.vw_feature_matrix f
INNER JOIN dbo.Bone_Tumor1 b
    ON f.patient_id = b.patient_id

WHERE b.[outcome_status ] IS NOT NULL
  AND b.[outcome_status ] NOT IN ('Unknown', 'N/A', '');
GO

/*==============================================================================
STAGE 07 - VALIDATION

==============================================================================*/

SELECT TOP (20) *
FROM dbo.vw_feature_matrix;
GO

SELECT TOP (20) *
FROM dbo.vw_ml_dataset;
GO

SELECT
    outcome_label,
    [outcome_status ],
    COUNT(*) AS cnt
FROM dbo.vw_ml_dataset
GROUP BY outcome_label, [outcome_status ]
ORDER BY outcome_label;
GO

SELECT
    patient_id,
    COUNT(*) AS cnt
FROM dbo.vw_feature_matrix
GROUP BY patient_id
HAVING COUNT(*) > 1;
GO

SELECT
    patient_id,
    COUNT(*) AS cnt
FROM dbo.vw_ml_dataset
GROUP BY patient_id
HAVING COUNT(*) > 1;
GO

SELECT
    SUM(CASE WHEN age_int IS NULL THEN 1 ELSE 0 END) AS age_int_null,
    SUM(CASE WHEN outcome_label IS NULL THEN 1 ELSE 0 END) AS outcome_label_null
FROM dbo.vw_ml_dataset;
GO

SELECT
    COUNT(*) AS total_rows,

    SUM(sex_male) AS sex_male_total,
    SUM(sex_female) AS sex_female_total,

    SUM(grade_low) AS grade_low_total,
    SUM(grade_intermediate) AS grade_intermediate_total,
    SUM(grade_high) AS grade_high_total,

    SUM(age_pediatric) AS age_pediatric_total,
    SUM(age_youngadult) AS age_youngadult_total,
    SUM(age_adult) AS age_adult_total,
    SUM(age_elderly) AS age_elderly_total,

    SUM(histology_osteosarcoma) AS histology_osteosarcoma_total,
    SUM(histology_chondrosarcoma) AS histology_chondrosarcoma_total,
    SUM(histology_ewing_sarcoma) AS histology_ewing_sarcoma_total,

    SUM(treatment_surgery) AS treatment_surgery_total,
    SUM(treatment_chemotherapy) AS treatment_chemotherapy_total,
    SUM(treatment_radiotherapy) AS treatment_radiotherapy_total

FROM dbo.vw_feature_matrix;
GO
