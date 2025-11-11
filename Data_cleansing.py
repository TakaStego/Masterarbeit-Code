###########################################################################################
###LOADING PACKAGES AND DATA
###########################################################################################
import pandas as pd
import numpy as np

#Loading the raw csv
df_raw = pd.read_csv(
    r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\20251008\20251008.csv",
    sep=",", on_bad_lines="warn"
)
print(f"Length prior to data cleansing: {len(df_raw)} observations")

###########################################################################################
###DATA PREPARATION AND CLEANING
###########################################################################################

#Renaming the columns for better readability
df_rename = df_raw.rename(columns={"IU_1": "used_social_media", "IU_1_8_TEXT": "used_social_media_text", "IU_2": "use_time_macro", "IU_3": "use_time_micro", "IU_4_1": "activeness", "IU_5_1": "trust_review", "IU_5_2": "trust_data", "IU_5_3": "trust_remove", 
                                   "IU_5_4": "trust_comply", "GK_1_1": "knowledge_gdpr", "GK_1_2": "knowledge_ai", "GK_1_3": "knowledge_dsa", "GK_1_4": "knowledge_dma", "GK_1_5": "knowledge_daa", "GK_1_6": "knowledge_tf", "IC_1": "seen_content", "IC_2": "reported_content",
                                   "IC_2.1": "reasons", "IC_2.1_9_TEXT": "reasons_text", "IC_3": "attention_check", "DK_1": "interest", "DK_2": "used_rights", "DK_3": "reasons_rights", "DK_3_8_TEXT": "reasons_rights_text", "DC_1": "shaping_space", "DC_2": "citizenship",
                                   "DC_3_1": "improving_protection", "DC_3_2": "worrying", "DG_1": "citizenship2", "DG_2": "gender", "DG_2_4_TEXT": "gender_text", "DG_3": "education", "DG_3_9_TEXT": "education_text", "DG_4": "age" 
                                   })
print(df_rename.columns.tolist())

#Transforming numeric values to actual numbers, since they were exported as strings by Qualtrics
numeric_cols = ["Duration (in seconds)", "Progress", "use_time_micro", "use_time_macro", "activeness", "interest", 
                "trust_review", "trust_data", "trust_remove", 
                "trust_comply", "knowledge_gdpr", "knowledge_ai",
                "knowledge_dsa", "knowledge_dma", "knowledge_daa",
                "knowledge_tf", "reported_content", "attention_check", "interest", "shaping_space", "citizenship", 
                "improving_protection", "worrying", "gender", "education", "age"]

for col in numeric_cols:
    if col in df_rename.columns:
        df_rename[col] = pd.to_numeric(df_rename[col], errors="coerce")

#Creating dummy variables
df_rename["reported_content_dummy"] = df_rename["reported_content"].isin([1, 3]).astype(int)
df_rename["used_rights_dummy"] = df_rename["used_rights"].astype(str).isin(["-3", "8"]).astype(int)
df_rename["seen_content_dummy"] = df_rename["seen_content"].astype(str).isin(["-3", "1"]).astype(int)
df_rename["knowledge_dsa_dummy"] = 0
df_rename["knowledge_dsa_dummy"] = (df_rename["knowledge_dsa"] > 2).astype(int)
df_rename["used_rights_dummy"] = 1 - df_rename["used_rights_dummy"]
df_rename["seen_content_dummy"] = 1 - df_rename["seen_content_dummy"]
df_rename["gender_male"] = (df_rename["gender"] == 1).astype(int)
df_rename["gender_female"] = (df_rename["gender"] == 2).astype(int)


#Removing responses with low quality based on progress, duration and attention check
df_progress = df_rename[df_rename["Progress"] >= 95].copy()
print(f"Length after progress-cleansing: {len(df_progress)}")
cutoff_lower = np.quantile(df_progress["Duration (in seconds)"], 0.05)
df_duration = df_progress[df_progress["Duration (in seconds)"] >= cutoff_lower].copy()
print(f"Length after time-cleansing: {len(df_duration)}")
df = df_duration[df_duration["attention_check"]==5].copy()
print(f"Length after attention-check: {len(df)}")

#Constructing average-variables
df["knowledge_average"] = ((df["knowledge_dsa"] + df["knowledge_tf"] + df["knowledge_dma"] + df["knowledge_gdpr"] + df["knowledge_ai"])/5)
df["trust_average"] = ((df["trust_review"] + df["trust_data"] + df["trust_remove"] + df["trust_comply"])/4)


###########################################################################################
###LABELLING VARIABLES
###########################################################################################

###USE TIME
##Macro - IU_2
df_use_time_macro= df["use_time_macro"]

use_time_macro_map = {
    1: "Everyday or almost every day",
    2: "Two or three times a week",
    3: "About once a week",
    4: "Two or three times a month",
    5: "Less often",
    -1: "Don´t know",
    -3: "Prefer not to answer" 
}

df["use_time_macro_label"] = df_use_time_macro.map(use_time_macro_map)


##Micro - IU_3
df_use_time_micro= df["use_time_micro"]

use_time_micro_map = {
    1: "Up to 1 hour",
    2: "1 to 2 hours",
    3: "2 to 3 hours",
    4: "3 to 4 hours",
    5: "4 to 5 hours",
    6: "More than 5 hours",
    -1: "Don´t know",
    -3: "Prefer not \n to answer" 
}

df["use_time_micro_label"] = df_use_time_micro.map(use_time_micro_map)


###ACTIVENESS
activeness_map = {
    1: "Completley passive",
    2: "Mostly passive",
    3: "Somewhat passive",
    4: "Both passive and active",
    5: "Somewhat active",
    6: "Mostly active",
    7: "Completly active"
}

df["activeness_label"] = df["activeness"].map(activeness_map)

###TRUST
##Review
trust_map = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Undecided",
    4: "Somewhat agree",
    5: "Strongly agree"
}
df["trust_review_label"] = df["trust_review"].map(trust_map)

##data
df["trust_data_label"] = df["trust_data"].map(trust_map)


##remove
df["trust_remove_label"] = df["trust_remove"].map(trust_map)

##comply
df["trust_comply_label"] = df["trust_comply"].map(trust_map)

###KNOWLEDGE
##GDPR
knowledge_map = {
    1: "No knowledge",
    2: "Little knowledge",
    3: "Basic knowledge",
    4: "Good knowledge",
    5: "Very good knowledge"
}

df["knowledge_gdpr_label"] = df["knowledge_gdpr"].map(knowledge_map)

##AI-Act
df["knowledge_ai_label"] = df["knowledge_ai"].map(knowledge_map)

##DSA
df["knowledge_dsa_label"] = df["knowledge_dsa"].map(knowledge_map)

##DMA
df["knowledge_dma_label"] = df["knowledge_dma"].map(knowledge_map)

##DAA
df["knowledge_daa_label"] = df["knowledge_daa"].map(knowledge_map)

##Trusted flaggers
df["knowledge_tf_label"] = df["knowledge_tf"].map(knowledge_map)

#REPORTED_CONTENT
reported_content_map = {
    1: "Yes",
    2: "No",
    3: "Not sure if illegal",
    -3: "Prefer not to answer" 
}
df["reported_content_label"] = df["reported_content"].map(reported_content_map)


###Interest in digital policy
interest_map = {
    1: "Completely uninterested",
    2: "Mostly uninterested",
    3: "Indifferent",
    4: "Somewhat interested",
    5: "Very interested"
}
df["interest_label"] = df["interest"].map(interest_map)

###Seeing the granted rights as possibility to shape the digital space
shaping_space_map = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neutral",
    4: "Somewhat agree",
    5: "Strongly agree"
}
#df_shaping_space = df["shaping_space"]
df["shaping_space_label"] = df["shaping_space"].map(shaping_space_map)

###Seeing onself as digital citizen
citizenship_map = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neutral",
    4: "Somewhat agree",
    5: "Strongly agree"
}
df["citizenship_label"] = df["citizenship"].map(citizenship_map)


###IMPROVING PROTECTION
improving_protection_map = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neutral",
    4: "Somewhat agree",
    5: "Strongly agree"
}
df["improving_protection_label"] = df["improving_protection"].map(improving_protection_map)

###IMPROVING PROTECTION
worrying_map = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neutral",
    4: "Somewhat agree",
    5: "Strongly agree"
}
df["worrying_label"] = df["worrying"].map(worrying_map)


###EDUCATION
education_map = {
    1: "Left school withour a degree",
    2: "Still a pupil",
    3: "Middle School",
    4: "High School",
    5: "Apprenticeship or vocational training",
    6: "Bachelor´s degree",
    7: "Master´s degree",
    8: "PhD",
    9: "Other",
    -3: "Prefer not to answer"
}
df["education_label"] = df["education"].map(education_map)


###ISCED-Labelling
isced_value_map = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    -3: 3
    }

isced_label_map = {
    0: "Early childhood education",
    1: "Primary education",
    2: "Lower secondary education",
    3: "Upper secondary education",
    4: "Post-secondary non-tertiary education",
    5: "Short-cycle tertiary education",
    6: "Bachelor´s or equivalent level",
    7: "Master´s or equivalent level",
    8: "Doctoral or equivalent level",
    9: "Prefer not to answer"
}

df["isced"] = df["education"].map(isced_value_map)
df["isced_label"] = df["isced"].map(isced_label_map)


###AGE
###Age groups
bins = [0, 24, 34, 44, 54, 64, 101]
group = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
df["age_group_label"] = pd.cut(df["age"], bins=bins, labels=group, right=False).dropna()

###GENDER
gender_map = {
    1: "Male",
    2: "Female",
    3: "Non-Binary",
    4: "Self-Description preferred",
    -3: "Prefer not to answer"
}

df["gender_label"] = df["gender"].map(gender_map)

df.to_csv(
    r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\cleaned\analysis.csv",
    sep=";")