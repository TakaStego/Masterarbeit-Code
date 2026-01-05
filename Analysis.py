###########################################################################################
#                   MUNICH SCHOOL OF POLITICS AND PUBLIC POLICY 
#                        Technical University of Munich
#                           Chair of Digital Governance

#                   Supplementary Code for the Master Thesis
#                                Noah Jakob Steidel
#                                Mat-No.: 03786165
#                            noah.steidel@protonmail.com
###########################################################################################

###########################################################################################
###LOADING PACKAGES
###########################################################################################
import pandas as pd
import xlsxwriter
import numpy as np
import scipy as scipy
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import statsmodels.api as sm 
from pandas import DataFrame as DF
from scipy import stats as stats
from scipy.stats.distributions import chi2
from pathlib import Path
from stargazer.stargazer import Stargazer
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor

###########################################################################################
###IMPORTING DATA (adapt to your file system)
###########################################################################################
#Adapt absolute path to your OS
path_abs = path_data = Path(r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit")


path_viz = Path(rf"{path_abs}\Data\output\visualizations")
path_reg = Path(rf"{path_abs}\Data\output\regression")
path_desc = Path(rf"{path_abs}\Data\output\descriptive")
path_data = Path(rf"{path_abs}\Data")

def latex(path, filename, dataframe):
    with open(path / filename, "w", encoding="utf-8") as f:
        f.write(dataframe.to_latex(float_format="%.0f"))

def latex_float(path, filename, dataframe):
    with open(path / filename, "w", encoding="utf-8") as f:
        f.write(dataframe.to_latex(float_format="%.2f"))


df = pd.read_csv(path_data / "cleaned" / "analysis.csv",
    sep=";")
row_count = len(df)
###########################################################################################
###Ordering Likert-scale and knowledge-variables & creating value counts
###########################################################################################

#Defining Likert-Order
likert_order = [
    "Strongly disagree",
    "Somewhat disagree",
    "Neutral",
    "Somewhat agree",
    "Strongly agree"
]

# List of all columns based on Likert-Scales
likert_cols = [
    "citizenship_label",
    "shaping_space_label",
    "improving_protection_label",
    "worrying_label",
    "trust_data_label",
    "trust_remove_label",
    "trust_review_label",
    "trust_comply_label"
]

knowledge_order = [
    "No knowledge",
    "Little knowledge",
    "Basic knowledge",
    "Good knowledge",
    "Very good knowledge"
]

interest_order = [
    "Completely uninterested",
    "Mostly uninterested",
    "Indifferent",
    "Somewhat interested",
    "Very interested"
]

# Setting categories
for col in likert_cols:
    df[col] = pd.Categorical(df[col], categories=likert_order, ordered=True)

for col in df.columns:
    if col.endswith("_label"):
        base = col.replace("_label", "")
        var_name = f"counts_{base}"
        globals()[var_name] = df[col].value_counts()
    
###INTERNET USAGE
##Used Networks
###NETWORKS - IU_1
network_map = {
    1: "Facebook",
    2: "Instagram",
    3: "Pinterest",
    4: "Snapchat",
    5: "TikTok",
    6: "X",
    7: "YouTube",
   -1: "I do not use social media at all",
   -2: "I do not use any of these platforms at least once a week",
   -3: "Prefer not to answer"
}

#Transforming lists to integer-values 
def safe_split(x):
    if pd.isna(x):
        return []
    parts = str(x).split(",")
    return [int(p.strip()) for p in parts if p.strip() != ""]

df["used_social_media"] = df["used_social_media"].apply(safe_split)

#Exploding and mapping labels
df_exploded_networks = df.explode("used_social_media")
df_exploded_networks["networks_label"] = df_exploded_networks["used_social_media"].map(network_map)
df_exploded_networks["networks_numeric"] = df_exploded_networks["used_social_media"].map(network_map)

#creating and exporting value counts
counts_networks = df_exploded_networks["networks_label"].value_counts()
counts_networks = (counts_networks/row_count)
latex(path_desc, "used networks.txt", counts_networks)

###SEEN CONTENT
seen_content_map = {
    1: "Not seen illegal content",
    2: "Seen this type of content directed at others",
    3: "Personally been targeted by such content",
    -3: "Prefer not to answer"
}

df["seen_content"] = df["seen_content"].apply(safe_split)

# Exploding and mapping labels
df_exploded_seen_content = df.explode("seen_content")
df_exploded_seen_content["exploded_seen_content_label"] = df_exploded_seen_content["seen_content"].map(seen_content_map)

###REASONS FOR NOT REPORTING
reasons_map = {
    1: "Not sure whether content was illegal",
    2: "No knowledge reporting was possible",
    3: "Reporting process too complicated",
    4: "Reporting does not make a difference",
    5: "No trust towards companies",
    6: "Not bothered by content",
    7: "Illegal content is seen too often to care",
    8: "Afraid of negative consequences",
    9: "Other",
   -1: "Don´t know",
   -3: "Prefer not to answer"
}

df["reasons"] = df["reasons"].apply(safe_split)

#Exploding and mapping labels
df_exploded_reasons = df.explode("reasons")
df_exploded_reasons["labelled_reasons"] = df_exploded_reasons["reasons"].map(reasons_map)

###USED RIGHTS
used_rights_map = {
    1: "Report illegal content directed at me to platform",
    2: "Report illegal content directed at others to platform",
    3: "Access information about recommender systems",
    4: "Access information about advertising systems",
    5: "Contest a moderation decision",
    6: "Use a non-personalized recommender system",
    7: "Report illegal content to a trusted flagger",
    8: "No rights usage",
   -3: "Prefer not to answer"
}

df["used_rights"] = df["used_rights"].apply(safe_split)

#Exploding and mapping labels
df_exploded_rights = df.explode("used_rights")
df_exploded_rights["used_rights_label"] = df_exploded_rights["used_rights"].map(used_rights_map)


###REASONS FOR NOT USING RIGHTS
###REASONS FOR NOT REPORTING
reasons_rights_map = {
    1: "Not aware of those rights",
    2: "Not interested in using rights",
    3: "Information or tool too hard to find",
    4: "Did not feel like action would have an impact",
    5: "Afraid of potential negative consequences",
    6: "Did not encounter a situation where rights use was necessary",
    7: "Plan to use those rights in the future",
    8: "Other",
   -3: "Prefer not to answer"
}

df["reasons_rights"] = df["reasons_rights"].apply(safe_split)

#Exploding and mapping labels
df_exploded_reasons_rights = df.explode("reasons_rights")
df_exploded_reasons_rights["labelled_reasons_rights"] = df_exploded_reasons_rights["reasons_rights"].map(reasons_rights_map)

###########################################################################################
###DESCRIPTIVE ANALYSIS
###########################################################################################
###Socio demographics
###creating a table including all variables with interpretable descriptive values
table_vars = [
    "use_time_micro",
    "use_time_macro",
    "activeness",
    "trust_review",
    "trust_data",
    "trust_comply",
    "trust_remove",
    "trust_average",
    "knowledge_gdpr",
    "knowledge_dsa",
    "knowledge_tf",
    "knowledge_dma",
    "knowledge_ai",
    "knowledge_average",
    "interest",
    "shaping_space",
    "citizenship",
    "improving_protection",
    "worrying",
    "isced",
    "age"
]

df_non_negative = df[~(df[table_vars] < 0).any(axis=1)].copy()

descriptive_table = df_non_negative[table_vars].agg(["median", "std", "min", "max"]).transpose()
latex_float(path_desc, "descriptive_table.txt", descriptive_table)

###Language
df_language = df["UserLanguage"]
count_language = df_language.value_counts()
latex(path_desc, "Language distribution.txt", count_language)

###Graphics
#Setting color palette
cmap = plt.get_cmap("viridis")

#Gender
colors = cmap(np.linspace(0, 1, len(counts_gender)))
fig, ax = plt.subplots()
ax.bar(counts_gender.index, counts_gender.values, color=colors)
fig.savefig(path_viz / "gender_distribution.jpg")
plt.close(fig)

#Education
colors = cmap(np.linspace(0, 1, len(counts_education)))
fig, ax = plt.subplots(figsize = (12,6))
ax.pie(counts_education.values, colors=colors)
ax.legend(labels=counts_education.index, loc=3, bbox_to_anchor=(1.0, 0.5))
plt.savefig(path_viz / "education_counts.jpg", dpi = 600)
plt.close(fig)

#Age distribution
colors = cmap(np.linspace(0, 1, len(counts_age_group)))
fig, ax = plt.subplots()
ax.bar(counts_age_group.index, counts_age_group.values, color=colors)
fig.savefig(path_viz / "age_distribution.jpg")
plt.close(fig)

#Time taken to answer survey
fig, ax = plt.subplots()
sns.kdeplot(df["Duration (in seconds)"], fill = True)
fig.savefig(path_viz / "time_histogram.jpg")
plt.close(fig)

#Use time - Micro
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_use_time_micro.index, counts_use_time_micro.values, color = colors)
fig.savefig(path_viz / "Use time micro.jpg")
plt.close(fig)

#Use time - Macro
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_use_time_macro.index, counts_use_time_macro.values, color = colors)
fig.savefig(path_viz / "Use time macro.jpg")
plt.close(fig)

##Knowledge distribution
#DSA
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_dsa.index, counts_knowledge_dsa.values, color = colors)
fig.savefig(path_viz / "DSA knowledge.jpg")
plt.close(fig)

#DMA
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_dma.index, counts_knowledge_dma.values, color = colors)
fig.savefig(path_viz / "DMA knowledge.jpg")
plt.close(fig)

#Trusted flaggers
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_tf.index, counts_knowledge_tf.values, color = colors)
fig.savefig(path_viz / "TF knowledge.jpg")
plt.close(fig)

#GDPR
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_gdpr.index, counts_knowledge_gdpr.values, color = colors)
fig.savefig(path_viz / "GDPR knowledge.jpg")
plt.close(fig)

#AI Act
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_ai.index, counts_knowledge_ai.values, color = colors)
fig.savefig(path_viz / "AI Act knowledge.jpg")
plt.close(fig)

#DAA - fictive legislation
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_knowledge_daa.index, counts_knowledge_daa.values, color = colors)
fig.savefig(path_viz / "DAA knowledge.jpg")
plt.close(fig)

##Seen content
counts_seen_content = df_exploded_seen_content["exploded_seen_content_label"].value_counts()
labels_wrapped_content = ['\n'.join(textwrap.wrap(str(label), 20)) for label in counts_seen_content.index]
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(labels_wrapped_content, counts_seen_content.values, color = colors)
fig.savefig(path_viz / "Counts seen content.jpg")
plt.close(fig)

##Used networks
labels_wrapped_networks = ['\n'.join(textwrap.wrap(str(label), 15)) for label in counts_networks.index]
fig, ax = plt.subplots(figsize = (20, 10))
ax.bar(labels_wrapped_networks, counts_networks.values, color = colors)
fig.savefig(path_viz / "networks.jpg")
plt.close(fig)

#############################################################
###REGRESSIONS
#############################################################
#############################
#Creating data frames
#############################
#Creating a data frame with a dummy for reported content
#Respondents with "prefer not to answer" (numerical value = -3, but data is in list-format) are excluded, since no definite statement can be made about them.
mask = df["reported_content"] == -3
df_model_1 = df[~mask]

#For the logit with reported content as dummy variable, only those respondents with seen_content_dummy == 1 are being considered. Seeing illegal content is a necessary condition for reporting illegal content, irrespective the reporting mechanism.
#Model 1 is the model with reporting dummy as DV.
df_model_1 = df_model_1[df_model_1["seen_content_dummy"] == 1].copy()

#Creating a data frame with a dummy for used rights
#Respondents with "prefer not to answer" (numerical value = -3, but data is in list-format) are excluded, since no definite statement can be made about them 
mask2 = df["used_rights"].apply(lambda x: -3 in x)
df_model_2 = df[~mask2]

#Re-creating the centered variables, since cases were removed from the dataframe, thereby also impacting the mean values. 
#creating new interaction term
df_model_1["knowledge_dsa_centered"] = (df_model_1["knowledge_dsa"] - df_model_1["knowledge_dsa"].mean())
df_model_1["trust_average_centered"] = (df_model_1["trust_average"] - df_model_1["trust_average"].mean())
df_model_1["interaction_term"] = (df_model_1["knowledge_dsa_centered"]*df_model_1["trust_average_centered"])

#Centering the knowledge- and trust variable and creating an interaction term. Model 2 is the model with using rights as DV.
df_model_2["knowledge_dsa_centered"] = (df_model_2["knowledge_dsa"] - df_model_2["knowledge_dsa"].mean())
df_model_2["trust_average_centered"] = (df_model_2["trust_average"] - df_model_2["trust_average"].mean())
df_model_2["interaction_term"] = (df_model_2["knowledge_dsa_centered"]*df_model_2["trust_average_centered"])

#############################
#MAIN MODEL - EXTENDED MODEL - NO INTERACTION TERM
#############################
###Model without interaction term - creating predictor variables
X_report = df_model_1[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
print(f"so viele nans sind hier drin: {X_report.isna().sum()}")
X_report = X_report.fillna(X_report.median())
X_report = sm.add_constant(X_report, has_constant= "add")

#Model construction 1 - Reporting content as DV
regression = []
y=df_model_1["reported_content_dummy"]
regression_report = sm.Logit(y, X_report).fit()
regression.append(regression_report)

X_rights = df_model_2[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
X_rights = X_rights.fillna(X_rights.median())
X_rights = sm.add_constant(X_rights, has_constant= "add")

#Model construction 2 - Used rights as DV
y=df_model_2["used_rights_dummy"]
regression_rights = sm.Logit(y, X_rights).fit()
regression.append(regression_rights)

stargazer = Stargazer(regression)
with open(path_reg / "Main Model.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))

#Controlling the Variation Inflation Index to identify possible multicollinearity issues for Model 1
vif_report = pd.Series([variance_inflation_factor(X_report.values, i)
                        for i in range(X_report.shape[1])],
                        index = X_report.columns)
latex_float(path_reg, "multicollinearity check report.txt", vif_report)

#Controlling the Variation Inflation Index to identify possible multicollinearity issues for Model 2
vif_rights = pd.Series([variance_inflation_factor(X_rights.values, i)
                        for i in range(X_rights.shape[1])],
                        index = X_rights.columns)
latex_float(path_reg, "multicollinearity check rights.txt", vif_rights)

#Calculating Average Marginal Effects
ame_rights = regression_rights.get_margeff(at = "overall")
ame_report = regression_report.get_margeff(at = "overall")
print(f"AMEs for using rights:{ame_rights.summary()}")
print(f"AMEs for reporting content: {ame_report.summary()}")

##############################################
###APPENDIX - Supplementary MODELS
##############################################
#############################
#Main Model - excluding interest to check for possible ommission
#############################
#Model creation - incl. interaction term
X_report_omission = df_model_1[["age", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
X_report_omission = X_report_omission.fillna(X_report_omission.median())
X_report_omission = sm.add_constant(X_report_omission, has_constant= "add")

#Model creation 1- Reporting content as dependent variable
regression_omission = []
y=df_model_1["reported_content_dummy"]
omission_report = sm.Logit(y, X_report_omission).fit()
regression_omission.append(omission_report)

X_rights_omission = df_model_2[["age", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
X_rights_omission = X_rights_omission.fillna(X_rights_omission.median())
X_rights_omission = sm.add_constant(X_rights_omission, has_constant= "add")

#Model creation 2 - Used rights as dependent variable
y=df_model_2["used_rights_dummy"]
omission_rights = sm.Logit(y, X_rights_omission).fit()
regression_omission.append(omission_rights)

stargazer = Stargazer(regression_omission)
with open(path_reg / "Omission of interest.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))


#############################
#Main Model - seeing oneself as digital citizen instead of shaping space as IV
#############################
###Model without interaction term - creating predictor variables, filling nan with 
X_report_citizenship = df_model_1[["age", "interest", "citizenship", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
X_report_citizenship = X_report_citizenship.fillna(X_report_citizenship.median())
X_report_citizenship = sm.add_constant(X_report_citizenship, has_constant= "add")

#Model construction 1 - Reporting content as DV
regression_citizenship = []
y=df_model_1["reported_content_dummy"]
regression_report_citizenship = sm.Logit(y, X_report_citizenship).fit()
regression_citizenship.append(regression_report_citizenship)

X_rights_citizenship = df_model_2[["age", "interest", "citizenship", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]]
X_rights_citizenship = X_rights_citizenship.fillna(X_rights_citizenship.median())
X_rights_citizenship = sm.add_constant(X_rights_citizenship, has_constant= "add")

#Model construction 2 - Used rights as DV
y=df_model_2["used_rights_dummy"]
regression_rights_citizenship = sm.Logit(y, X_rights_citizenship).fit()
regression_citizenship.append(regression_rights_citizenship)

stargazer = Stargazer(regression_citizenship)
with open(path_reg / "Main Model - Citizenship.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))

#############################
#Main Model - strictly reporting using the NaAM
#############################
###Model without interaction term - creating predictor variables, filling nan with 
X_report_strict = df_model_1[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "worrying"]].replace([np.inf, -np.inf], np.nan)
X_report_strict = X_report_strict.fillna(X_report_strict.median())
X_report_strict = sm.add_constant(X_report_strict, has_constant= "add")

#Model construction 1 - Reporting content as DV
regression_strict = []
y=df_model_1["reported_content_dummy_naam"]
regression_report_strict = sm.Logit(y, X_report_strict).fit()
regression_strict.append(regression_report_strict)

stargazer = Stargazer(regression_strict)
with open(path_reg / "Main Model - Strict NaAM.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))


##############################################
###Extended Model - with interaction term
##############################################
#Model creation - incl. interaction term
X_report_interaction = df_model_1[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "interaction_term", "worrying"]].replace([np.inf, -np.inf], np.nan)
X_report_interaction = X_report_interaction.fillna(X_report_interaction.median())
X_report_interaction = sm.add_constant(X_report_interaction, has_constant= "add")

#Model creation 1- Reporting content as dependent variable
regression_interaction = []
y=df_model_1["reported_content_dummy"]
model2_report = sm.Logit(y, X_report_interaction).fit()
regression_interaction.append(model2_report)

X_rights_interaction = df_model_2[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "interaction_term", "worrying"]].replace([np.inf, -np.inf], np.nan)
X_rights_interaction = X_rights_interaction.fillna(X_rights_interaction.median())
X_rights_interaction = sm.add_constant(X_rights_interaction, has_constant= "add")

#Model creation 2 - Used rights as dependent variable
y=df_model_2["used_rights_dummy"]
model2_rights = sm.Logit(y, X_rights_interaction).fit()
regression_interaction.append(model2_rights)

stargazer = Stargazer(regression_interaction)
with open(path_reg / "Logistic regression - Extended Model.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))


##############################################
###Baseline-Model - less IVs
##############################################
X_report_reduced = df_model_1[["age", "interest", "use_time_micro", "knowledge_dsa_centered", "worrying"]].replace([np.inf, -np.inf], np.nan)
X_report_reduced = X_report_reduced.fillna(X_report_reduced.median())
X_report_reduced = sm.add_constant(X_report_reduced, has_constant= "add")

#Reporting content as dependent variable
regression_reduced = []
y=df_model_1["reported_content_dummy"]
regression_reduced_report = sm.Logit(y, X_report_reduced).fit()
regression_reduced.append(regression_reduced_report)

X_rights_reduced = df_model_2[["age", "interest", "use_time_micro", "knowledge_dsa_centered", "worrying"]].replace([np.inf, -np.inf], np.nan)
X_rights_reduced = X_rights_reduced.fillna(X_rights_reduced.median())
X_rights_reduced = sm.add_constant(X_rights_reduced, has_constant= "add")

#Model creation 2 - Used rights as dependent variable
y=df_model_2["used_rights_dummy"]
model1_rights = sm.Logit(y, X_rights_reduced).fit()
regression_reduced.append(model1_rights)

stargazer = Stargazer(regression_reduced)
with open(path_reg / "Logistic regression - Reduced model.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))

#############################
#Main Model - no interaction term - believing that DSA improves protection
#############################
###Model without interaction term - creating predictor variables, filling nan with 
X_report_improving = df_model_1[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "improving_protection"]].replace([np.inf, -np.inf], np.nan)
X_report_improving = X_report_improving.fillna(X_report_improving.median())
X_report_improving = sm.add_constant(X_report_improving, has_constant= "add")

#Model construction 1 - Reporting content as dependent variable
regression_improving = []
y=df_model_1["reported_content_dummy"]
regression_improving_report = sm.Logit(y, X_report_improving).fit()
regression_improving.append(regression_improving_report)

X_rights_improving = df_model_2[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "improving_protection"]].replace([np.inf, -np.inf], np.nan)
X_rights_improving = X_rights_improving.fillna(X_rights_improving.median())
X_rights_improving = sm.add_constant(X_rights_improving, has_constant= "add")

#Model construction 2 - Used rights as dependent variable
y=df_model_2["used_rights_dummy"]
model1_rights = sm.Logit(y, X_rights_improving).fit()
regression_improving.append(model1_rights)

stargazer = Stargazer(regression_improving)
with open(path_reg / "Logistic regression - Improving protection.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))


###########################################################################################
###CREATING OUTPUT FOR LATEX
###########################################################################################
###Counts
#Gender counts
latex(path_desc, "Count gender distribution.txt", counts_gender)

#Education
latex(path_desc, "Counts Education (ISCED-Scale).txt", counts_isced)

#Used rights
counts_used_rights = df_exploded_rights["used_rights_label"].value_counts()
latex(path_desc, "Count used rights.txt", counts_used_rights)

#Seen illegal content
latex(path_desc, "Count seen illegal content.txt", counts_seen_content)
share_seen_content = (counts_seen_content/row_count)
latex(path_desc, "share seen content.txt", share_seen_content)

#Reported illegal content
counts_reported_content = df["reported_content_label"].value_counts()
latex(path_desc, "Counts reported illegal content.txt", counts_reported_content)

#Counts of reported knowledge
table_knowledge = pd.DataFrame({
    "GDPR": counts_knowledge_gdpr,
    "AI Act": counts_knowledge_ai,
    "DSA": counts_knowledge_dsa,
    "DMA": counts_knowledge_dma,
    "DAA": counts_knowledge_daa,
    "Trusted Flaggers": counts_knowledge_tf
}).fillna(0).reindex(knowledge_order)
latex(path_desc, "Count Knowledge 2.txt", table_knowledge)

#Count of digital citizenship & shaping space
citizenship_perception = pd.DataFrame({
    "Digital Citizenship": counts_citizenship,
    "Rights are a possibility to shape the digital space": counts_shaping_space
}).fillna(0).reindex(likert_order)
latex(path_desc, "Digital citizenship & shaping space.txt", citizenship_perception)

#Count of shaping space
latex(path_desc, "Shaping space.txt", counts_shaping_space.reindex(likert_order))

#Count of the reasons not to report illegal content
reasons_counts = df_exploded_reasons["labelled_reasons"].value_counts()
latex(path_desc, "Reasons for not reporting.txt", reasons_counts)
print(f"Number of people who gave reasons for not reporting when they saw illegal content: {df["reasons"].notnull().sum()}")

#Count of the reasons not to make use of rights
reasons_rights_counts = df_exploded_reasons_rights["labelled_reasons_rights"].value_counts()
latex(path_desc, "Reasons for not using rights.txt", reasons_rights_counts)
print(f"Number of people who gave reasons for not using rights under the DSA: {df["reasons_rights"].notna().sum()}")

#Perception of the DSA
table_perception = pd.DataFrame({
    "Improving protection": counts_improving_protection,
    "Worrying": counts_worrying
}).reindex(likert_order)
latex(path_desc, "Perception of the DSA.txt", table_perception)

###Crosstables
#Crosstable between daily use time and knowledge
df_use_time_micro_na = df[df["use_time_micro"].between(1, 6)].copy()
crosstable_knowledge = pd.crosstab(df_use_time_micro_na["use_time_micro"], df_use_time_micro_na["knowledge_gdpr"])
latex(path_desc, "Crosstable knowledge & daily use time.txt", crosstable_knowledge)

#Crosstable between people who used their rights and people who made reports
df_filtered = df_exploded_rights[df_exploded_rights["used_rights"].isin([1, 2])]
crosstable_reports = pd.crosstab(df_filtered["used_rights_label"], df_filtered["reported_content_label"], margins=True)
latex(path_desc, "reporting crosstable.txt", crosstable_reports)

#Crosstable between people who have seen illegal content and who reported content
df_exploded_seen_content_na = df_exploded_seen_content[df_exploded_seen_content["reported_content_label"] != "-3"]
crosstable_reporting_seen = pd.crosstab(df_exploded_seen_content_na["exploded_seen_content_label"], df_exploded_seen_content_na["reported_content_label"])
latex(path_desc, "Reporting seen content.txt", crosstable_reporting_seen)

crosstable_reporting_dummy_seen = pd.crosstab(df_exploded_seen_content["exploded_seen_content_label"], df_exploded_seen_content["reported_content_dummy"])
latex(path_desc, "Reporting dummy seen.txt", crosstable_reporting_dummy_seen)

#Crosstable between knowledge about the DSA and reporting
crosstable_reporting_knowledge = pd.crosstab(df["knowledge_dsa_label"], df["reported_content_label"])
latex(path_desc, "Reporting knowledge crosstable.txt", crosstable_reporting_knowledge)

#Crosstable between dummy variable for having used a right under the DSA and seen illegal content
crosstab_rights_seen = pd.crosstab(df_model_2["used_rights_dummy"], df_model_2["seen_content_dummy"])
print(crosstab_rights_seen)
latex(path_desc, "crosstable used rights & seen content.txt", crosstab_rights_seen)

#Crosstable between daily use time and knowledge of the DSA
crosstab_usetime_knowledge = pd.crosstab(df_use_time_micro_na["use_time_micro"], df_use_time_micro_na["knowledge_dsa_label"])
crosstab_usetime_knowledge = crosstab_usetime_knowledge[knowledge_order]
latex(path_desc, "crosstable use time & knowledge.txt", crosstab_usetime_knowledge)

#Crosstable between interest and knowledge of the DSA
crosstab_interest_knowledge = pd.crosstab(df["knowledge_dsa_label"], df["interest_label"]).reindex(knowledge_order).reindex(columns = interest_order)
latex(path_desc, "crosstable interest & knowledge.txt", crosstab_interest_knowledge)

print(f" Correlation between education and interest: {df[["isced", "interest"]].corr()}")
print(f"Correlation between knowledge about DSA and interest: {df_model_1[["knowledge_dsa", "interest"]].corr()}")
print(f"Correlation between trust and interest: {df_model_2[["trust_average", "interest"]].corr()}")
df_seen_reported_content = df[df["seen_content_dummy"] == 1]
print(f"correlation between interest and using rights: {df[["used_rights_dummy", "interest"]].corr()}")
print(f"Share of people who reported content upon seeing illegal content: {(df_seen_reported_content["reported_content_dummy"].sum())/len(df_seen_reported_content)}")
print(f"Median duration: {df["Duration (in seconds)"].median()}")
print(len(df_model_2))
print(f"Number of people who used more than 1 right: {df_model_2["used_rights"].astype(str).str.contains(',').sum()}")

'''
###TESTSITE
print((df["knowledge_dsa"].sum())/len(df["knowledge_dsa"]))
print((df["knowledge_dma"].sum())/len(df["knowledge_dma"]))
print((df["knowledge_daa"].sum())/len(df["knowledge_daa"]))
print((df["knowledge_ai"].sum())/len(df["knowledge_ai"]))
print((df["knowledge_gdpr"].sum())/len(df["knowledge_gdpr"]))
print((df["knowledge_tf"].sum())/len(df["knowledge_tf"]))

df_social_desirability = df[df["knowledge_daa"]==1].copy()
print((df_social_desirability["knowledge_dsa"].sum())/len(df_social_desirability["knowledge_dsa"]))
print((df_social_desirability["knowledge_dma"].sum())/len(df_social_desirability["knowledge_dma"]))
print((df_social_desirability["knowledge_daa"].sum())/len(df_social_desirability["knowledge_daa"]))
print((df_social_desirability["knowledge_ai"].sum())/len(df_social_desirability["knowledge_ai"]))
print((df_social_desirability["knowledge_gdpr"].sum())/len(df_social_desirability["knowledge_gdpr"]))
print((df_social_desirability["knowledge_tf"].sum())/len(df_social_desirability["knowledge_tf"]))

crosstable_knowledge = pd.crosstab(df_use_time_micro_na["use_time_micro"], df_use_time_micro_na["knowledge_gdpr"])

with open(path_lat / "crosstable 1.txt", "w", encoding="utf-8") as f:
    f.write(crosstable_knowledge.to_latex())

with open(path_lat / "knowledge table.txt", "w", encoding="utf-8") as f:
    f.write(df_out.to_latex(index = False, float_format="%.0f"))

df_trust_average_non_reporting = df[df["reasons"].astype(str) != "[]"]
df_trust_average_reporting = (df[df["reasons"].astype(str) == "[]"])
print(len(df_trust_average_non_reporting))
print(df_trust_average_non_reporting["trust_average"].sum()/len(df_trust_average_non_reporting))
print(len(df_trust_average_reporting))
print(df_trust_average_reporting["trust_average"].sum()/len(df_trust_average_reporting))
print(len(df_trust_average_non_reporting))
print(f"trust removal non reporting: {df_trust_average_non_reporting["trust_remove"].sum()/len(df_trust_average_non_reporting)}")
print(len(df_trust_average_reporting))
print(f"trust removal reporting: {df_trust_average_reporting["trust_remove"].sum()/len(df_trust_average_reporting)}")
print(df["trust_remove"].sum()/len(df))
'''