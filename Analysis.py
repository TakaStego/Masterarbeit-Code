
###########################################################################################
###lOADING PACKAGES
###########################################################################################
import pandas as pd
from pandas import DataFrame as DF
import xlsxwriter
import numpy as np
import scipy as scipy
from scipy import stats as stats
from scipy.stats.distributions import chi2
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import textwrap
import statsmodels.api as sm 
from stargazer.stargazer import Stargazer
from statsmodels.nonparametric.smoothers_lowess import lowess

###########################################################################################
###IMPORTING DATA (adapt to your file system)
###########################################################################################

path_viz = Path(r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\output\visualizations")
path_tab = Path(r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\output\tables")
path_lat = Path(r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\output\latex")
path_data = Path(r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data")

def latex(filename, dataframe):
    with open(path_lat / filename, "w", encoding="utf-8") as f:
        f.write(dataframe.round(0).to_latex(float_format="%.0f"))

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

# Setting categories
for col in likert_cols:
    df[col] = pd.Categorical(df[col], categories=likert_order, ordered=True)

for col in df.columns:
    if col.endswith("_label"):
        base = col.replace("_label", "")
        var_name = f"counts_{base}"
        globals()[var_name] = df[col].value_counts()
        print(f"{var_name} erstellt")
    
###Language
df_language = df["UserLanguage"]
count_language = df_language.value_counts()
print(count_language)


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
latex("used networks.txt", counts_networks)

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

#creating and exporting value counts
counts_seen_content = df_exploded_seen_content["exploded_seen_content_label"].value_counts()
share_seen_content = (counts_seen_content/row_count)
print(share_seen_content)
latex("seen content.txt", counts_seen_content)
latex("share seen content.txt", share_seen_content)

#Reported content
counts_reported_content = df["reported_content_label"].value_counts()
print(counts_reported_content)
latex("reported content.txt", counts_reported_content)

###REASONS
reasons_map = {
    1: "Not sure whether content was illegal",
    2: "No knowledge reporting was possible",
    3: "Reporting process too complicated",
    4: "Reporting does not make a difference",
    5: "No trust towards companies",
    6: "Not bothered by content",
    7: "Illegal content is seen to often to care",
    8: "Afraid of negative consequences",
   -1: "Don´t know",
   -2: "Prefer not to answer"
}

df["reasons"] = df["reasons"].apply(safe_split)

#Exploding and mapping labels
df_exploded_reasons = df.explode("reasons")
df_exploded_reasons["labelled_reasons"] = df_exploded_reasons["reasons"].map(reasons_map)

#creating and exporting value coutnts
counts_reasons = df_exploded_reasons["labelled_reasons"].value_counts()
latex("reasons to not report.txt", counts_reasons)

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

#creating and exporting value counts
counts_used_rights = df_exploded_rights["used_rights"].value_counts()
latex("used rights.txt", counts_used_rights)

###########################################################################################
###DESCRIPTIVE ANALYSIS
###########################################################################################
###Socio demographics
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(counts_gender)))

#Gender
fig, ax = plt.subplots()
ax.bar(counts_gender.index, counts_gender.values, color=colors)
fig.savefig(path_viz / "gender_distribution.jpg")
plt.close(fig)

#Education
colors = cmap(np.linspace(0, 1, len(counts_education)))
fig, ax = plt.subplots()
ax.pie(counts_education.values, labels=counts_education.index, colors=colors)
plt.savefig(path_viz / "education_counts.jpg")
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
print(f"Use time micro value counts: {df["use_time_micro_label"].value_counts()}")

#Use time - Macro
fig, ax =plt.subplots(figsize=(12, 6))
ax.bar(counts_use_time_macro.index, counts_use_time_macro.values, color = colors)
fig.savefig(path_viz / "Use time macro.jpg")
plt.close(fig)
print(f"Use time macro value counts: {df["use_time_macro_label"].value_counts()}")

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

#Average knowledge


##Seen content
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

##############################################################
###Scatter- and boxplots
##############################################################
#Average knowledge over usetime
df_use_time_micro_na = df[df["use_time_micro"].between(1, 6)].copy()
f = sns.catplot(data=df[~df["use_time_micro"].isin([-1, -3])], x= "use_time_micro", y="knowledge_average", kind="box")
f.set_axis_labels("Use time per day", "Average knowledge")
f.savefig(path_viz / "boxplot_knowledge_usetime.jpg")
plt.close(f.fig)

#############################################################
###Regressions
#############################################################
###Reporting and using rights as dependent variable
#preparing for logit by creating centered variables
cols_regression = ["age", "interest", "shaping_space", "use_time_micro", "trust_average", "knowledge_dsa"]

#Creating a data frame with a dummy for used rights
#Respondents with "prefer not to answer" (numerical value = -3, but data is in list-format) are excluded, since no definite statement can be made about them 
mask = df["used_rights"].apply(lambda x: -3 in x)
df_used_rights_dummy = df[~mask]

#Creating a data frame with a dummy for used rights
#Respondents with "prefer not to answer" (numerical value = -3, but data is in list-format) are excluded, since no definite statement can be made about them
mask2 = df_used_rights_dummy["seen_content"].apply(lambda x: -3 in x)
df_seen_content_dummy = df_used_rights_dummy[~mask2]

#Creating a data frame with a dummy for reported content
#Respondents with "prefer not to answer" (numerical value = -3) are excluded, since no definite statement can be made about them
df_pre_logit = df_seen_content_dummy[df_seen_content_dummy["reported_content"] != -3.0]
#Centering
for col in cols_regression:
    var_name = f"{col}_centered"
    df_pre_logit[var_name] = (df_pre_logit[col] - df_pre_logit[col].mean())
    print(f"{var_name} erstellt")
df_pre_logit["interaction_term"] = (df_pre_logit["knowledge_dsa_centered"]*df_pre_logit["trust_average_centered"])

#For the logit with reported content as dummy variable, only those respondents with seen_content_dummy == 1 are being considered. Seeing illegal content is a necessary condition for reporting illegal content, irrespective the reporting mechanism.
df_logit = df_pre_logit[df_pre_logit["seen_content_dummy"] == 1]
df_logit = df_logit[df_logit["gender"] != 4]
#Re-creating the centered variables, since cases were removed from the dataframe, thereby also impacting the mean values. 
#creating new interaction term
for col in cols_regression:
    var_name = f"{col}_centered"
    df_logit[var_name] = (df_logit[col] - df_logit[col].mean())
    print(f"{var_name} erstellt")
df_logit["interaction_term"] = (df_logit["knowledge_dsa_centered"]*df_logit["trust_average_centered"])


X = df_logit[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "interaction_term"]].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

#Reporting content as dependent variable
models = []
y=df_logit["reported_content_dummy"]
model = sm.Logit(y, X).fit()
models.append(model)

X = df_pre_logit[["age", "interest", "shaping_space", "use_time_micro", "trust_average_centered", "knowledge_dsa_centered", "interaction_term"]].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
y=df_pre_logit["used_rights_dummy"]
model = sm.Logit(y, X).fit()
models.append(model)

stargazer = Stargazer(models)
with open(path_lat / "Logit reporting & used rights_new.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))

###Knowledge about legislation as dependent variable - OLS or Logit?


###########################################################################################
###CREATING OUTPUT IN EXCEL AND LATEX
###########################################################################################
###Latex
#Crosstable between daily use time and knowledge
crosstable_knowledge = pd.crosstab(df_use_time_micro_na["use_time_micro"], df_use_time_micro_na["knowledge_gdpr"])
latex("test-file.txt", crosstable_knowledge)

#Crosstable between interest and knowledge -- ADAPT TO NOT BE AI KNOWLEDGE - DOES NOT MAKE SENSE
crosstable_interest = pd.crosstab(df["interest_label"], df["knowledge_ai_label"])
latex("crosstable test.txt", crosstable_interest)

#Gender counts
latex("gender distribution.txt", counts_gender)

#Used rights
latex("used rights.txt", counts_used_rights)

#Seen illegal content
latex("seen illegal content.txt", counts_seen_content)

#Reported illegal content
latex("reported illegal content.txt", counts_reported_content)

#Crosstable between people who used their rights and people who made reports
df_filtered = df_exploded_rights[df_exploded_rights["used_rights"].isin([1, 2])]
crosstable_reports = pd.crosstab(df_filtered["used_rights_label"], df_filtered["reported_content_label"], margins=True)
latex("reporting crosstable.txt", crosstable_reports)

#Crosstable between people who have seen illegal content and who reported content
df_exploded_seen_content_na = df_exploded_seen_content[df_exploded_seen_content["reported_content_label"] != "-3"]
crosstable_reporting_seen = pd.crosstab(df_exploded_seen_content_na["exploded_seen_content_label"], df_exploded_seen_content_na["reported_content_label"])
latex("Reporting seen content.txt", crosstable_reporting_seen)

crosstable_reporting_dummy_seen = pd.crosstab(df_exploded_seen_content["exploded_seen_content_label"], df_exploded_seen_content["reported_content_dummy"])
latex("Reporting dummy seen.txt", crosstable_reporting_dummy_seen)

#Crosstable between knowledge about the DSA and reporting
crosstable_reporting_knowledge = pd.crosstab(df["knowledge_dsa_label"], df["reported_content_label"])
latex("Reporting knowledge crosstable.txt", crosstable_reporting_knowledge)

#Count of the reasons not to report illegal content
reasons_counts = df_exploded_reasons["labelled_reasons"].value_counts()
latex("Reasons for not reporting.txt", reasons_counts)

#Crosstable between dummy variable for having used a right under the DSA and seen illegal content
crosstab_rights_seen = pd.crosstab(df_used_rights_dummy["used_rights_dummy"], df_seen_content_dummy["seen_content_dummy"])
print(crosstab_rights_seen)
latex("crosstable used rights & seen content.txt", crosstab_rights_seen)

#Crosstable between daily use time and knowledge of the DSA
crosstab_usetime_knowledge = pd.crosstab(df_use_time_micro_na["use_time_micro"], df_use_time_micro_na["knowledge_dsa_label"])
crosstab_usetime_knowledge = crosstab_usetime_knowledge[knowledge_order]
latex("crosstable use time & knowledge.txt", crosstab_usetime_knowledge)

#Perception of the DSA
df_perception = pd.DataFrame({"Improving protection": counts_improving_protection, "Worrying": counts_worrying})
latex("Perception of the DSA.txt", df_perception)

###Excel
#Setting path and creating workbooks
with pd.ExcelWriter(path_tab / "output.xlsx", engine="xlsxwriter") as writer:
    
    #Gender
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Gender")
    writer.sheets["Knowledge"] = worksheet
    (
    df["gender_label"]
      .value_counts()
      .rename_axis("gender")
      .reset_index(name="count")
      .to_excel(writer, sheet_name="Gender", index=False)
)
    #Knowledge
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Knowledge")
    writer.sheets["Knowledge"] = worksheet
    knowledge_cols = {
    "knowledge_dsa_label": "DSA",
    "knowledge_dma_label": "DMA",
    "knowledge_ai_label": "AI Act",
    "knowledge_gdpr_label": "GDPR",
    "knowledge_daa_label": "DAA",
    "knowledge_tf_label": "Trusted Flaggers"
}

    #Setting a common dataframe for all knowledge values
    df_out = pd.DataFrame({
        name: df[col].value_counts()
        for col, name in knowledge_cols.items()
    })

    #Setting index for column for knowledge labels
    df_out = df_out.reset_index().sort_values("DSA").rename(columns={"index": "Knowledge"})

    #Writing knowledge values to Excel
    df_out.to_excel(writer, sheet_name="Knowledge", index=False)

    #Education
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Education")
    writer.sheets["Education"] = worksheet
    (
    df["isced_label"]
      .value_counts()
      .reset_index(name="Education")
      .to_excel(writer, sheet_name="Education", index=False)
)
    #Used rights
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Used rights")
    writer.sheets["Used rights"] = worksheet
    (
    counts_used_rights
      .reset_index(name="Used rights")
      .to_excel(writer, sheet_name="Used rights", index=False)
)
    #Digital citizenship
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Digital citizenship")
    writer.sheets["Digital citizenship"] = worksheet
    (
    df["citizenship_label"].sort_values()
      .value_counts()
      .reindex(likert_order)
      .to_excel(writer, sheet_name="Digital citizenship", index=True)
)
    #Shaping space - Exporting to same sheet as digital citizenship
    df["shaping_space_label"].value_counts().reindex(likert_order).to_excel(writer, sheet_name = "Digital citizenship", startcol=2, index=True)
    
    #Perception of the DSA
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Perception of DSA")
    writer.sheets["Perception of DSA"] = worksheet
    (
    df["worrying_label"].sort_values()
      .value_counts()
      .reindex(likert_order)
      .to_excel(writer, sheet_name="Perception of DSA", index=True)
)
    df["improving_protection_label"].value_counts().reindex(likert_order).to_excel(writer, sheet_name = "Perception of DSA", startcol=2, index=True)

print(row_count)


print(f" DAS TESTET WIEVIEL DAS KORRELIERT: {df[['isced', 'interest']].corr()}")
print(df_logit["reported_content_dummy"].value_counts())
print(df_pre_logit["used_rights_dummy"].value_counts())
print(df_pre_logit["knowledge_dsa_centered"].mean())
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