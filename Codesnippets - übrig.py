###Codesnippets
###Use time - Macro

def counting_value():
    for col in df:
        if col is in numeric_cols:


#Interest in digital policy
counts_interest = df["interest_label"].value_counts()

#Seeing the rights granted as possibility to shape the digital space
counts_shaping_space = df["shaping_space_label"].value_counts()

#Seeing oneself as digital citizen
counts_citizenship = df["citizenship_label"].value_counts()

#Improving protection
counts_improving_protection_label = df["improving_protection_label"].value_counts()

#Worrying about governmental control
counts_worrying_label = df["worrying_label"].value_counts()

###SOCIO-DEMOGRAPHICS
#Education
counts_education_label = df["education_label"].value_counts()

#ISCED-Values
counts_isced_label = df["isced_label"].value_counts()

#Age
df_age = df["age"]

#Age groups
counts_age_group = df["age_group"].dropna().value_counts()

#Gender
counts_gender = df["gender_label"].value_counts()

counts_use_time_macro = df["use_time_macro_label"].value_counts()

#Use time - Micro
counts_use_time_micro = df["use_time_micro_label"].value_counts()

#Activeness
counts_activeness = df["activeness"].value_counts()

#Activeness - LABEL IN ACTIVENESS DATA SET

###TRUST
#Review
counts_trust_review = df["trust_review_label"].value_counts()

#Data
counts_trust_data = df["trust_data_label"].value_counts()

#Remove

counts_trust_remove = df["trust_remove_label"].value_counts()

#Compliance
counts_trust_comply = df["trust_comply_label"].value_counts()

###KNOWLEDGE
#GDPR
counts_knowledge_gdpr = df["knowledge_gdpr_label"].value_counts()

#AI Act
counts_knowledge_ai = df["knowledge_ai_label"].value_counts()

#DSA
counts_knowledge_dsa = df["knowledge_dsa_label"].value_counts()

#DMA
counts_knowledge_dma = df["knowledge_dma_label"].value_counts()

#DAA
counts_knowledge_daa = df["knowledge_daa_label"].value_counts()

#Trusted flaggers
counts_knowledge_tf = df["knowledge_tf_label"].value_counts()
print(counts_knowledge_tf)

###GPT
##############KURZER TEST
# === Fit dein Logit-Modell ===
y = df["reported_content_dummy"].astype(int)
X = sm.add_constant(df[["age", "use_time_micro", "interest"]]).replace([np.inf, -np.inf], np.nan).dropna()
model_partial = sm.Logit(y.loc[X.index], X).fit()

# === Predicted probabilities ===
df.loc[X.index, "p_hat"] = np.clip(model_partial.predict(X), 1e-6, 1 - 1e-6)

# === Logit-Transformation ===
df.loc[X.index, "logit"] = np.log(df.loc[X.index, "p_hat"] / (1 - df.loc[X.index, "p_hat"]))

# === Cleaning ===
df_plot = df.loc[X.index, ["use_time_micro", "logit"]].replace([np.inf, -np.inf], np.nan).dropna()

# === LOWESS-Smooth ===
smooth = lowess(df_plot["logit"], df_plot["use_time_micro"], frac=0.3)

# === Plot ===
plt.scatter(df_plot["use_time_micro"], df_plot["logit"], alpha=0.3, label="Logit(p)")
plt.plot(smooth[:, 0], smooth[:, 1], color="red", label="LOWESS")
plt.xlabel("Age")
plt.ylabel("Logit(P)")
plt.legend()
plt.title("Check linearity in the logit: micro use time")
plt.show()


'''
#Reporting
df["used_rights_dummy"] = (df["used_rights"] !=-3|1).astype(int)
X = df[["interest", "use_time_micro", "age", "isced", "gender"]].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

models = []
y=df["reported_content_dummy"].replace([np.inf, -np.inf], np.nan)
y=y.fillna(y.mean())
model = sm.OLS(y, X).fit()
models.append(model)
stargazer = Stargazer(models)
with open(path_lat / "regression reporting.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))
'''
'''
###Modell-tests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera

def check_model_diagnostics(model, df, predictors):
    """
    model: dein sm.OLS.fit() Ergebnis
    df: DataFrame mit den Originaldaten
    predictors: Liste der erklärenden Variablen (ohne 'const')
    """

    print("="*80)
    print("📊 REGRESSIONSDIAGNOSTIK")
    print("="*80)

    # --- Residuen ---
    residuals = model.resid
    fitted = model.fittedvalues

    # 1️⃣ Residuen vs. Fitted
    plt.figure(figsize=(6,4))
    plt.scatter(fitted, residuals)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residuals vs. Fitted")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

    # 2️⃣ Verteilung der Residuen
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Residuals")
    plt.show()

    # --- Jarque-Bera Test (Normalität) ---
    jb_stat, jb_p = jarque_bera(residuals)
    print(f"Jarque–Bera-Test: stat={jb_stat:.3f}, p={jb_p:.3f}")
    if jb_p > 0.05:
        print("✅ Residuen sind annähernd normalverteilt.")
    else:
        print("⚠️  Residuen weichen signifikant von der Normalverteilung ab.")
    print()

    # --- Breusch–Pagan Test (Homoskedastizität) ---
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    labels = ["LM stat", "LM p-value", "F stat", "F p-value"]
    bp_results = dict(zip(labels, bp_test))
    print("Breusch–Pagan-Test:")
    for k, v in bp_results.items():
        print(f"  {k}: {v:.4f}")
    if bp_results["LM p-value"] > 0.05:
        print("✅ Keine Hinweise auf Heteroskedastizität.")
    else:
        print("⚠️  Heteroskedastizität möglich – nutze robuste Standardfehler.")
    print()

    # --- VIF (Multikollinearität) ---
    X = df[predictors].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("Variance Inflation Factors (VIF):")
    print(vif_data.to_string(index=False))
    print("="*80)

predictors = ["interest", "use_time_micro", "age", "education"]
check_model_diagnostics(model_knowledge, df, predictors)
'''

print(np.average(df["Duration (in seconds)"]))

print(df["trust_average"].sum()/len(df))

###########################################################################################
###FIRST IMPRESSIONS
###########################################################################################


def latex(filename, dataframe):
    with open(path_lat / filename, "w", encoding="utf-8") as f:
        f.write(dataframe.round(0).to_latex(float_format="%.0f"))

        
print(row_count)

df_cor = X.corr()
print(f"THIS IS CHECKING FOR MULTICOLLINEARITY: {pd.DataFrame(np.linalg.inv(X.corr().values), index = df_cor.index, columns=df_cor.columns)}")

print(f" DAS TESTET WIEVIEL DAS KORRELIERT: {X[['use_time_micro', 'seen_content_dummy']].corr()}")

#Reporting content MAKE SURE THIS IS LOGIT OR PROBIT + include dummy variable for seen content
X = df_reported_content_dummy[["citizenship", "shaping_space", "trust_average", "age", "gender", "knowledge_dsa", "use_time_micro", "interest", "seen_content_dummy"]].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

models = []
for target in ["reported_content_dummy"]:
    y=df_reported_content_dummy[target]
    model_full = sm.Logit(y, X).fit()
    models.append(model_full)

stargazer = Stargazer(models)
with open(path_lat / "Logit reporting & used rights.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))


obs = pd.crosstab(df["reported_content_dummy"], df["citizenship"])
print(obs)
res = scipy.stats.chi2_contingency(obs)
print(res)

#Creating and saving the plot
f_new = sns.scatterplot(data=df[~df["use_time_micro"].isin([-1, -3])], x= "use_time_micro", y="knowledge_average")
export = f_new.get_figure()
export.savefig(path_viz / "scatterplot_knowledge_usetime.jpg")
plt.close()


fig, ax = plt.subplots()
X = sm.add_constant(df[["interest", "isced"]])  # Beispiel
y = df["knowledge_dsa"]
model = sm.OLS(y, X).fit()
# Residuen checken
sm.qqplot(model.resid, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()
# Histogramm der Residuen
plt.hist(model.resid, bins=20, edgecolor="black")
plt.title("Residual Distribution")
plt.show()



#####OLS-Tryout
df_ols = df_logit
df_ols = df_ols[df_ols["reported_content"] != 2]
df_ols.loc[df_ols["reported_content"] == 3, "reported_content"] = 0
df_ols = df_ols[df_ols["reported_content"].notna()]
df_ols.to_csv(
    r"C:\Users\Admin\OneDrive\Desktop\Master\4. Semester\Masterarbeit\Data\cleaned\df_ols.csv",
    sep=";")

#Using rights under the DSA as dependent variable
X = df_ols[["age", "gender", "interest", "interaction_term"]].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

models1 = []
y=df_ols["reported_content"]
print(len(df_ols))
model1 = sm.Logit(y, X).fit()
models1.append(model1)

stargazer = Stargazer(models1)
with open(path_lat / "Test OLS.txt", "w", encoding = "utf-8") as f:
    f.write(stargazer.render_latex(stargazer))