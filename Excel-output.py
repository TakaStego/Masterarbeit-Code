####
#THINK REALLY CRITICALLY IF THESE PARTS OF CODE ARE NEEDED - MOST PROBABLY JUST TRANSFER THEM TO LATEX-TABLES - NO REDUNDANCY AND CONFUSION ABOUT DOUBLE TABLES
###Excel
#Setting path and creating workbooks
'''    
#Knowledge --> Done
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

    #Education --> done
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Education")
    writer.sheets["Education"] = worksheet
    (
    df["isced_label"]
      .value_counts()
      .reset_index(name="Education")
      .to_excel(writer, sheet_name="Education", index=False)
)

    #Used rights --> done
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Used rights")
    writer.sheets["Used rights"] = worksheet
    (
    counts_used_rights
      .reset_index(name="Used rights")
      .to_excel(writer, sheet_name="Used rights", index=False)
)
    #Digital citizenship --> done
    workbook  = writer.book
    worksheet = workbook.add_worksheet("Digital citizenship")
    writer.sheets["Digital citizenship"] = worksheet
    (
    df["citizenship_label"].sort_values()
      .value_counts()
      .reindex(likert_order)
      .to_excel(writer, sheet_name="Digital citizenship", index=True)
)
    #Shaping space - Exporting to same sheet as digital citizenship --> done
    df["shaping_space_label"].value_counts().reindex(likert_order).to_excel(writer, sheet_name = "Digital citizenship", startcol=2, index=True)
    
    #Perception of the DSA --> done
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
'''