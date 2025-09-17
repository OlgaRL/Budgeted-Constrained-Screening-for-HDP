# Budgeted-Constrained-Screening-for-HDP
Maccabi Home Task – DS Position
Budget-Constrained Screening for Hypertensive Disorders of Pregnancy (HDP)

End-to-end workflow for triaging at week-15 under a fixed testing budget: EDA → feature curation → ranking models → budgeted evaluation → interpretation and operating-point recommendation.

Repo contents

Jup_MCCB_H_TSK_OlgaRapp.ipynb — main notebook (HTML export included). 

Jup_MCCB_H_TSK_OlgaRapp

Jup_MCCB_H_TSK_OlgaRapp.html — notebook export for quick review. 

Jup_MCCB_H_TSK_OlgaRapp

Main_MCCB_H_TSK_OlgaRapp.py — script version of the pipeline. 

Main_MCCB_H_TSK_OlgaRapp

Budgeted-Constrained Screening for HDP.pptx — slide deck with results and recommendations. 

Budgeted-Constrained Screening …

ds_home_ass.docx — task description / rubric. 

ds_home_ass

Data

Place ds_assignment_data.csv locally and update the path in the code if needed.

Script uses a hard-coded Windows path (csv_path = Path("C:/Users/USER/.../ds_assignment_data.csv")); change this line to your local path. 

Main_MCCB_H_TSK_OlgaRapp

The notebook shows the standard pd.read_csv(csv_path) pattern once csv_path is set. 

Jup_MCCB_H_TSK_OlgaRapp

Tip: for portability, consider placing the CSV in the repo root and setting csv_path = Path("ds_assignment_data.csv").

Environment

Tested with Python 3.10+. Install dependencies:

pip install -U pandas numpy matplotlib scikit-learn scipy statsmodels ipykernel

Quick start
Run the notebook

Open Jup_MCCB_H_TSK_OlgaRapp.ipynb.

Set csv_path, run all cells.

Run the script
python Main_MCCB_H_TSK_OlgaRapp.py


The script:

Loads data, builds a spec-aligned working table, and deduplicates exact duplicate columns. 

Main_MCCB_H_TSK_OlgaRapp

 

Main_MCCB_H_TSK_OlgaRapp

Saves a deduped CSV (adjust the save path if not on Windows). 

Main_MCCB_H_TSK_OlgaRapp

Runs EDA (size, missingness, target distribution) and plots. 

Main_MCCB_H_TSK_OlgaRapp

Trains Logistic Regression, Random Forest, and Gradient Boosting; evaluates at fixed budgets (Top-K). 

Main_MCCB_H_TSK_OlgaRapp

Compares models via recall/precision vs. budget and a decision table at a chosen budget. 

Main_MCCB_H_TSK_OlgaRapp

What the pipeline does

Data alignment & feature curation

Maps labs/vitals to short, consistent names; constructs BP summaries and trends. 

Main_MCCB_H_TSK_OlgaRapp

 

Main_MCCB_H_TSK_OlgaRapp

Keeps demographics; fills missing diagnostic subtype counts; orders columns by spec. 

Main_MCCB_H_TSK_OlgaRapp

 

Main_MCCB_H_TSK_OlgaRapp

Drops exact duplicate columns by hashing. 

Main_MCCB_H_TSK_OlgaRapp

EDA

Size/memory, missingness table, target distribution, numeric summaries. 

Main_MCCB_H_TSK_OlgaRapp

Optional donut chart for Y distribution. 

Main_MCCB_H_TSK_OlgaRapp

Modeling & budgeted evaluation

Train/valid split (80/20 stratified).

Models: Logistic Regression (scaled), Random Forest, Gradient Boosting.

Evaluate at budgets B∈{5,10,15,20,25%}: Recall@B (cases captured) and PPV@B (positive rate among tested).

Compare models and produce a decision table at a chosen budget (e.g., 20%). 

Main_MCCB_H_TSK_OlgaRapp

Interpretation (notebook)

Permutation importance (global) and a simple local LOFO-style explanation for a top-ranked case.

Capacity-to-cutoff table and score histogram with the selected cutoff.

Reproducing the main figures

Missingness & coverage: generated during EDA section in the script/notebook. 

Main_MCCB_H_TSK_OlgaRapp

Correlation heatmap and overlay histograms for key signals (Protein-U, PLT).

Recall/Precision vs. Budget per model and model comparison overlays.

Decision table @ 20% budget summarizing K, Recall, PPV, NNT, and lifts.

Permutation importance (Random Forest) using sklearn.inspection.permutation_importance. 

Jup_MCCB_H_TSK_OlgaRapp

Outputs & where they go

Intermediate deduped CSV path is currently hard-coded to a Windows OneDrive folder; change df_data_dedup_path to a local folder if needed. 

Main_MCCB_H_TSK_OlgaRapp

Plots and tables are shown inline; save paths can be added where desired.

Troubleshooting

FileNotFoundError: set csv_path to the actual location. The script currently uses a Windows path; change it if running elsewhere. 

Main_MCCB_H_TSK_OlgaRapp

Plots not showing when running as a script: ensure a GUI backend or add plt.savefig(...) calls; in Jupyter, outputs render inline by default.

Imputer warnings for all-missing columns: harmless; those features are effectively ignored.

Task alignment (per the assignment)

Task 1 — Data Exploration: size, missingness, distributions, and relationships covered. 

Main_MCCB_H_TSK_OlgaRapp

Task 2 — Feature Engineering: curated labs, vitals, demographics; leakage-prone fields handled; duplicates deduped. 

Main_MCCB_H_TSK_OlgaRapp

 

Main_MCCB_H_TSK_OlgaRapp

Task 3 — Screening-Prioritization Modeling: multiple ranking models trained and compared. 

Main_MCCB_H_TSK_OlgaRapp

Task 4 — Budget-Constrained Evaluation: Recall/PPV across budgets with a clear decision table at B=20%.

Task 5 — Interpretation & Recommendations: global importance + local explanation; capacity-to-cutoff guidance.

For the full rubric and deliverables list, see the attached task brief
