# Predicting Bioactivity of Drug Compounds Against EGFR Using Machine Learning

This project demonstrates a **regression-based approach** to predict the bioactivity (pIC50) of drug-like compounds against the Epidermal Growth Factor Receptor (EGFR), a crucial target in cancer drug discovery. By predicting continuous IC50 values, the model provides valuable insights into compound potency and can aid in identifying potential inhibitors for EGFR, an oncogenic protein associated with cell proliferation and survival in multiple cancers.

## Dataset
- **Source:** ChEMBL Database
- **Dataset URL:** [EGFR Bioactivity Dataset](https://www.ebi.ac.uk/chembl/web_components/explore/activities/STATE_ID:At7xOAwu8FvFSLN4R0SYKA%3D%3D)

The dataset consists of bioactivity data for compounds tested against EGFR. Each molecule’s IC50 value is provided, along with several molecular descriptors. The IC50 values (inhibitory concentration 50%) were transformed to pIC50 (log-transformed IC50), making the data suitable for regression analysis and providing a continuous target variable to better understand bioactivity levels.

## Objectives
1. Predict the bioactivity (pIC50) of drug-like compounds against EGFR using a **regression-based machine learning model**.
2. Identify key molecular descriptors that contribute to EGFR inhibition, providing insights for early-stage drug discovery.
3. Understand the relationship between molecular descriptors and pIC50 values to guide compound selection and optimisation for EGFR-targeted therapies.

## Methodology
### Data Cleaning and Transformation
- Filtered the dataset to include only relevant entries with IC50 values in nM units and standardised all measurements to this unit.
- Converted IC50 values to pIC50 using a log transformation, facilitating continuous target variable regression modelling.
- Dropped redundant or irrelevant columns and ensured all necessary descriptors were available for model training.

### Feature Engineering
- **Molecular Descriptors**: Calculated descriptors like **Molecular Weight**, **LogP** (octanol-water partition coefficient), **Hydrogen Bond Donors/Acceptors**, and **Ligand Efficiency Metrics** using RDKit.
- **Feature Selection**: Selected essential features such as **Ligand Efficiency LLE**, **LogP**, and **#RO5 Violations** based on domain knowledge and exploratory analysis.

### Modelling
- **Primary Model**: Random Forest Regressor, chosen for its robustness and ability to handle complex interactions between features.
- **Hyperparameter Tuning**: Performed extensive tuning using GridSearchCV to optimise parameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`) and improve model performance.
- **Evaluation Metrics**: Assessed the model using **R²** (coefficient of determination) and **RMSE** (Root Mean Squared Error) on both training and test sets to evaluate accuracy and predictive power.

### Feature Importance and Correlation Analysis
- **Feature Importance**: Analysed the importance of each feature in the Random Forest model, identifying **Ligand Efficiency LLE** and **LogP** as the most impactful descriptors for predicting pIC50.
- **Correlation Matrix**: Generated a heatmap of the top features to investigate relationships between descriptors and pIC50, revealing significant correlations that align with established chemical and biological knowledge.

## Results
### Model Performance
- **R² Score**: 0.998 (after tuning), indicating that the model explains nearly all the variance in pIC50 values.
- **RMSE**: 0.135, reflecting low error and high accuracy in predictions.

### Key Findings
- **Top Features**: The model identified **Ligand Efficiency LLE** and **LogP** as the most influential factors, consistent with their known importance in compound potency and drug-like properties.
- **Correlation Insights**: A strong correlation between Ligand Efficiency LLE and pIC50 values suggests this descriptor plays a critical role in EGFR inhibition, while LogP, Molecular Weight, and RO5 violations also contribute to bioactivity but to a lesser extent.

### High Bioactivity Analysis
- Compounds with high bioactivity (pIC50 ≥ 8) were further analysed to identify features that significantly influence potent EGFR inhibition.
- Distribution analysis of pIC50 values shows that while many compounds display moderate bioactivity (pIC50 ≥ 6), fewer exhibit very high potency (pIC50 ≥ 9 or 10), emphasising the challenge in identifying strong inhibitors.

## Conclusion
This regression-based project provides a comprehensive workflow for predicting the bioactivity of EGFR inhibitors, highlighting critical molecular descriptors that influence bioactivity. By leveraging machine learning in a regression framework, this study offers insights into compound prioritisation and optimisation in EGFR-targeted drug discovery, with potential applications in oncology.

## Next Steps
- **Classification Analysis**:  
  Classification Analysis: A potential next step could involve defining more stringent bioactivity thresholds (e.g., pIC50 ≥ 8 for Highly Active, 6 ≤ pIC50 < 8 for Moderately Active, and 5 ≤ pIC50 < 6 for Intermediate) to classify compounds based on varying levels of activity. Conducting a classification analysis with these categories may offer additional insights by highlighting the molecular properties that differentiate highly potent compounds from those with moderate or low activity. This approach complements the regression analysis by providing a categorical perspective on compound efficacy and facilitating targeted identification of strong inhibitors.

- **Alternative Models**:  
  Explore other machine learning models, such as XGBoost, LightGBM, or Support Vector Machines, and compare their performance to the Random Forest model to assess if further improvements in accuracy can be achieved.

- **Deployment**:  
  Deploy the trained model as a web application using Flask or a cloud platform (e.g., AWS) to provide bioactivity predictions for new compounds, facilitating real-time support for EGFR-focused drug discovery.

---

This project demonstrates the potential of machine learning to assist in the early stages of drug discovery by identifying influential features and providing accurate predictions of bioactivity against critical cancer targets like EGFR.
