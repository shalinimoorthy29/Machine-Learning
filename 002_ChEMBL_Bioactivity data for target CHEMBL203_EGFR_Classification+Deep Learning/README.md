# Predicting Bioactivity of Pre-Clinical Drug-Like Compounds Against EGFR Using Classification-Based Machine Learning and Deep Learning

This project utilises **classification-based machine learning** and **deep learning** to predict the bioactivity of drug-like compounds against the Epidermal Growth Factor Receptor (EGFR). EGFR is a significant target in cancer treatment, and identifying active compounds can aid in developing potent cancer therapeutics.

## Dataset
- **Source:** ChEMBL Database
- **Dataset URL:** [EGFR Bioactivity Dataset](https://www.ebi.ac.uk/chembl/web_components/explore/activities/STATE_ID:At7xOAwu8FvFSLN4R0SYKA%3D%3D)

The dataset contains bioactivity data for compounds tested against EGFR. Bioactivity values, given as IC50, were transformed to pIC50 and categorised into classes (e.g., Active, Intermediate, or Inactive) for classification analysis.

## Objectives
1. Classify compounds as **Active**, **Intermediate**, or **Inactive** against EGFR using both machine learning and deep learning approaches.
2. Identify significant molecular descriptors associated with EGFR inhibition.
3. Interpret model predictions to guide early-stage drug discovery and compound prioritisation.

## Methodology

### Data Cleaning and Transformation
- Filtered and prepared the dataset, converting IC50 values to pIC50 and categorising them into bioactivity classes.
- Cleaned redundant columns to retain only essential molecular descriptors for model training.

### Feature Engineering
- **Molecular Descriptors**: Calculated descriptors, such as **Molecular Weight**, **LogP**, **Hydrogen Bond Donors/Acceptors**, and **Ligand Efficiency Metrics** using RDKit.
- **Label Encoding**: Categorical labels were encoded for classification purposes.

### Modelling

#### Machine Learning Models
- **Primary Models**: Random Forest, XGBoost, SVC, and LightGBM classifiers were tested for performance.
- **Hyperparameter Tuning**: Applied GridSearchCV for optimising parameters and improving classification accuracy.
- **Evaluation Metrics**: Used confusion matrix, accuracy, F1 score, and recall to evaluate each model’s performance.

#### Deep Learning Model
- **Model Architecture**: Designed a neural network using TensorFlow/Keras, with fully connected dense layers and ReLU activation functions. The network architecture was optimised based on validation performance.
- **Training and Optimisation**: Utilised dropout regularisation and Adam optimiser to prevent overfitting and enhance model generalisation.
- **Evaluation**: Assessed deep learning model using accuracy and loss metrics on the test set to compare with traditional classifiers.

### Model Interpretability
- **SHAP Values**: SHAP (SHapley Additive exPlanations) values were used to interpret both the machine learning and deep learning models, identifying influential features that impact classification predictions.

## Results
### Model Performance
- **Best Machine Learning Model**: The Random Forest classifier achieved high classification accuracy, effectively distinguishing active compounds.
- **Deep Learning Performance**: The neural network performed comparably, showing strong accuracy, especially in detecting active and inactive compounds, while slightly underperforming on intermediate classes.

### Key Findings
- **Top Features**: Descriptors such as **LogP**, **Ligand Efficiency LLE**, and **Hydrogen Bond Acceptors** were found significant in classifying EGFR inhibition levels.
- **Comparative Insights**: Both the machine learning and deep learning approaches offered robust classification, with deep learning demonstrating potential for further tuning and generalisation improvements.

## Conclusion
The combined use of machine learning and deep learning models provides a comprehensive framework for predicting EGFR bioactivity. The study underscores the value of molecular descriptors in guiding compound prioritisation and offers scalable methods for drug discovery applications.

## Next Steps
- **Deep Learning Enhancements**: Explore additional architectures (e.g., CNNs for structural data) to further improve classification of complex molecular properties.
- **Model Deployment**: Implement a web-based application (e.g., Flask) for bioactivity predictions, accessible for real-time drug discovery support.
- **Advanced Feature Selection**: Integrate feature selection techniques to refine descriptors for even greater model interpretability and performance.
