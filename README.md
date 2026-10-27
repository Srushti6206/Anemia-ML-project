# Machine Learning-Based Anemia Classification Among Pregnant Women

## 📌 Project Overview

This project uses Machine Learning (ML) to classify anemia severity among pregnant women using data from the **National Family Health Survey (NFHS-5), India**.

The system predicts one of four anemia categories based on selected demographic, socioeconomic, and pregnancy-related features. It explores multiple ML algorithms and uses SHAP (SHapley Additive exPlanations) to interpret the features influencing model predictions.

**Note:** The model classifies anemia severity; it does not independently confirm iron deficiency or provide a clinical diagnosis.

## 🎯 Objectives

- Develop an ML-based anemia classification system for pregnant women.
- Preprocess survey data and handle missing values.
- Train and compare Random Forest and XGBoost classifiers.
- Address class imbalance using balanced sample weights.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Interpret feature contributions using SHAP.

## 📊 Dataset

- **Source:** National Family Health Survey (NFHS-5), India.
- **Dataset type:** Individual Recode (IR) survey data.
- **Target variable:** `v457` — anemia classification.
- **Population:** Respondents identified as currently pregnant with a known anemia classification.

### Target Classes

| Class label | Anemia category |
|---|---|
| 1 | Severe anemia |
| 2 | Moderate anemia |
| 3 | Mild anemia |
| 4 | Not anemic |

The dataset contains demographic, educational, socioeconomic, and pregnancy-related variables. The hemoglobin measurements used to derive the anemia classification were excluded from the predictor features to reduce target leakage.

The original survey dataset is not included in this repository. Obtain it through the official DHS/NFHS data access process and comply with its data-use conditions.

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Joblib

## ⚙️ Project Workflow

1. **Data loading:** Load selected variables from the NFHS-5 dataset.
2. **Data filtering:** Select pregnant respondents with known anemia classifications.
3. **Preprocessing:** Handle missing values using median imputation and create a missingness indicator for age at first birth.
4. **Feature selection:** Exclude the target variable and direct hemoglobin measurements from the predictors.
5. **Train-test split:** Split the dataset into 80% training and 20% testing data using stratification.
6. **Model training:** Train Random Forest and XGBoost classifiers.
7. **Class imbalance handling:** Train a balanced XGBoost model using sample weights.
8. **Evaluation:** Compare model accuracy, macro F1-score, and class-specific recall.
9. **Explainability:** Use SHAP to examine the contribution of input features to predictions.

## 📈 Model Performance

The following results were obtained on the held-out test set.

| Model | Accuracy | Macro F1-score | Severe anemia recall |
|---|---:|---:|---:|
| Random Forest | 47.22% | 0.2599 | 0.00% |
| XGBoost | 49.74% | 0.2377 | 0.00% |
| Balanced XGBoost | 36.48% | 0.2762 | 22.62% |

### Results Interpretation

- Standard XGBoost achieved the highest overall accuracy.
- Balanced XGBoost achieved the highest macro F1-score among the three models.
- Class balancing improved recall for severe anemia from 0% to approximately 22.62%.
- The balanced model had lower overall accuracy, demonstrating a trade-off between overall correctness and minority-class detection.
- Severe-anemia precision was approximately 2%, indicating that many severe-anemia predictions were incorrect.

These results show that the models learned some patterns in the data, but predictive performance remains limited, particularly for severe anemia.

## 🔍 SHAP Explainability

SHAP was used to analyze feature contributions to the XGBoost model's predictions.

Features appearing among the most influential in the feature-importance plot included:

- Duration of current pregnancy (`v214`)
- Education in single years (`v133`)
- Body Mass Index (`v445`)
- Respondent's age (`v012`)
- Births in the last five years (`v208`)
- Household wealth index (`v190`)

SHAP importance describes the magnitude of feature contributions to model outputs. It does not establish causation or prove that a feature increases or decreases anemia severity.

See `results/shap_feature_importance.png` for the feature-importance visualization, if included.


## 🚀 How to Run

1. Open `anemia_classification.ipynb` in Google Colab.
2. Obtain the required NFHS-5 dataset through the authorized data-access process.
3. Upload or mount the dataset in Colab.
4. Install the required libraries if they are not already available:

   ```python
   %pip install pandas numpy scikit-learn xgboost shap pyreadstat joblib matplotlib
   ```

5. Update the dataset path in the notebook.
6. Run the notebook cells in order to preprocess the data, train the models, evaluate performance, and generate SHAP plots.

The saved model files can be loaded with Joblib, provided the compatible Python and library versions are used.

## ⚠️ Limitations

- The model's predictive performance is limited, and it should not be used for clinical decision-making.
- Severe-anemia precision is low, so predictions require particular caution.
- Class imbalance affects the model's ability to classify minority categories.
- The model has not been independently validated on an external dataset.
- The target variable represents anemia classification and does not independently confirm iron deficiency.
- Survey data may contain limitations in measurement, missingness, and representativeness.
- Feature importance indicates predictive contribution, not causation.

## 🔮 Future Improvements

- Improve minority-class prediction and calibration.
- Evaluate additional metrics and confusion matrices.
- Validate the model using an independent dataset.
- Investigate feature selection and hyperparameter tuning.
- Assess the suitability of pregnancy-related BMI measurements.
- Explore datasets containing direct iron-status indicators if the objective is specifically to identify iron deficiency.

## 🏁 Conclusion

This project demonstrates an end-to-end machine learning workflow for classifying anemia severity among pregnant women using NFHS-5 data. It implements Random Forest, XGBoost, class balancing, performance evaluation, and SHAP-based explainability.

Balanced XGBoost achieved the highest macro F1-score among the evaluated models and improved severe-anemia recall, although overall accuracy and severe-anemia precision remained limited. Further model development and external validation are needed before any practical or clinical application.

## 📚 Data Source

National Family Health Survey (NFHS-5), India, 2019–2021.

DHS Program: https://dhsprogram.com/

India NFHS portal: https://rchiips.org/nfhs/


