# Healthcare Fraud Detection
### Problem statement - Fraud Classification
Do an exploratory analysis of the dataset provided, decide on feature selection, preprocessing before training a model to classify as class `0` or class `1` and train the model using any technique.

### Running this project on local machine
Dependencies:
- Python 3.11

#### 1. Clone this repository
```bash
git@github.com:neeraj1909/healthcare_fraud_classification.git
cd healthcare_fraud_classification
```

#### 2. Create the virtualenv using `venv`
Run following command from `healthcare_fraud_classification/` directory:
```bash
python3 -m venv venv
```

#### 3. Install python packages
Activate the virtualenv which you created in the previous step:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Run Jupyter Notebook
```bash
jupyter notebook
```

## Relevant Thought Process
<!-- 1. We have three types of data- beneficiary data, inpatient data and outpatient data.
First we analyzed the beneficiary training data, inpatient training data, outpatient training data, and provider data.

2. For the further analysis, we merged the inpatient data and outpatient data on the common features to create the `train_ip_op_data`. After that benificiary data is merged with this train_ip_op_data, and we created the `train_data`. Finally, this `train_data` is merged with provider data to create the `train_data_with_provider`. -->

<!-- 3. We did the Exploratory Data Analysis (EDA) on the `train_data_with_provider` data. 
The list of performed EDA is:
(a) Distribution of PotentialFraud in the Aggregated Data
(b) Percentage distribution of `Fraud` and `Non-Fraud`: ~38% cases are from Fraud category while rest of the cases are from Non-Fraudulent category.
(c) Statewise beneficiary distribution of `Fraud` and `Non-Fraud`: States 5,10,45 are top states in terms of beneficiary percentage
(d) Countywise distribution of `Fraud` and `Non-Fraud`: we observe that same 'Country' codes have similar percentages of fraudulent and non-fraudulent claims
(e) Racewise distribution of `Fraud` and `Non-Fraud`: We can see the similar percentage for fraudulent and non-fraudulent cases for each Race. So, this feature is not helpful.
(f) Genderwise distribution of `Fraud` and `Non-Fraud`: We can see the similar percentage for fraudulent and non-fraudulent cases for both the Gender. So, this feature is not helpful.
(g) ClmDiagnosisCode_1 distribution of `Fraud` and `Non-Fraud`: we observed that diagnosis 4019, 4011, and 2724 are top diagnosis (in terms of money involved)
(h) ClmProcedureCode_1 distribution of `Fraud` and `Non-Fraud`: we observed that Procedure 9904, 8154, and 66 are top procedures (in terms of money involved)
(i) AttendingPhysician distribution of `Fraud` and `Non-Fraud`: Attending Physician PHY330576, PHY350277, and PHY412132 are top Attending Physician (in terms of number of frauds)
(j) OperatingPhysician distribution of `Fraud` and `Non-Fraud`: Operating Physician PHY330576, PHY424897, and PHY314027 are top Operating Physician (in terms of number of frauds)
(k) OtherPhysician distribution of `Fraud` and `Non-Fraud`: Other Physician PHY412132, PHY341578, PHY338032, and PHY337425 are top Other Physician (in terms of number of frauds)
(l) Age-Group distribution of `Fraud` and `Non-Fraud`: we observe that most of the fraudulent cases are for the age group of old (60-80) and very old people (80+).
(m) 'IPAnnualDeductibleAmt and IPAnnualReimbursementAmt in both `Fraud` and `Non-Fraud` categories: there is no visible difference in IpAnnualDeductibleAmt and IPAnnualReimbursementAmt.
(n) 'DeductibleAmtPaid and InscClaimAmtReimbursed in both `Fraud` and `Non-Fraud` categories: We can not differentiate between fraud and non fraud cases based only on DeductibleAmtPaid and 
InscClaimAmtReimbursed.This lets us derive more features from datasets.
(o) 'Insurance Claim Amount Reimbursed vs Age' in both `Fraud` and `Non-Fraud` categories: We see that occurance of fraud cases is more frequent in lower age groups(30-70 years) compared to higher 
age groups(70+ years).
(p) Providers with more than one claim:  we can see that Provider PRV51004 has filed the 149 claims and provider PRV51005 has filed 1165 claims. -->

1. **Data Analysis Preparation:**
   - Initially, we segmented the available data into three main types: beneficiary data, inpatient data, and outpatient data.
   - Each dataset was individually analyzed to understand its structure and contents.

2. **Data Integration:**
   - We merged the inpatient and outpatient datasets based on common features to create a unified dataset called `train_ip_op_data`.
   - The beneficiary data was then merged with `train_ip_op_data` to form the `train_data`.
   - Finally, the `train_data` was enriched by merging it with the provider data to create `train_data_with_provider`.

3. **Exploratory Data Analysis (EDA):**
    - Conducted EDA on `train_data_with_provider` to gain insights into the dataset.
    - Key analyses performed include:
        - Distribution of PotentialFraud in the Aggregated Data.
            - observed the distribution of potential fraud cases within the dataset.
        - Percentage distribution of Fraud and Non-Fraudulent cases.
            - Approximately 38% of cases belong to the Fraud category, while the rest are Non-Fraudulent.
        - Statewise and Countywise beneficiary distribution.
            - States 5, 10, and 45 have the highest beneficiary percentages. Similarly, each 'Country' codes have similar percentages of fraudulent and non-fraudulent claims
        - Racewise and Genderwise distribution analysis.
            - Similar percentages are observed for both fraudulent and non-fraudulent cases within each race, and gender, rendering this feature less helpful.
        - Analysis of top diagnosis, procedures, attending physicians, operating physicians, and other physicians.
            - Diagnosis codes 4019, 4011, and 2724 are among the top diagnoses in terms of monetary involvement.
            - Procedures 9904, 8154, and 66 are top procedures in terms of monetary involvement.
            - Top attending physicians for fraud cases include PHY330576, PHY350277, and PHY412132.
            - Top operating physicians for fraud cases include PHY330576, PHY424897, and PHY314027.
            - Top other physicians for fraud cases include PHY412132, PHY341578, PHY338032, and PHY337425.
        - Age-group distribution analysis.
            - Most fraudulent cases occur in the age group of old (60-80) and very old people (80+).
        - Examination of deductible and reimbursement amounts.
            - No significant difference is observed in `IPAnnualDeductibleAmt` and `IPAnnualReimbursementAmt` between Fraud and Non-Fraud categories.
        - Analysis of insurance claim amount reimbursed vs age.
            - Fraud occurrences are more frequent in lower age groups (30-70 years) compared to higher age groups (70+ years).
        - Identification of providers with multiple claims.
            - Provider PRV51004 filed 149 claims, while PRV51005 filed 1165 claims.

4. **Model Training:**
    - We have observed that all the providers in the `Provider` data is not present in the beneficiary data, inpatient data and outpatient data.
    - We added the `number_of_claims` feature
    - We processed the `claim_diagnose` and `claim_procedure` in the aggregated training as well test data.
    - For the aggregated training and test data, we removed the missing values for `AttendingPhysician`, `OperatingPhysician`, and `OtherPhysician`, and converted their data type as `int64`.
    - For the `ChronicCondition`, we replaced the classes `2` with `0`.
    - For `RenalDiseaseIndicator`, we replaced the class `Y` by `1`.
    - For the aggregated training data, we dleted the column that have only one value.
    - For the training purpose, we selected only those features that are either `integer` or `float` type.
    - We used the `Kendall ` correlation to further select only those features which are positive or negatively correlated above the threshold `0.01`
    - We convert the categorical values to numerical values for `PotentialFraud` feature.
    - We used the 5-Fold for cross-validation.
    - We used the following five models for the training:
        - Logistic Regression
        - Random Forest
        - LightGBM
        - Decision Tree
        - SVM
        - Artificial Neural Network (ANN)

5. **Model Performance Analysis**
    - Model's Performance
        - For the `Logistic Regression` model, the Accuracy is (`0.44 +/-  0.28`), Precision is (`0.13 +/-  0.10`), Recall is (`0.60 +/-  0.49`), `F1-Score` is (`0.21 +/-  0.17`), and `ROC-AUC` is (`0.50 +/-  0.00`).

        - For the `Random Forest` model, the Accuracy is (`0.84 +/-  0.01`), Precision is (`0.71 +/-  0.05`), Recall is (`0.43 +/-  0.05`), `F1-Score` is (`0.53 +/-  0.03`), and `ROC-AUC` is (`0.69 +/-  0.02`).
        
        - For the `LightGBM` model, the Accuracy is (`0.83 +/-  0.01`), Precision is (`0.63 +/-  0.03`), Recall is (`0.44 +/-  0.05`), `F1-Score` is (`0.52 +/-  0.04`), and `ROC-AUC` is (`0.69 +/-  0.02`).

        - For the `Decision Tree` model, the Accuracy is (`0.84 +/-  0.01`), Precision is (`0.68 +/-  0.03`), Recall is (`0.47 +/-  0.09`), `F1-Score` is (`0.55 +/-  0.06`), and `ROC-AUC` is (`0.71 +/-  0.04`).

        - For the `SVM` model, the Accuracy is (`0.84 +/-  0.01`), Precision is (`0.78 +/-  0.04`), Recall is (` 0.37 +/-  0.05`), `F1-Score` is (`0.50 +/-  0.05`), and `ROC-AUC` is (`0.67 +/-  0.02`).

        - For the `ANN` model, the Accuracy is (`0.85 +/-  0.01`), Precision is (`0.80 +/-  0.06`), Recall is (`0.39 +/-  0.04`), `F1-Score` is (`0.52 +/-  0.03`), and `ROC-AUC` is (`0.68 +/-  0.02`).
    

    - For Prediction on the test data, we used the `Decision Tree` model.
    

    - ### Performance Risks
        - The `Logistic Regression` model shows very low performance across all metrics with particularly high variability, especially in Recall (`0.60 +/- 0.49`) and Precision (`0.13 +/- 0.10`). This indicates a substantial risk of both false positives and false negatives, rendering it unreliable for critical decision-making processes. The ROC-AUC score (`0.50 +/- 0.00`) is exactly 0.5, suggesting performance no better than random guessing, which highlights a significant risk in its predictive capability.

        - The `Random Forest` model demonstrates strong performance in Accuracy (`0.84 +/- 0.01`) and Precision (`0.71 +/- 0.05`), but with a moderate Recall (`0.43 +/- 0.05`). The low variability in its performance metrics indicates consistent performance across validation sets. However, the Recall suggests that it might miss a significant number of positive cases, posing a risk for applications where identifying all positive cases is crucial.

        - `LightGBM` has similar performance to Random Forest in terms of Accuracy (`0.83 +/- 0.01`) and ROC-AUC (`0.69 +/- 0.02`), but with slightly lower Precision (`0.63 +/- 0.03`) and similar Recall (`0.44 +/- 0.05`). This suggests a consistent performance with a risk similar to that of Random Forest, particularly in missing positive cases.

        - The `Decision Tree` model shows high Accuracy (`0.84 +/- 0.01`), Precision (`0.68 +/- 0.03`), and the highest Recall among the models (`0.47 +/- 0.09`), indicating it can predict positive cases more effectively than some other models. However, the higher variability in Recall suggests its performance in identifying true positives might be less consistent. Its ROC-AUC (`0.71 +/- 0.04`) is among the highest, indicating a good balance between sensitivity and specificity.

        - `SVM` shows high Accuracy (`0.84 +/- 0.01`) and the highest Precision (`0.78 +/- 0.04`) but with the lowest Recall (`0.37 +/- 0.05`), indicating a significant risk of missing true positive cases. This could be problematic in scenarios where failing to detect positives is critical. Its ROC-AUC (`0.67 +/- 0.02`) is relatively high, suggesting it is better than random but not the best in distinguishing between classes.

        - The `ANN` model demonstrates the highest Accuracy (`0.85 +/- 0.01`) and Precision (`0.80 +/- 0.06`) among the models, with moderate Recall (`0.39 +/- 0.04`), which is a concern for missing positive cases. The model's F1 Score (`0.52 +/- 0.03`) and ROC-AUC (`0.68 +/- 0.02`) suggest it has a balanced performance but may not be the best choice where high sensitivity is required.

    
6. ### Conclusion
    - **Decision Tree** stands out as potentially the best model for medical fraud detection among the ones evaluated. It offers a good balance with a relatively high F1 Score (0.55), the highest ROC AUC (0.71), moderate recall (0.47) and precision (0.68). These characteristics are desirable for minimizing both false negatives and false positives in fraud detection.

    - **ANN and SVM** are also strong contenders due to their high precision, which is important for minimizing the investigation of false positives. However, their recall rates are lower, which could be a drawback in missing fraudulent cases

    - **Random Forest and LightGBM** provide good accuracy and a reasonable balance between precision and recall; their slightly lower F1 Scores and ROC AUCs compared to the Decision Tree model make them slightly less favourable for fraud detection in this context.


7. ### General Risks and Considerations
    - **Overfitting**: High precision with lower recall in models like SVM and ANN could indicate a risk of overfitting, meaning the model might not generalize well to unseen data.
    
    - **Underfitting**: The Logistic Regression model's poor performance across the board suggests it may be underfitting the data, unable to capture the underlying patterns effectively.

    - **Application Context**: The model choice should consider the application's specific context. For example, missing out on true positives (low recall) in medical diagnosis could be more detrimental than false positives (low precision).

    - **ROC-AUC Consideration**: While models like Decision Trees and SVM show relatively high ROC-AUC values, indicating better overall performance in distinguishing between classes, it's important to balance this with considerations of precision and recall based on the application's needs.
