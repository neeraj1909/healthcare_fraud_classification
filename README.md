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
        - Decision Tree
        - SVM
        - Artificial Neural Network (ANN)

5. **Model Performance Analysis**
    - Model's Performance
        - For the `Logistic Regression` model, the Accuracy is (`0.67 +/- 0.22`), Precision is (`0.08 +/- 0.10`), Recall is (`0.20 +/-  0.40`), `F1-Score` is (`0.08 +/-  0.15`), and `ROC-AUC` is (`0.49 +/-  0.01`).

        - For the `Random Forest` model, the Accuracy is (`0.84 +/-  0.02`), Precision is (`0.71 +/-  0.10`), Recall is (`0.43 +/-  0.08`), `F1-Score` is (`0.53 +/-  0.08`), and `ROC-AUC` is (`0.69 +/-  0.04`).

        - For the `Decision Tree` model, the Accuracy is (`0.89 +/-  0.01`), Precision is (`0.90 +/-  0.02`), Recall is (`0.52 +/-  0.06`), `F1-Score` is (`0.66 +/-  0.05`), and `ROC-AUC` is (`0.75 +/-  0.03`).

        - For the `SVM` model, the Accuracy is (`0.89 +/-  0.01`), Precision is (`0.90 +/-  0.02`), Recall is (`0.52 +/-  0.06`), `F1-Score` is (`0.66 +/-  0.05`), and `ROC-AUC` is (`0.75 +/-  0.03`).

        - For the `ANN` model, the Accuracy is (`0.85 +/-  0.02`), Precision is (`0.83 +/-  0.10`), Recall is (`0.38 +/-  0.09`), `F1-Score` is (`0.51 +/-  0.08`), and `ROC-AUC` is (`0.68 +/-  0.04`).
    

    - For Prediction on the test data, we used the `Decision Tree` model.

    - ### Performance Risks
        - The `Logistic Regression` model shows low performance across all metrics with high variability, particularly in Recall (`0.20 +/-  0.40`) and Precision (`0.08 +/- 0.10`). This indicates a high risk of both false positives and false negatives, making it unreliable for critical decision-making processes. The ROC-AUC score (`0.49 +/-  0.01`) is close to 0.5, suggesting no better than random guessing, highlighting a significant risk in its predictive capability.

        - The `Random Forest` model demonstrates a good balance between accuracy (`0.84 +/-  0.02`) and precision (`0.71 +/-  0.10`), with moderate recall (`0.43 +/-  0.08`). The variability in its performance metrics is relatively low, indicating consistent performance. However, the recall suggests that it might miss a significant number of positive cases, posing a risk for applications where missing out on true positives is critical.

        - `Decision Tree` and `SVM`, both show identical performance metrics, with high accuracy (`0.89 +/-  0.01`), precision (`0.90 +/-  0.02`), and moderate recall (`0.52 +/-  0.06`). These models offer a strong predictive capability, as evidenced by their ROC-AUC scores (`0.75 +/-  0.03`), indicating a good balance between sensitivity and specificity. However, the moderate recall indicates that while they are highly precise, they may still miss a fair number of true positives, which could be a risk in certain scenarios.

        - The ANN model has a good accuracy (`0.85 +/-  0.02`) and high precision (`0.83 +/-  0.10`), but its recall (`0.38 +/-  0.09`) is the lowest among the models discussed. This indicates a significant risk in failing to identify a large portion of actual positive cases. The model's consistency, as shown by its relatively stable metrics, suggests reliability in its precision but raises concerns about its sensitivity to positive cases.


    - ### General Risks and Considerations
        - **Overfitting**: High precision with lower recall in models like Decision Tree, SVM, and ANN could indicate a risk of overfitting.
        
        - **Underfitting**: The Logistic Regression model's poor performance across the board suggests it may be underfitting the data, unable to capture the underlying patterns effectively.

        - **Application Context**: The choice of model should consider the specific context of the application. For example, in medical diagnosis, missing out on true positives (low recall) could be more detrimental than false positives (low precision).

        - **ROC-AUC**: Higher ROC-AUC values in Decision Tree, and SVM indicate better overall performance in distinguishing between classes.

