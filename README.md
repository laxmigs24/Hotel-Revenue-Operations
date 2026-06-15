# Hotel Booking Cancellation & Revenue Risk Analysis

## Project Goal

This project builds an analytics and machine learning framework to help hotel managers predict booking cancellations, identify high-risk customer and market segments, and support revenue-protection decisions.

Hotel cancellations create operational and financial uncertainty. They affect occupancy planning, revenue forecasting, staffing, inventory allocation, and pricing strategy. This project uses hotel booking data to understand cancellation behaviour and train machine learning models that estimate the probability of a booking being cancelled.

The final goal is to move from simple historical analysis to a decision-support system that can help hotel managers answer questions such as:

* Which bookings are most likely to cancel?
* Which customer or market segments create the highest revenue risk?
* What booking features increase cancellation probability?
* Which reservations should be monitored more closely?
* How can hotel teams prioritise intervention before revenue is lost?

Dataset: Hotel Booking Demand Dataset
Project Type: Data Science, Machine Learning, Revenue Operations
Author: Laxmi Gupte

---

## Business Context

Hotels receive bookings through different channels such as Online Travel Agencies, direct reservations, corporate clients, groups, and offline travel agents. These bookings do not all behave the same way.

Some bookings are more stable, while others have a higher chance of being cancelled. For example, a long lead-time booking from an Online Travel Agency with no deposit may carry more risk than a short lead-time direct booking with strong guest commitment.

Cancellation risk matters because hotels operate with limited room inventory. When a guest cancels late, the hotel may not have enough time to resell the room at the same price. This can lead to revenue loss, weaker occupancy, and less accurate operational planning.

This project treats cancellation prediction as a revenue operations problem, not only as a machine learning classification task.

---

## Problem Statement

The main business problem is:

Can we predict whether a hotel booking is likely to be cancelled using booking-level information, and use that prediction to support better revenue and operational decisions?

The project focuses on predicting the target variable:

`is_canceled`

Where:

* `0` means the booking was not cancelled
* `1` means the booking was cancelled

---

## Project Objectives

The project aims to:

1. Audit and clean the hotel booking dataset.
2. Explore cancellation patterns across hotel, customer, market, and booking features.
3. Engineer meaningful business features related to lead time, stay duration, revenue value, guest behaviour, and booking commitment.
4. Train and compare multiple classification models.
5. Evaluate model performance using relevant classification metrics.
6. Optimise the prediction threshold for business decision-making.
7. Explain model behaviour using feature importance and SHAP analysis.
8. Profile cancellation risk at segment level.
9. Estimate revenue exposure from predicted cancellations.
10. Save the best model for future deployment inside a Streamlit app.

---

## Repository Structure

The project is organised into the following main folders and files:

### app/

Contains the Streamlit application file.

* `streamlit_app.py` — Streamlit app for the hotel cancellation risk assistant.

### data/

Contains the raw, cleaned, and feature-engineered datasets.

* `hotel_bookings.csv` — original hotel booking dataset.
* `hotel_bookings_clean.csv` — cleaned dataset after the data audit stage.
* `hotel_bookings_feature_engineered.csv` — final dataset used for modelling.

### models/

Contains the saved machine learning model bundle.

* `xgboost_cancellation_model_bundle.joblib` — saved model bundle containing the trained XGBoost model, preprocessing pipeline, selected features, threshold, and related metadata.

### notebooks/

Contains the main project notebooks.

* `01_data_audit.ipynb` — data audit, quality checks, missing values, duplicates, and initial cleaning.
* `02_eda.ipynb` — exploratory data analysis and business pattern discovery.
* `03_feature_eng.ipynb` — feature engineering and creation of the modelling dataset.
* `04_modeling_evaluation.ipynb` — model training, evaluation, explainability, threshold optimisation, segment profiling, and model saving.

### README.md

Main project documentation.

---

## Notebook Workflow

The project is divided into four main notebooks.

---

## 01_data_audit.ipynb

This notebook focuses on understanding the raw dataset and preparing it for analysis.

Main tasks include:

* Loading the original hotel booking dataset.
* Inspecting the dataset shape.
* Reviewing column names and data types.
* Checking missing values.
* Checking duplicate records.
* Understanding numerical and categorical variables.
* Identifying potential data quality issues.
* Preparing a cleaned dataset for the next stage.

The purpose of this notebook is to make sure the dataset is reliable before moving into deeper analysis and modelling.

---

## 02_eda.ipynb

This notebook explores cancellation behaviour from a business perspective.

The analysis investigates how cancellations vary across different booking and customer attributes.

Key areas explored include:

* Overall cancellation distribution.
* Cancellation rate by hotel type.
* Cancellation patterns by market segment.
* Cancellation patterns by customer type.
* Lead time and cancellation behaviour.
* ADR and cancellation relationship.
* Deposit type and booking commitment.
* Special requests and guest engagement.
* Previous cancellations and repeat guest behaviour.
* Stay duration patterns.
* Arrival month and seasonal behaviour.
* Revenue-related booking patterns.

The purpose of this notebook is to identify useful business signals that can later be converted into model features.

---

## 03_feature_eng.ipynb

This notebook creates new features to improve model performance and make the dataset more useful for business decision-making.

The feature engineering process transforms raw booking data into more meaningful predictive signals.

### Stay Duration Features

These features describe how long the guest is expected to stay.

* `total_nights`
* `stay_length_category`
* `is_zero_night_stay`
* `is_weekend_stay`

### Lead Time Features

These features describe the time gap between booking date and arrival date.

* `lead_time_bucket`
* `is_last_minute_booking`
* `is_long_lead_booking`

### Revenue Features

These features estimate booking value and pricing behaviour.

* `estimated_booking_value`
* `adr_bucket`
* `is_zero_adr`
* `is_high_adr`
* `is_high_value_booking`

### Guest Behaviour Features

These features capture past guest behaviour and booking history.

* `previous_total_bookings`
* `previous_cancellation_rate`
* `has_previous_cancellation`
* `has_previous_successful_booking`

### Booking Commitment Features

These features capture signs of guest engagement or uncertainty.

* `has_booking_changes`
* `has_special_requests`
* `is_waitlisted`

### Date and Seasonality Features

These features help capture time-based booking patterns.

* `arrival_month_number`
* `arrival_season`
* `is_peak_season`

### Room Type Feature

This feature checks whether the reserved room type and assigned room type are different.

* `room_type_changed`

The output of this notebook is saved as:

`data/hotel_bookings_feature_engineered.csv`

---

## 04_modeling_evaluation.ipynb

This notebook builds and evaluates the machine learning models.

The modelling workflow includes:

1. Define project objective.
2. Load the feature-engineered dataset.
3. Define the target variable and candidate features.
4. Review and remove leakage-prone columns.
5. Split the data into training and test sets.
6. Build a preprocessing pipeline.
7. Train a baseline Logistic Regression model.
8. Train tree-based models:

   * Random Forest
   * XGBoost
   * LightGBM
9. Compare model performance.
10. Evaluate models using:

* Confusion matrix
* ROC-AUC
* Precision
* Recall
* F1-score
* Precision-Recall curve

11. Optimise the decision threshold.
12. Explain model behaviour using:

* Feature importance
* SHAP values

13. Perform segment-level risk profiling.
14. Translate results into a business narrative.
15. Save the best model.

---

## Leakage Review

Before training the models, the project reviews features that may leak information from the future.

Some columns are not suitable for prediction because they reveal information that would only be known after the booking outcome is already decided.

Examples of leakage-prone columns include:

* `reservation_status`
* `reservation_status_date`

These columns are excluded from modelling because they can artificially inflate model performance and make the model unrealistic for real-world prediction.

The goal is to only use information that would reasonably be available before the cancellation outcome is known.

---

## Preprocessing Pipeline

The project uses a preprocessing pipeline to prepare different types of features for modelling.

The pipeline handles:

* Numerical features
* Nominal categorical features
* Ordinal categorical features
* Binary engineered features
* Missing values

### Numerical Features

Numerical features are processed so they can be used effectively by machine learning models.

Examples include:

* `lead_time`
* `adr`
* `total_nights`
* `previous_cancellation_rate`
* `estimated_booking_value`
* `booking_changes`
* `total_of_special_requests`

### Categorical Features

Categorical features are encoded so that machine learning models can use them.

Examples include:

* `hotel`
* `market_segment`
* `distribution_channel`
* `deposit_type`
* `customer_type`
* `lead_time_bucket`
* `adr_bucket`
* `arrival_season`

### Ordinal Features

Some categorical features have a meaningful order.

Example:

* `stay_length_category`

### Binary Features

Binary engineered features are passed into the model as 0/1 indicators.

Examples include:

* `is_repeated_guest`
* `is_last_minute_booking`
* `is_long_lead_booking`
* `is_weekend_stay`
* `has_special_requests`
* `room_type_changed`
* `is_high_value_booking`

Missing values are handled inside the preprocessing pipeline to reduce the risk of data leakage.

---

## Models Trained

The project compares multiple classification models.

---

## Logistic Regression

Logistic Regression is used as the baseline model.

It provides a simple and interpretable benchmark before testing more complex tree-based models.

The Logistic Regression model is useful because it helps establish whether more advanced models are actually improving performance.

---

## Random Forest

Random Forest is used as a tree-based ensemble model.

It is able to capture non-linear relationships and interactions between booking features.

Random Forest is useful in this project because cancellation behaviour is unlikely to be explained by one single variable. Instead, it depends on combinations of features such as lead time, deposit type, customer type, and market segment.

---

## XGBoost

XGBoost is used as the main gradient boosting model.

It is well suited for structured tabular data and often performs strongly on classification problems.

XGBoost is selected as the final saved model because it provides a strong balance of:

* Predictive performance
* Compatibility with preprocessing
* Feature importance analysis
* SHAP explainability
* Business usability

---

## LightGBM

LightGBM is also trained for comparison.

It is another gradient boosting model designed for efficient training and strong performance on tabular datasets.

The project compares LightGBM with XGBoost and Random Forest to understand which model performs best for this cancellation prediction task.

---

## Model Evaluation Strategy

The project does not rely only on accuracy because cancellation prediction is an imbalanced classification problem.

A model can appear accurate by mostly predicting the majority class, but that may not be useful for hotel revenue decisions.

The evaluation uses:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Precision-Recall curve

---

## Business Meaning of Evaluation Metrics

### Precision

Precision answers:

When the model predicts that a booking will cancel, how often is it correct?

High precision means fewer unnecessary interventions.

### Recall

Recall answers:

Out of all bookings that actually cancel, how many did the model successfully detect?

High recall means fewer missed cancellation risks.

### F1-score

F1-score balances precision and recall.

It is useful when both false positives and false negatives matter.

### ROC-AUC

ROC-AUC measures how well the model separates cancelled bookings from non-cancelled bookings across different thresholds.

### Confusion Matrix

The confusion matrix shows the model’s prediction outcomes:

* True negatives: correctly predicted non-cancellations
* True positives: correctly predicted cancellations
* False positives: predicted cancellation, but booking was not cancelled
* False negatives: predicted no cancellation, but booking was cancelled

---

## Business Cost of Prediction Errors

### False Negatives

A false negative means the model predicts that a booking will not cancel, but it actually cancels.

Business impact:

* Missed opportunity to intervene
* Possible unsold room inventory
* Lower occupancy accuracy
* Weaker revenue forecasting
* Higher risk of preventable revenue loss

### False Positives

A false positive means the model predicts that a booking may cancel, but it does not cancel.

Business impact:

* Unnecessary staff attention
* Possible over-contacting of guests
* Less efficient operations
* Potentially unnecessary revenue-management action

The best threshold depends on the hotel’s business strategy.

A hotel that wants to catch more risky bookings may prefer higher recall. A hotel that wants fewer unnecessary interventions may prefer higher precision.

---

## Threshold Optimisation

The default classification threshold of 0.50 is not always ideal for business use.

This project tests different probability thresholds and compares model performance across:

* Accuracy
* Precision
* Recall
* F1-score

The selected threshold is based on balancing model performance and business usefulness.

The final threshold used in the modelling workflow is:

`0.55`

This means:

If the predicted cancellation probability is greater than or equal to `0.55`, the booking is classified as likely to cancel.

This threshold can be adjusted in future depending on the hotel’s operational priorities.

---

## Model Comparison

The project compares the baseline model against tree-based models.

Overall modelling observations:

* Logistic Regression provides a useful baseline.
* Random Forest improves the ability to capture non-linear patterns.
* XGBoost performs strongly and is selected as the final saved model.
* LightGBM provides a strong additional comparison point.

The final model selected for deployment preparation is:

`XGBoost`

---

## Explainability

Explainability is important because hotel managers need to understand why a booking is considered risky.

The project uses two explainability approaches:

* XGBoost feature importance
* SHAP values

---

## Feature Importance

Feature importance helps identify which features contribute most to the model’s predictions.

Important model drivers include features related to:

* Lead time
* Market segment
* Deposit type
* Customer type
* Room type changes
* Special requests
* Previous cancellation behaviour
* ADR
* Booking changes
* Required parking spaces

These features help connect the technical model output to business reasoning.

---

## SHAP Analysis

SHAP values provide more detailed model explainability.

They help explain:

* Which features push a prediction toward cancellation
* Which features reduce cancellation risk
* How individual feature values affect predictions
* Which features matter most across the dataset

SHAP analysis makes the model more transparent and more suitable for business presentation.

---

## Segment-Level Risk Profiling

After model training, the project analyses cancellation risk at segment level.

This helps move beyond individual booking prediction and supports broader revenue-management decisions.

Segment-level profiling includes:

### Market Segment Risk

The project compares risk across market segments such as:

* Online TA
* Offline TA/TO
* Direct
* Groups
* Corporate
* Aviation
* Complementary

### Customer Type Risk

The project compares cancellation risk across customer types such as:

* Transient
* Transient-Party
* Contract
* Group

### Hotel Type Risk

The project compares risk between:

* City Hotel
* Resort Hotel

---

## Revenue at Risk

The project estimates revenue exposure using predicted cancellation probability.

The calculation is:

`estimated revenue at risk = estimated booking value × predicted cancellation probability`

This allows hotel teams to prioritise bookings and segments not only by cancellation probability, but also by commercial importance.

A high-risk low-value booking may not require the same level of attention as a moderate-risk high-value booking.

---

## Key Business Insights

The project shows that cancellation risk is influenced by a combination of booking, customer, and revenue factors.

Important business insights include:

* Lead time is a strong signal in cancellation behaviour.
* Online Travel Agency bookings often need closer monitoring.
* Deposit type can influence guest commitment.
* Previous cancellation behaviour is an important risk indicator.
* Special requests may indicate stronger guest engagement.
* Customer type affects cancellation behaviour.
* Room type changes can influence prediction patterns.
* Segment-level revenue exposure is more useful than cancellation count alone.
* Revenue teams should prioritise bookings based on both risk probability and booking value.

---

## Saved Model

The final model is saved as a joblib bundle:

`models/xgboost_cancellation_model_bundle.joblib`

The saved bundle is designed to include:

* Trained XGBoost model
* Preprocessing transformer
* Selected model features
* Decision threshold
* Model metadata

This makes the model reusable for future deployment inside an application.

---

## Streamlit App

A Streamlit app is included in the repository as the application layer of the project.

The app is intended to support:

* Booking cancellation risk prediction
* Risk explanation for hotel managers
* Batch booking risk review
* Segment-level risk analysis
* Revenue-at-risk interpretation

The app section can be expanded further as the interface and functionality are finalised.

---

## How to Run the App

From the project root folder, run:

`python -m streamlit run app/streamlit_app.py`

If using a Conda environment, activate the environment first:

`conda activate hotel-ai`

Then run:

`python -m streamlit run app/streamlit_app.py`

The app should open locally at:

`http://localhost:8501`

---

## Technologies Used

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* SHAP
* Joblib
* Jupyter Notebook
* Streamlit

---

## Environment Setup

A clean Conda environment is recommended.

Create the environment:

`conda create -n hotel-ai python=3.12 -y`

Activate the environment:

`conda activate hotel-ai`

Install the main dependencies:

`conda install -c conda-forge numpy pandas scikit-learn joblib matplotlib streamlit py-xgboost llvm-openmp ipykernel -y`

Optional packages:

`pip install shap lightgbm`

If an environment file is available, use:

`conda env create -f environment.yml`

Then activate it:

`conda activate hotel-ai`

---

## How to Run the Notebooks

Open the project in VS Code or Jupyter Notebook and run the notebooks in order:

1. `01_data_audit.ipynb`
2. `02_eda.ipynb`
3. `03_feature_eng.ipynb`
4. `04_modeling_evaluation.ipynb`

Recommended workflow:

1. Run the data audit notebook.
2. Run the EDA notebook.
3. Run the feature engineering notebook.
4. Run the modelling and evaluation notebook.
5. Save the final model bundle.
6. Run the Streamlit app.

---

## Project Outputs

This project produces:

* Cleaned hotel booking dataset
* Feature-engineered modelling dataset
* Exploratory data analysis
* Cancellation pattern insights
* Machine learning preprocessing pipeline
* Logistic Regression baseline model
* Random Forest model
* XGBoost model
* LightGBM model
* Model comparison table
* Confusion matrices
* ROC-AUC evaluation
* Precision, recall, and F1-score evaluation
* Precision-Recall curve
* Threshold optimisation
* Feature importance analysis
* SHAP explainability
* Segment-level risk profiling
* Revenue-at-risk analysis
* Saved XGBoost model bundle
* Streamlit app foundation

---

## Business Value

This project demonstrates how machine learning can support hotel revenue operations.

Potential business applications include:

* Identifying bookings likely to cancel
* Prioritising reconfirmation messages
* Supporting revenue-management decisions
* Improving occupancy forecasting
* Monitoring high-risk booking channels
* Understanding segment-level exposure
* Estimating potential revenue at risk
* Supporting overbooking and inventory strategy
* Helping hotel managers act earlier and more systematically

The model is not designed to replace human judgement. It is designed to support better prioritisation and decision-making.

---

## Limitations

This project uses historical booking data, so predictions should be interpreted carefully.

Important limitations include:

* Predictions are based on past patterns and are not guaranteed outcomes.
* External events such as weather, airline disruption, local events, and economic conditions are not included.
* The model should not be used as the only basis for guest treatment or commercial decisions.
* Dataset patterns may not fully generalise to every hotel, region, or time period.
* Revenue at risk is estimated, not actual final lost revenue.
* Model performance should be validated before real operational use.
* Business thresholds may need to be adjusted depending on hotel strategy.

---

## Future Improvements

Possible future improvements include:

* Improve the Streamlit interface and user experience.
* Add manual new-booking prediction.
* Add batch CSV upload for hotel managers.
* Add individual booking SHAP explanations.
* Add cost-sensitive threshold optimisation.
* Add expected revenue-loss simulation.
* Add overbooking recommendation logic.
* Add model monitoring.
* Add API deployment.
* Add cloud deployment.
* Add automated retraining pipeline.
* Add dashboard screenshots to the README.
* Add a formal business case study section.
* Add scenario analysis for different revenue-management strategies.

---

## Project Status

Current status:

* Data audit completed
* Exploratory data analysis completed
* Feature engineering completed
* Model training completed
* Model evaluation completed
* Threshold optimisation completed
* Explainability completed
* Segment profiling completed
* Best model saved
* Streamlit app foundation included

---

## Author

Laxmi Gupte

Data Science and Machine Learning project focused on hotel revenue operations, cancellation prediction, explainable AI, and business decision support.

---

## License

This project is intended for educational and portfolio purposes.
