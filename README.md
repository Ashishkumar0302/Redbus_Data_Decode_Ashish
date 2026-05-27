# 🚌 RedBus Seat Occupancy Forecasting 

**Predictive Modeling for Route-Level Seat Demand**

## 📌 Project Overview
This project builds an advanced machine learning pipeline to predict the `final_seatcount` for specific bus routes on any given Date of Journey (DOJ). By leveraging historical booking trajectories, search momentums, and regional characteristics, this solution enables proactive capacity planning and dynamic pricing strategies for transport aggregators.

## 📊 Dataset Description
The analysis and modeling pipeline utilizes two primary datasets:
1. **Train Dataset (`train.csv`):** Contains the ground truth target variables with 67,200 records. Key features include Date of Journey (`doj`), Source ID (`srcid`), Destination ID (`destid`), and `final_seatcount`.
2. **Transactions Dataset (`transactions.csv`):** A granular, time-series dataset with ~2.26 million records capturing the booking trajectory. Features include Date of Booking (`doi`), region and tier classifications for sources/destinations, cumulative seats booked, cumulative search counts, and Days Before Departure (`dbd`).

## 🛠️ Architecture & Data Pipeline
1. **Data Ingestion & EDA:** Exploratory Data Analysis to check data sparsity, distributions, and missing values. Verified a highly skewed seat count distribution indicating few extremely high-demand routes.
2. **Calendar & Holiday Integration:** Mapped regional holidays (e.g., Andhra Pradesh, Maharashtra and Goa, Delhi, Rajasthan, Kerala) to assess demand spikes natively.
3. **Advanced Feature Engineering:** * **Lag Features:** Captured daily transaction diffs (daily_seats, daily_searches) and shifted them across multiple lags (1, 2, 3, and 7 days).
   * **Rolling Window Statistics:** Calculated 7-day rolling averages, standard deviations, max values, and skewness for seat bookings.
   * **Momentum Indicators:** Computed ratio metrics (`momentum_3d_7d`) to capture sudden surges in booking or search velocity leading up to the DOJ.
   * **Target Encoding:** Mapped historical route averages and source averages to the baseline data.
4. **Ensemble Modeling:** Trained highly optimized LightGBM and XGBoost gradient boosting regressors. 

## ⚙️ Model Training & Blending
* **Cross-Validation Strategy:** Implemented a robust 5-Fold Cross Validation pipeline to ensure models generalize well across varying timeframes and routes.
* **Hyperparameter Tuning:** Fine-tuned `feature_fraction`, `bagging_fraction`, and L1/L2 regularizations (`lambda_l1`, `lambda_l2`) to avoid overfitting on the sparse transaction tails.
* **Model Ensembling:** Blended predictions from LightGBM and XGBoost to neutralize individual model biases, capturing both broad regional trends and sudden granular momentum shifts. 

## 📈 Output
The pipeline produces a final blended prediction file `submission_full_pipeline_ensemble.csv`, mapping a unique `route_key` (e.g., `2025-02-11_46_45`) to the forecasted `final_seatcount`.

## 🚀 How to Run the Project
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt` (Ensure you have `xgboost` and `lightgbm` installed).
3. Place `train.csv` and `transactions.csv` in your designated data directory and update the `file_path` variables in the Jupyter Notebook.
4. Run the cells sequentially to trigger preprocessing, K-Fold training, and final ensemble generation.

## 💻 Tech Stack
* **Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** LightGBM, XGBoost, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
