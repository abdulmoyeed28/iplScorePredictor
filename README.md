
-----

### IPL Score Predictor

This is an end-to-end machine learning project that predicts the final score of an IPL match. The model is trained on historical data and deployed as a user-friendly web application using Streamlit. This project was developed as a key part of my internship, allowing me to apply my skills across the entire data science pipeline, from data cleaning to model deployment.

### Project Overview

The goal of this project is to provide a real-time score prediction for an ongoing IPL match. The model takes into account key features of the game at a given point, such as current runs, wickets, and overs, to estimate the final score with a high degree of accuracy. The final application is designed to be interactive and easy to use.

-----

### Core Concepts and Workflow

This project follows a standard machine learning workflow.

#### 1\. Data Collection and Cleaning

  * **Dataset**: The model is built using a rich dataset of historical IPL match data.
  * **Data Cleaning**: Raw data was prepared by removing irrelevant features that wouldn't contribute to the model's predictive power, such as `mid` (match ID) and `date`.
  * **Consistency**: To ensure the model is trained on a stable set of teams, I filtered the data to include only the 8 most consistent teams in the league.
  * **Filtering**: I removed the first 5 overs of every match, as the initial phase of an innings is highly variable and can introduce noise, impacting the model's ability to predict a final score.

#### 2\. Exploratory Data Analysis (EDA)

  * **Visualization**: I used **Seaborn** to create visualizations, including distribution plots for wickets and total runs, to understand the data's characteristics.
  * **Correlation Matrix**: A heatmap was generated to identify correlations between different numerical features, which helped validate feature importance.

#### 3\. Data Preprocessing and Encoding

  * **Label Encoding**: Categorical team names (`bat_team`, `bowl_team`) were converted into numerical labels using **Scikit-learn's `LabelEncoder`**.
  * **One-Hot Encoding**: To avoid the model misinterpreting the labels as having an ordinal relationship, I applied **One-Hot Encoding** using **`ColumnTransformer`**. This created new binary columns for each team, which is the correct way to handle nominal categorical data in machine learning models.

#### 4\. Model Building and Evaluation

  * **Algorithm Selection**: I tested and evaluated several regression algorithms, including **Linear Regression**, **Decision Tree**, **Random Forest**, **Support Vector Machine**, and **XGBoost**.
  * **Model Performance**: I used standard metrics like **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** to evaluate each model's accuracy. A bar plot was created to visually compare the performance of all models.
  * **Final Model**: The **Random Forest Regressor** demonstrated the highest accuracy and was chosen as the final model to be used for predictions. It's a robust algorithm that handles a mix of numerical and encoded categorical data well.

#### 5\. Model Export and Deployment

  * **Export**: The trained **Random Forest** model was saved as a binary file (`ml_model.pkl`) using **Python's `pickle` library**. This allows the model to be loaded quickly in the Streamlit application without needing to be retrained every time.
  * **Deployment**: The final application was built using **Streamlit**, a powerful framework for creating data apps with Python. The `app.py` script loads the pickled model and sets up an interactive user interface where users can input match parameters to get a score prediction.

-----

### How to Run the App Locally

To set up and run this project on your local machine, follow these simple steps.

#### Prerequisites

  * Python (3.7 or higher recommended)
  * `pip` (Python package manager)

#### Setup Instructions

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/abdulmoyeed28/ipl-score-predictor.git
    cd ipl-score-predictor
    ```

2.  **Create and Activate a Virtual Environment**: It's a best practice to use a virtual environment to manage project dependencies.

    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Required Libraries**:

    ```bash
    pip install -r requirements.txt
    ```

    (Note: If you don't have a `requirements.txt` file, you'll need to install the libraries manually: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, and `seaborn`).

4.  **Run the Streamlit App**:

    ```bash
    streamlit run app.py
    ```

Your default web browser should open automatically to display the IPL Score Predictor app. Enjoy\!
