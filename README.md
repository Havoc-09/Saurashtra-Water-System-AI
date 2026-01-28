# AI for Drought Resilience & Water Security (Saurashtra)

A comprehensive AI-powered dashboard for monitoring drought risk, forecasting water availability, and analyzing demand-supply gaps in the Saurashtra region.

## 🌟 Features

*   **Regional Monitoring**: Interactive dashboard for 8 districts in Saurashtra (Rajkot, Jamnagar, etc.).
*   **Drought Risk Prediction**: Random Forest model classifying risk into Safe, Warning, or Critical based on real-time parameters.
*   **Water Availability Forecasting**: Predictive modeling for future water gaps using historical trends.
*   **Gap Analysis**: Real-time visualization of Supply vs. Demand.
*   **Explainable AI**: Interpretation of why specific risk levels were predicted (e.g., impact of reservoir levels vs. rainfall).
*   **Geospatial Risk Map**: Visual heatmap of drought stress across the region.

## 🛠️ Installation

1.  **Prerequisites**: Python 3.8+ installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

*   `app.py`: Main application file containing:
    *   Synthetic Data Generation (Realistic simulation for Saurashtra)
    *   AI Model Training (Random Forest)
    *   Streamlit Dashboard UI
*   `requirements.txt`: List of Python libraries required.

## 🚀 How to Use

1.  Launch the app directly in your browser or mobile device (Streamlit is responsive).
2.  Use the **Sidebar** to select a specific district.
3.  Navigate through the **Tabs**:
    *   **Overview**: Key metrics and historical trends.
    *   **Forecast**: Short-term water availability predictions.
    *   **Explainable AI**: Understand the factors driving the drought risk.
    *   **Risk Map**: See the bigger picture across the entire region.

## 🧠 AI Methodology

*   **Risk Classification**: Uses rainfall averages, temperature, groundwater levels, and reservoir percentages to classify drought risk.
*   **Forecasting**: Uses lag-based features (past rainfall) to predict future water gaps.
*   **Synthetic Data**: Since real-time API data requires paid subscriptions, this project generates realistic patterns based on Saurashtra's climate profile (Monsoon seasonality, groundwater depletion curves).
