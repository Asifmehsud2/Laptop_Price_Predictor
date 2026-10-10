# Laptop Price Predictor

A machine learning project that estimates laptop prices from hardware and software specifications through an interactive Streamlit application.

## Features

- Select laptop specifications such as brand, laptop type, display, processor, RAM, storage, graphics, operating system, and weight.
- Generate a price estimate using a saved machine learning model.
- Keep the user interface separate from the model pipeline.
- Explore the data and model development in Jupyter Notebook.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost (if used by the saved model)
- Streamlit
- Joblib
- Jupyter Notebook

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit application and input form |
| `pipeline.py` | Data preprocessing and model pipeline code |
| `model.joblib` | Saved trained model or pipeline |
| `laptop_data.csv` | Dataset used for exploration/training, if included |
| `requirements.txt` | Python package dependencies |
| `*.ipynb` | Model development and experiments |

## Run Locally

### 1. Clone this repository

```bash
git clone https://github.com/Asifmehsud2/Laptop_Price_Predictor.git
cd Laptop_Price_Predictor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser.

## How It Works

1. Choose laptop specifications in the app.
2. The application prepares the input in the format expected by the saved model.
3. The model returns an estimated price.
4. The result is shown in the Streamlit interface.

## Notes

- Predictions are estimates and may differ from actual market prices.
- Keep the feature names, preprocessing, and input order consistent with the model used to create `model.joblib`.
- Include the dataset only if its license permits redistribution.
- Do not commit API keys, passwords, access tokens, or local environment files.

## Author

**M. Asif Khan**

- GitHub: [@Asifmehsud2](https://github.com/Asifmehsud2)
- LinkedIn: [@AsifMehsud]((https://www.linkedin.com/in/asif-mehsud/))

---

If you improve the model or interface, document the changes and update the screenshots so users can see the current version.
