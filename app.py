
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model.joblib: {e}")
    st.stop()



def prepare_features(raw_df):
    df = raw_df.copy()

    df["Ram"] = (
        df["Ram"].astype(str)
        .str.upper()
        .str.replace("GB", "", regex=False)
        .str.strip()
        .astype(int)
    )

    df["Weight"] = (
        df["Weight"].astype(str)
        .str.lower()
        .str.replace("kg", "", regex=False)
        .str.strip()
        .astype(float)
    )

    screen = df["ScreenResolution"].astype(str)

    df["IPs"] = screen.str.contains(
        "IPS", case=False, regex=False
    ).astype(int)

    df["TouchScreen"] = screen.str.contains(
        "Touch", case=False, regex=False
    ).astype(int)

    resolution = screen.str.extract(
        r"(\d{3,4})\s*x\s*(\d{3,4})"
    )

    df["X_res"] = pd.to_numeric(
        resolution[0], errors="coerce"
    )

    df["Y_res"] = pd.to_numeric(
        resolution[1], errors="coerce"
    )

    cpu = df["Cpu"].astype(str)

    df["CPU_Brand"] = cpu.str.split().str[0]

    clock_speed = cpu.str.extract(
        r"(\d+(?:\.\d+)?)\s*GHz",
        flags=re.IGNORECASE
    )[0]

    df["ClockSpeed"] = pd.to_numeric(
        clock_speed, errors="coerce"
    )

    df["GPU_Brand"] = (
        df["Gpu"].astype(str).str.split().str[0]
    )

    memory = df["Memory"].astype(str).str.upper()

    for storage_type in ["SSD", "HDD", "Flash", "Hybrid"]:
        df[storage_type] = memory.str.contains(
            storage_type, regex=False
        ).astype(int)

    def total_storage(value):
        capacities = re.findall(
            r"(\d+(?:\.\d+)?)\s*(TB|GB)",
            value.upper()
        )

        total = 0.0

        for amount, unit in capacities:
            amount = float(amount)

            if unit == "TB":
                amount *= 1000

            total += amount

        return total

    df["Total_Storage"] = memory.apply(total_storage)

    df.drop(
        columns=[
            "ScreenResolution",
            "Cpu",
            "Gpu",
            "Memory"
        ],
        inplace=True
    )

    df = pd.get_dummies(
        df,
        columns=[
            "Company",
            "TypeName",
            "OpSys",
            "CPU_Brand",
            "GPU_Brand"
        ],
        drop_first=True
    )

    df = df.replace({True: 1, False: 0})

    return df


st.title("💻 Laptop Price Predictor")

st.write(
    "Select the laptop specifications below "
    "and click Predict Price."
)

st.divider()

with st.form("laptop_form"):

    st.subheader("Laptop Specifications")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Company",
            [
                "HP", "Dell", "Lenovo", "Asus",
                "Acer", "Apple", "MSI", "Toshiba",
                "Samsung", "Other"
            ]
        )

        type_name = st.selectbox(
            "Laptop Type",
            [
                "Notebook", "Ultrabook", "Gaming",
                "2 in 1 Convertible", "Workstation",
                "Netbook"
            ]
        )

        inches = st.selectbox(
            "Screen Size (inches)",
            [11.6, 12.5, 13.3, 14.0, 15.6, 17.3]
        )

        screen_res = st.selectbox(
            "Screen Resolution",
            [
                "Full HD 1920x1080",
                "1366x768",
                "1600x900",
                "2560x1440",
                "3840x2160"
            ]
        )

        cpu = st.selectbox(
            "CPU",
            [
                "Intel Core i3 2.4GHz",
                "Intel Core i5 2.5GHz",
                "Intel Core i7 2.8GHz",
                "Intel Celeron 1.6GHz",
                "AMD Ryzen 5 3.0GHz",
                "AMD Ryzen 7 3.2GHz"
            ]
        )

    with col2:
        ram = st.selectbox(
            "RAM",
            ["4GB", "8GB", "16GB", "32GB", "64GB"]
        )

        memory = st.selectbox(
            "Storage",
            [
                "128GB SSD",
                "256GB SSD",
                "512GB SSD",
                "1TB SSD",
                "1TB HDD",
                "1TB HDD + 256GB SSD",
                "2TB HDD"
            ]
        )

        gpu = st.selectbox(
            "GPU",
            [
                "Intel HD Graphics 620",
                "Intel UHD Graphics",
                "Intel Iris Xe Graphics",
                "Nvidia GeForce GTX 1050",
                "Nvidia GeForce GTX 1650",
                "AMD Radeon"
            ]
        )

        opsys = st.selectbox(
            "Operating System",
            [
                "Windows 10",
                "Windows 11",
                "Windows 7",
                "No OS",
                "Linux",
                "macOS"
            ]
        )

        weight = st.selectbox(
            "Weight",
            ["1.2kg", "1.5kg", "1.86kg", "2.0kg", "2.5kg", "3.0kg"]
        )

    submitted = st.form_submit_button(
        "Predict Price",
        type="primary",
        use_container_width=True
    )


if submitted:
    raw_df = pd.DataFrame([{
        "Company": company,
        "TypeName": type_name,
        "Inches": float(inches),
        "ScreenResolution": screen_res,
        "Cpu": cpu,
        "Ram": ram,
        "Memory": memory,
        "Gpu": gpu,
        "OpSys": opsys,
        "Weight": weight
    }])

    try:
        expected_columns = getattr(
            model, "feature_names_in_", None
        )

        raw_columns = list(raw_df.columns)

        if (
            expected_columns is not None
            and set(expected_columns).issubset(raw_columns)
        ):
            model_input = raw_df[list(expected_columns)]

        else:
            model_input = prepare_features(raw_df)

            if expected_columns is not None:
                model_input = model_input.reindex(
                    columns=list(expected_columns),
                    fill_value=0
                )
            else:
                raise ValueError(
                    "The model does not expose feature_names_in_. "
                    "Save the exact training feature columns or "
                    "save the complete fitted preprocessing pipeline."
                )

        prediction = np.asarray(
            model.predict(model_input)
        ).reshape(-1)[0]

        if not np.isfinite(prediction):
            raise ValueError("The model returned an invalid prediction.")

        st.divider()
        st.subheader("Predicted Laptop Price")

        st.success(f"Rs. {prediction:,.0f}")

        st.caption(
            "This is an estimated price generated by your trained model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.info(
            "Check that the feature engineering, category names, "
            "and training columns match your notebook exactly."
        )


st.divider()

st.caption("Laptop Price Predictor | By M Asif Khan")
