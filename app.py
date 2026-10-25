"""General-purpose CSV explorer with charts and two-feature linear regression."""

import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CSV Data Explorer", layout="wide")
st.title("CSV Data Explorer")
st.write("Upload a CSV containing at least three numeric columns.")

# Interpret familiar numeric formats without using dataset-specific column names.
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def parse_number(value):
    """Turn ordinary numeric text into a number; return NaN for invalid text."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if text.endswith("%"):
        text = text[:-1].strip()  # 25% is represented as 25.
    if text and unicodedata.category(text[0]) == "Sc":
        text = text[1:].strip()
    if text and unicodedata.category(text[-1]) == "Sc":
        text = text[:-1].strip()
    # Interpret commas as thousands separators only when grouping is valid.
    if "," in text:
        grouped = re.fullmatch(
            r"([+-]?\d{1,3})(,\d{3})+(\.\d*)?([eE][+-]?\d+)?", text
        )
        if grouped is None:
            return np.nan
        text = text.replace(",", "")
    if NUMBER_PATTERN.fullmatch(text) is None:
        return np.nan
    number = pd.to_numeric(text, errors="coerce")
    return float(number) if pd.notna(number) and np.isfinite(number) else np.nan


def clean_data(raw_df):
    """Detect numeric columns; interpret invalid entries as missing values.

    A text column is treated as numeric if at least half of its nonblank
    entries look numeric. This is a heuristic, not a universal type detector.
    """
    data = raw_df.copy()
    converted = []
    for column in data.columns:
        series = data[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            data[column] = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            continue
        nonblank = series.notna() & series.astype("string").str.strip().ne("")
        if not nonblank.any():
            continue
        parsed = series.map(parse_number).astype(float)
        if parsed.notna().sum() / nonblank.sum() >= 0.5:
            data[column] = parsed
            converted.append(str(column))
    numeric_columns = [
        col for col in data.select_dtypes(include="number").columns
        if data[col].notna().any()
    ]
    return data, numeric_columns, converted


# 1. Upload the dataset and preview its contents.
file = st.file_uploader("Upload CSV", type="csv")
if file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(file)
    df, numeric_columns, converted_columns = clean_data(raw_df)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, ValueError) as error:
    st.error(f"Could not read this CSV: {error}")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(10), hide_index=True)
if converted_columns:
    st.caption("Converted numeric-looking text in: " + ", ".join(converted_columns))

# This is the sole dataset-eligibility requirement.
if len(numeric_columns) < 3:
    st.error("Please upload a CSV with at least three usable numeric columns.")
    st.stop()

# 2. Visualize the data (drop missing values only for columns used in the chart).
st.subheader("Data Visualization")
chosen_column = st.selectbox("Choose a numeric column", numeric_columns)
chart_type = st.selectbox("Choose a chart type", ["Histogram", "Scatter", "Line"])

if chart_type == "Histogram":
    fig = px.histogram(
        df.dropna(subset=[chosen_column]),
        x=chosen_column,
        title=f"Distribution of {chosen_column}",
    )
elif chart_type == "Scatter":
    x_column = st.selectbox(
        "Choose an X-axis column", [c for c in numeric_columns if c != chosen_column]
    )
    chart_data = df.dropna(subset=[x_column, chosen_column])
    fig = px.scatter(
        chart_data, x=x_column, y=chosen_column,
        title=f"{chosen_column} vs. {x_column}",
    )
else:
    # Without a date column, a line chart follows the original row order.
    chart_data = pd.DataFrame({
        "Row": np.arange(1, len(df) + 1),
        chosen_column: df[chosen_column].to_numpy(),
    })
    fig = px.line(chart_data, x="Row", y=chosen_column,
                  title=f"{chosen_column} by CSV Row")
    st.caption("A row number is not necessarily a date or time.")

st.plotly_chart(fig, use_container_width=True)

# 3. Choose any two distinct numeric inputs and a different numeric target.
st.subheader("Linear Regression")
target = st.selectbox("Choose the target", numeric_columns)
feature_options = [c for c in numeric_columns if c != target]
feature1 = st.selectbox("Choose input 1", feature_options)
feature2 = st.selectbox("Choose input 2", [c for c in feature_options if c != feature1])

selected = [feature1, feature2, target]
model_df = df[selected].dropna()
removed_rows = len(df) - len(model_df)
st.caption(
    f"Cleaning: dropped {removed_rows:,} row(s) with missing or invalid "
    f"values in the three selected columns; {len(model_df):,} rows remain."
)

# A minimum number of complete rows is necessary to train/test a model;
# this is not an extra CSV validation rule.
if len(model_df) < 5:
    st.warning("Not enough complete rows for training and evaluation. Choose other columns or another CSV.")
    st.stop()

X = model_df[[feature1, feature2]]
y = model_df[target]
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions) if y_test.nunique() > 1 else np.nan
except (ValueError, OverflowError, FloatingPointError) as error:
    st.error(f"The selected columns could not be used for regression: {error}")
    st.stop()

st.write(f"**Test MAE:** {mae:,.3f}")
st.write(f"**Test R²:** {r2:.3f}" if np.isfinite(r2) else "**Test R²:** Not available")

# 4. Let the user enter any finite numeric values; no arbitrary min/max constraints.
st.write("**Enter predictor values:**")
left, right = st.columns(2)
with left:
    value1 = st.number_input(
        f"Enter {feature1}", value=float(X[feature1].median()),
        key=f"feature1_{feature1}",
    )
with right:
    value2 = st.number_input(
        f"Enter {feature2}", value=float(X[feature2].median()),
        key=f"feature2_{feature2}",
    )

# The feature names and order must match those used in model.fit().
new_row = pd.DataFrame([[value1, value2]], columns=[feature1, feature2])
try:
    predicted_value = model.predict(new_row)[0]
    if np.isfinite(predicted_value):
        st.metric(f"Predicted {target}", f"{predicted_value:,.3f}")
    else:
        st.error("The model could not produce a finite prediction.")
except (ValueError, OverflowError, FloatingPointError) as error:
    st.error(f"Could not calculate a prediction: {error}")

# 5. Assignment-required model purpose and consequences statement.
st.write(
    "This linear regression predicts one numeric variable using two others; "
    "incorrect predictions may lead to misleading conclusions or poor decisions."
)
