import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Amazon Product Explorer", layout="wide")
st.title("Amazon Product Explorer")

# Amazon columns need their symbols removed before pandas can treat them as numbers.
AMAZON_NUMERIC_COLS = [
    "discount_percentage", "rating", "rating_count",
    "actual_price", "discounted_price"
]


def clean_data(raw_df):
    """Convert known Amazon columns and treat invalid numeric data as missing."""
    cleaned = raw_df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    if cleaned.columns.duplicated().any():
        raise ValueError("The CSV has duplicate column names.")

    invalid_counts = {}
    for col in AMAZON_NUMERIC_COLS:
        if col not in cleaned.columns:
            continue

        original = cleaned[col]
        has_value = original.notna() & original.astype(str).str.strip().ne("")
        converted = pd.to_numeric(
            original.astype(str).str.replace(r"[₹,%]", "", regex=True).str.strip(),
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        # Apply the actual domain of each Amazon variable.
        if col == "rating":
            bad_range = converted.notna() & ~converted.between(1, 5)
        elif col == "discount_percentage":
            bad_range = converted.notna() & ~converted.between(0, 100)
        elif col == "rating_count":
            bad_range = converted.notna() & ((converted < 0) | (converted % 1 != 0))
        else:  # prices cannot be negative
            bad_range = converted.notna() & (converted < 0)

        invalid_counts[col] = int((has_value & converted.isna()).sum() + bad_range.sum())
        cleaned[col] = converted.mask(bad_range)

    # Also exclude infinities from any other (already numeric) CSV fields.
    numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
    cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return cleaned, {k: v for k, v in invalid_counts.items() if v > 0}


def get_user_input(feature, training_values):
    """Select valid widget limits based on the chosen predictor."""
    median = float(training_values.median())
    label = f"Enter {feature}"

    if feature == "rating":
        return st.number_input(
            label, min_value=1.0, max_value=5.0,
            value=round(min(max(median, 1.0), 5.0), 1),
            step=0.1, key=f"input_{feature}"
        )
    if feature == "rating_count":
        return st.number_input(
            label, min_value=0, value=max(0, int(round(median))),
            step=1, key=f"input_{feature}"
        )
    if feature == "discount_percentage":
        return st.number_input(
            label, min_value=0.0, max_value=100.0,
            value=min(max(median, 0.0), 100.0),
            step=1.0, key=f"input_{feature}"
        )
    if feature in ("actual_price", "discounted_price"):
        return st.number_input(
            label, min_value=0.0, value=max(0.0, median),
            step=1.0, key=f"input_{feature}"
        )
    # Other uploaded CSVs may have different numeric fields.
    return st.number_input(label, value=median, key=f"input_{feature}")


def valid_value(feature, value):
    """Second-level validation before a value reaches the model."""
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return False
    if feature == "rating":
        return 1 <= value <= 5
    if feature == "discount_percentage":
        return 0 <= value <= 100
    if feature == "rating_count":
        return value >= 0 and value == int(value)
    if feature in ("actual_price", "discounted_price"):
        return value >= 0
    return True


# 1. Upload, clean, and preview
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is None:
    st.info("Upload your Amazon CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("This CSV contains no data rows. Please upload a nonempty file.")
        st.stop()
    df, invalid_counts = clean_data(df)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if df[c].notna().any()
]

st.subheader("Dataset Preview")
st.dataframe(df.head(10), hide_index=True)
if invalid_counts:
    details = ", ".join(f"{col}: {count}" for col, count in invalid_counts.items())
    st.warning(
        "Invalid numeric entries were treated as missing and excluded from the "
        f"affected charts/models. Counts by column: {details}."
    )

if len(numeric_cols) < 3:
    st.error("The uploaded CSV needs at least three usable numeric columns.")
    st.stop()

# 2. Data visualization
st.subheader("Explore Product Data")
variable = st.selectbox("Choose a numeric column", numeric_cols)
chart_type = st.selectbox("Choose a chart type", ["Histogram", "Scatter", "Line"])

if chart_type == "Histogram":
    fig = px.histogram(df.dropna(subset=[variable]), x=variable,
                       title=f"Distribution of {variable}")
elif chart_type == "Scatter":
    x_options = [col for col in numeric_cols if col != variable]
    x_variable = st.selectbox("Choose an X-axis column", x_options)
    fig = px.scatter(df.dropna(subset=[x_variable, variable]),
                     x=x_variable, y=variable,
                     title=f"{variable} vs. {x_variable}")
else:
    fig = px.line(x=np.arange(1, len(df) + 1), y=df[variable],
                  labels={"x": "Product row", "y": variable},
                  title=f"{variable} by Product Row")
    st.caption("Row order is not a time sequence.")
st.plotly_chart(fig, width="stretch")

# 3. Linear regression. The discount percentage is calculated from prices,
# so exclude it from the regression to avoid a circular relationship.
st.subheader("Predict a Product Metric")
regression_cols = [col for col in numeric_cols if col != "discount_percentage"]
if len(regression_cols) < 3:
    st.error("At least three eligible numeric columns are needed for regression.")
    st.stop()

target = st.selectbox(
    "Choose the prediction target", regression_cols,
    index=regression_cols.index("discounted_price")
    if "discounted_price" in regression_cols else 0
)
feature_options = [col for col in regression_cols if col != target]
feature1 = st.selectbox(
    "Choose input 1", feature_options,
    index=feature_options.index("actual_price")
    if "actual_price" in feature_options else 0
)
feature2_options = [col for col in feature_options if col != feature1]
feature2 = st.selectbox(
    "Choose input 2", feature2_options,
    index=feature2_options.index("rating_count")
    if "rating_count" in feature2_options else 0
)

model_df = df[[feature1, feature2, target]].dropna()
if len(model_df) < 10:
    st.error("Fewer than 10 complete rows remain for these three columns. Choose others.")
    st.stop()

X = model_df[[feature1, feature2]]
y = model_df[target]
if any(X[col].nunique() < 2 for col in X.columns):
    st.error("A selected input has no variation. Choose a different predictor.")
    st.stop()
if y.nunique() < 2:
    st.error("The target has no variation. Choose a different target.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if y_test.nunique() > 1 else np.nan
except (ValueError, OverflowError, FloatingPointError) as exc:
    st.error(f"Could not fit this regression: {exc}")
    st.stop()

st.write("**Model Performance (Test Set)**")
st.write(f"R² Score: {r2:.3f}" if np.isfinite(r2)
         else "R² Score: unavailable (test target has no variation)")
st.write(f"Mean Absolute Error: {mae:,.2f}")
st.caption(f"Trained on {len(X_train):,} rows; evaluated on {len(X_test):,} rows.")

# 4. User-entered predictions
st.write("**Enter values to make a prediction:**")
col1, col2 = st.columns(2)
with col1:
    value1 = get_user_input(feature1, X[feature1])
with col2:
    value2 = get_user_input(feature2, X[feature2])

if not valid_value(feature1, value1) or not valid_value(feature2, value2):
    st.error("Invalid input. Please enter values within the allowed range.")
    st.stop()

for col, value in [(feature1, value1), (feature2, value2)]:
    if value < X_train[col].min() or value > X_train[col].max():
        st.warning(f"{col} is outside the training range; the prediction may be unreliable.")

new_data = pd.DataFrame({feature1: [value1], feature2: [value2]})
try:
    prediction = float(model.predict(new_data)[0])
except (ValueError, OverflowError, FloatingPointError) as exc:
    st.error(f"Could not make a prediction: {exc}")
    st.stop()
if not np.isfinite(prediction):
    st.error("The prediction was not a finite number. Try smaller input values.")
    st.stop()

formatted_prediction = (
    f"₹{prediction:,.2f}" if target in ("actual_price", "discounted_price")
    else f"{prediction:,.2f}"
)
st.metric(f"Predicted {target}", formatted_prediction)

# Linear regression has no built-in restriction on its OUTPUT values.
if target == "rating" and not 1 <= prediction <= 5:
    st.warning("The predicted rating is outside 1–5 and is not a valid rating.")
elif target == "rating_count":
    if prediction < 0:
        st.warning("The predicted number of ratings is negative and is not valid.")
    elif abs(prediction - round(prediction)) > 1e-6:
        st.caption("Rating count is a whole-number quantity; interpret this as an estimate.")
elif target in ("actual_price", "discounted_price") and prediction < 0:
    st.warning("The predicted price is negative and is not a valid price.")

# 5. Model purpose and consequences
st.write(
    "This model estimates an Amazon product metric using two other "
    "numeric characteristics; incorrect predictions could mislead "
    "price comparisons or purchasing decisions."
)
