import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Amazon Product Explorer", layout="wide")
st.title("Amazon Product Explorer")

st.markdown(
    """
    **Explore and predict Amazon product metrics.** Upload an Amazon product CSV to
    preview the records, compare numeric product attributes using a histogram,
    scatter plot, or line chart, and build a linear regression model. Choose two
    numeric inputs and a target, then enter your own values to see a predicted
    result. The app also reports test-set R² and mean absolute error to help you
    evaluate the model.

    **Automatic data cleaning:** The app removes currency symbols (₹), commas,
    and percent signs from the Amazon numeric fields, converts those fields to
    numbers, and treats missing, nonnumeric, or out-of-range entries as missing
    values. Affected rows are excluded only from charts or models that use those
    columns; the app does not impute missing values. The preview shows the cleaned
    data, and any invalid-entry counts appear below it.

    **Purpose and limitations:** This model estimates an Amazon product metric
    from two other numeric characteristics. Incorrect predictions could mislead
    price comparisons or purchasing decisions. Predictions are statistical
    estimates, not guarantees.
    """
)

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
    """Allow manual input; validate afterward so errors explain valid values."""
    median = float(training_values.median())
    if feature == "rating":
        step = 0.1
    elif feature == "rating_count":
        step = 1.0
    else:
        step = 1.0

    value = st.number_input(
        f"Enter {feature}",
        value=round(median, 1) if feature == "rating" else median,
        step=step,
        key=f"input_{feature}",
    )
    hints = {
        "rating": "Valid range: 1 to 5.",
        "rating_count": "Valid values: nonnegative whole numbers.",
        "discount_percentage": "Valid range: 0% to 100%.",
        "actual_price": "Valid values: 0 or greater.",
        "discounted_price": "Valid values: 0 or greater.",
    }
    if feature in hints:
        st.caption(hints[feature])
    return value


def input_error(feature, value):
    """Return a helpful error message, or None if the value is valid."""
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return f"{feature}: enter a finite numeric value."
    if feature == "rating" and not (1 <= value <= 5):
        return "Rating must be between 1 and 5 (inclusive)."
    if feature == "discount_percentage" and not (0 <= value <= 100):
        return "Discount percentage must be between 0% and 100% (inclusive)."
    if feature == "rating_count" and (value < 0 or value != int(value)):
        return "Rating count must be a nonnegative whole number (0, 1, 2, ...)."
    if feature in ("actual_price", "discounted_price") and value < 0:
        return f"{feature} must be 0 or greater (prices cannot be negative)."
    return None


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

errors = [
    message for feature, value in [(feature1, value1), (feature2, value2)]
    if (message := input_error(feature, value)) is not None
]
if errors:
    for message in errors:
        st.error(message)
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

