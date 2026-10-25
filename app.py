import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Amazon Product Explorer", layout="wide")
st.title("Amazon Product Explorer")

# 1. Upload and preview data
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload your Amazon CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Clean numeric columns (based on your existing notebook)
cols_to_convert = [
    "discount_percentage", "rating", "rating_count",
    "actual_price", "discounted_price"
]

for col in cols_to_convert:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[₹,%]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), hide_index=True)

if len(numeric_cols) < 3:
    st.error("The CSV must contain at least three numeric columns.")
    st.stop()

# 2. Data visualization
st.subheader("Explore Product Data")

variable = st.selectbox("Choose a numeric column", numeric_cols)
chart_type = st.selectbox(
    "Choose a chart type",
    ["Histogram", "Scatter", "Line"]
)

if chart_type == "Histogram":
    fig = px.histogram(df, x=variable, title=f"Distribution of {variable}")

elif chart_type == "Scatter":
    x_options = [col for col in numeric_cols if col != variable]
    x_variable = st.selectbox("Choose an X-axis column", x_options)
    fig = px.scatter(
        df, x=x_variable, y=variable,
        title=f"{variable} vs. {x_variable}"
    )

else:
    plot_df = df[[variable]].reset_index()
    plot_df = plot_df.rename(columns={"index": "Product row"})
    fig = px.line(
        plot_df, x="Product row", y=variable,
        title=f"{variable} by Product Row"
    )
    st.caption("Row order is not a time sequence.")

st.plotly_chart(fig, width="stretch")

# 3. Linear regression
st.subheader("Predict a Product Metric")

target = st.selectbox(
    "Choose the prediction target",
    numeric_cols,
    index=numeric_cols.index("discounted_price")
    if "discounted_price" in numeric_cols else 0
)

feature_options = [col for col in numeric_cols if col != target]

feature1 = st.selectbox(
    "Choose input 1",
    feature_options,
    index=feature_options.index("actual_price")
    if "actual_price" in feature_options else 0
)

feature2_options = [col for col in feature_options if col != feature1]

feature2 = st.selectbox(
    "Choose input 2",
    feature2_options,
    index=feature2_options.index("rating_count")
    if "rating_count" in feature2_options else 0
)

if (
    target in ["discounted_price", "actual_price"]
    and "discount_percentage" in [feature1, feature2]
):
    st.warning(
        "Discount percentage is calculated from the prices. "
        "Using it here may introduce target leakage."
    )

# Remove rows with missing values in selected columns
model_df = df[[feature1, feature2, target]].dropna()

if len(model_df) < 10:
    st.error("Not enough complete rows to train the model.")
    st.stop()

X = model_df[[feature1, feature2]]
y = model_df[target]

# Train on 80%, evaluate on 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("**Model Performance (Test Set)**")
st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")

# 4. User prediction inputs
st.write("**Enter values to make a prediction:**")

col1, col2 = st.columns(2)

with col1:
    value1 = st.number_input(
        f"Enter {feature1}",
        value=float(X[feature1].median())
    )

with col2:
    value2 = st.number_input(
        f"Enter {feature2}",
        value=float(X[feature2].median())
    )

new_data = pd.DataFrame({
    feature1: [value1],
    feature2: [value2]
})

prediction = model.predict(new_data)[0]

st.metric(f"Predicted {target}", f"{prediction:,.2f}")

# 5. Model purpose and consequences
st.write(
    "This model estimates an Amazon product metric using two other "
    "numeric characteristics; incorrect predictions could mislead "
    "price comparisons or purchasing decisions."
)
