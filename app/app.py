# For the GUI
import streamlit as st

# For Data Manipulation
import pandas as pd
import numpy as np  # for handling array
import pickle       # for serializing and deserializing

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# For Modelling, Training and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score



# Set page configuration
st.set_page_config(
    # page_title="Demand Forecasting System",
    page_icon="📊",
    # layout="wide"
)

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Visualization", "Prediction"])

# Function to load the data
@st.cache_data  # caches the output of the load_data()
def load_data(file=None):
    try:
        # If a file is uploaded, use it; otherwise, try to load the default file
        if file is not None:
            if hasattr(file, 'name') and file.name.endswith('.csv'):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
        else:
            # Try to load the default file (for development/testing)
            try:
                data = pd.read_csv("ecom.csv")
            except FileNotFoundError:
                st.warning("No file uploaded and default file not found. Please upload a file.")
                return None
        
        # Preprocess the data
        data['DateAdded'] = pd.to_datetime(data['DateAdded'], errors='coerce')
        
        # Extract useful date features
        data['Year'] = data['DateAdded'].dt.year
        data['Month'] = data['DateAdded'].dt.month
        data['Day'] = data['DateAdded'].dt.day
        data['DayOfWeek'] = data['DateAdded'].dt.dayofweek

        # Create a Revenue Feature
        data["Revenue"] = data['Sales'] * data["Price"] * (1 - data["Discount"] / 100)
        
        # Create season feature
        conditions = [
            (data['Month'] >= 3) & (data['Month'] <= 5),
            (data['Month'] >= 6) & (data['Month'] <= 8),
            (data['Month'] >= 9) & (data['Month'] <= 11),
            (data['Month'] == 12) | (data['Month'] <= 2)
        ]
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        data['Season'] = np.select(conditions, seasons, default='Unknown')
        
        data['DemandCategory'] = pd.qcut(data['Revenue'], q=3, labels=['Low', 'Medium', 'High'])
        
        return data
    # Error Prompt to return if there was a problem reading the file or processing it
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train the model
@st.cache_resource  # Avoids unnecessary retraining
def train_model(data):
    # Define features and target
    X = data.drop(['DemandCategory', 'DateAdded', 'ProductID', 'ProductName', 'Sales', 'Revenue'], axis=1)
    y = data['DemandCategory']
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # PRE-PROCESSING 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ensures that same pipeline can be applied consistently during training and prediction
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Gradient Boosting Classifier
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    
    # Train the model
    gb_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = gb_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
    
    return gb_pipeline, accuracy, X_test, y_test, numeric_cols, categorical_cols

# Function to make predictions
def predict_demand(model, product_category, price, discount, city, season):
    """Predict demand for a new product based on inputs."""
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Category': [product_category],
        'Price': [price],
        'Discount': [discount],
        'City': [city],
        'Season': [season],

        # Fill in default values for other required columns
        'Rating': [3.0],  # Default rating
        'NumReviews': [0],  # No reviews yet
        'StockQuantity': [100],  # Default stock
        'Sales': [200],  # No sales yet
        'Year': [2024],  # Default year
        'Month': [7],  # Default month
        'Day': [1],  # Default day
        'DayOfWeek': [0],  # Default day of week
        'Revenue': [price * (1 - discount / 100)]  # Calculate revenue
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Get probabilities for each class
    prob_dict = {class_name: prob for class_name, prob in zip(model.classes_, probabilities)}
    
    return prediction, prob_dict

# Home page content
def home_page(data):

    if data is not None:
        st.success(f"Dataset loaded successfully with {data.shape[0]} instances.")

        # Demand Distribution Chart
        st.subheader("Summary")
        # Inject CSS for styling the cards
        st.markdown("""
            <style>
            .metric-card {
                background-color: #f8f9fa; /* Light gray background */
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                text-align: center;
                margin: 10px 0;
            }
            .metric-header {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #007bff; /* Blue text */
            }
            </style>
        """, unsafe_allow_html=True)

        # Create four metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Total Revenue</div>
                <div class="metric-value">${data['Revenue'].sum() / 1000000:.2f}M</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Total Products Sold</div>
                <div class="metric-value">{data['StockQuantity'].sum():,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Average Price</div>
                <div class="metric-value">${data['Price'].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Average Discount</div>
                <div class="metric-value">{data['Discount'].mean() * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Show a sample of the dataset
        st.subheader("Sample Data")
        st.dataframe(data.head())

# Data exploration page content
def exploration_page(data):
    st.header("Data Visualization ")
    
    if data is not None:
        # Sidebar for exploration options
        exploration_option = st.sidebar.selectbox(
            "Choose an exploration view",
            ["Demand by Season", "Demand by Category", "Price Analysis", "Sales Analysis"]
        )
        
        if exploration_option == "Demand by Season":
            st.subheader("Demand by Season")

            # # List of seasons
            seasons = data["Season"].unique()

                # Define custom colors for better aesthetics
            custom_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

            # Create separate charts for each season
            for i, season in enumerate(seasons):
                st.subheader(f"{season} Demand Distribution")
                
                # Filter data for the current season
                season_data = data[data["Season"] == season]
                
                # Count total demand per product category
                category_demand_counts = season_data.groupby("Category").size().reset_index(name="Count")
                
                # Create a bar chart with enhanced aesthetics
                fig = px.bar(category_demand_counts, x="Category", y="Count",
                            title=f"{season} - Total Demand by Product Category",
                            labels={"Category": "Product Category", "Count": "Total Demand"},
                            template="plotly_dark",  # Dark theme for a sleek look
                            color_discrete_sequence=[custom_colors[i % len(custom_colors)]]  # Cycle through colors
                            )

                # Improve layout with grid and transparency
                fig.update_layout(xaxis_title="Product Category", yaxis_title="Total Demand",
                                hovermode="x unified", 
                                plot_bgcolor="rgba(0,0,0,0)", 
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"))

                # Display interactive Plotly chart
                st.plotly_chart(fig, use_container_width=True)

            
        elif exploration_option == "Demand by Category":
            
            # Group data to count occurrences of each DemandCategory in each Season
            seasonal_demand = pd.crosstab(data['Season'], data['DemandCategory'], normalize="index") * 100
            seasonal_demand = seasonal_demand.reset_index()

            # Convert to long format for visualization
            melted_data = seasonal_demand.melt(id_vars="Season", var_name="DemandCategory", value_name="Percentage")

            # Create a stacked bar chart
            fig = px.bar(
                melted_data, 
                x="Season", y="Percentage", color="DemandCategory", 
                title="Demand Category Distribution Across Seasons",
                labels={"Season": "Season", "Percentage": "Percentage (%)", "DemandCategory": "Demand Level"},
                barmode="stack", template="plotly_white"
            )

            # Improve layout
            fig.update_layout(yaxis=dict(title="Percentage (%)", tickformat=".2f"), 
                            xaxis_title="Season",
                            legend_title="Demand Category",
                            hovermode="x unified")

            # Display chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        elif exploration_option == "Price Analysis":
            # Aggregate price over time (monthly)
            monthly_price_trend = data.groupby(['Year', 'Month'])['Price'].mean().reset_index()

            # Create a 'Date' column for plotting
            monthly_price_trend['Date'] = pd.to_datetime(monthly_price_trend['Year'].astype(str) + '-' + 
                                                        monthly_price_trend['Month'].astype(str))

            # Sort the data
            monthly_price_trend = monthly_price_trend.sort_values(by="Date")

            # Create line chart
            fig = px.line(
                monthly_price_trend, 
                x="Date", 
                y="Price", 
                title="Average Price Trend Over Time",
                labels={"Date": "Time (Months)", "Price": "Average Price ($)"},
                markers=True,
                template="plotly_white"
            )

            # Customize layout
            fig.update_traces(line=dict(color="blue", width=2))
            fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified")

            # Display chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)



            
        elif exploration_option == "Sales Analysis":
            st.subheader("Historical Sales Analysis")
            monthly_sales = data.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

            # Create a 'Date' column representing the first day of each month
            monthly_sales['Date'] = pd.to_datetime(monthly_sales['Year'].astype(str) + '-' + 
                                                monthly_sales['Month'].astype(str))

            # Sort by Date
            monthly_sales = monthly_sales.sort_values(by="Date")

            # Create an interactive line chart
            fig = px.line(monthly_sales, x="Date", y="Sales", markers=True, 
                        title=" Monthly Sales Over Time",
                        labels={"Date": "Time (Months)", "Sales": "Total Sales ($)"},
                        template="plotly_white")

            # Customize layout
            fig.update_traces(line=dict(color="blue", width=2))  # Adjust line color & thickness
            fig.update_layout(xaxis_title="Date", yaxis_title="Total Revenue ($)", 
                            hovermode="x unified")  # Ensures tooltips appear per month

            # Display the interactive Plotly chart
            st.plotly_chart(fig, use_container_width=True)


# Assuming `data` is already loaded
def prediction_page(data):
    st.header("Demand Prediction")
    st.markdown("Enter the details of the product to predict its demand level.")

    if data is not None:
        model, accuracy, _, _, _, _ = train_model(data)

        # Get unique categories & cities
        categories = sorted(data['Category'].unique())
        cities = sorted(data['City'].unique())

        # **Move Category & Product selection OUTSIDE the form**
        category = st.selectbox("Product Category", categories)

        # Dynamically update product list when category changes
        products_in_category = data[data['Category'] == category]['ProductName'].unique()
        product = st.selectbox("Select Product", products_in_category)

        # Get default price for selected product
        default_price = data[(data['Category'] == category) & (data['ProductName'] == product)]['Price'].mean()
        if pd.isna(default_price):
            default_price = data[data['Category'] == category]['Price'].mean()
        if pd.isna(default_price):
            default_price = 100.0  # Default fallback price

        # Define price range
        min_price = max(0.0, data[data['Category'] == category]['Price'].min() * 0.5)
        max_price = data[data['Category'] == category]['Price'].max() * 1.5
        if pd.isna(min_price) or min_price <= 0:
            min_price = 0.0
        if pd.isna(max_price) or max_price <= min_price:
            max_price = min_price + 10000.0

        # **Now, Create the Form for Other Inputs**
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                price = st.number_input("Price", min_value=float(min_price), max_value=float(max_price), value=float(default_price), step=10.0)
                discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

            with col2:
                city = st.selectbox("City", cities)
                season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

            submit_button = st.form_submit_button("Predict Demand")

        if submit_button:
            prediction, probabilities = predict_demand(model, category, price, discount, city, season)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prediction Result")
                if prediction == "High":
                    st.success(f"**Predicted Demand: {prediction}**")
                elif prediction == "Medium":
                    st.warning(f"**Predicted Demand: {prediction}**")
                else:
                    st.error(f"**Predicted Demand: {prediction}**")

                prediction_prob = probabilities[prediction] * 100
                st.write(f"**Confidence Level:** {prediction_prob:.2f}%")

                # Inventory Recommendation
            
                st.subheader("Inventory Recommendation")
                if prediction == "High":
                    st.write("**High demand expected**. Consider stocking up more.")
                elif prediction == "Medium":
                    st.write("**Medium demand expected**. Maintain standard inventory levels.")
                else:
                    st.write("**Low demand expected**. Consider reducing inventory or increasing discounts.")
            
            with col2:
            # Probability Bar Chart
                fig = px.bar(
                    x=list(probabilities.keys()),
                    y=[v * 100 for v in probabilities.values()],
                    color=list(probabilities.keys()),
                    title="Prediction Probabilities",
                    labels={"x": "Demand Category", "y": "Probability (%)"},
                    template="plotly_white"
                )
                fig.update_traces(marker=dict(line=dict(width=2)))
                st.plotly_chart(fig, use_container_width=True)

            # Display Input Summary
            st.subheader("Input Summary")
            summary_data = {
                "Category": [category],
                "Product": [product],
                "Price": [f"${price:.2f}"],
                "Discount": [f"{discount:.1f}%"],
                "City": [city],
                "Season": [season],
                "Revenue": [f"${price * (1 - discount / 100):.2f}"]
            }
            st.table(pd.DataFrame(summary_data))

            # Recommendations
            st.subheader("Recommendations")
            if discount < 10 and prediction == "Low":
                st.write("🔹 **Consider increasing the discount** to boost sales.")
            if discount > 30 and prediction == "High":
                st.write("🔹 **You may be offering too high of a discount. Try lowering it to increase profit margins.**")

            seasonal_trends = {
                "Winter": "Winter sees high demand for holiday-related products and indoor electronics.",
                "Spring": "Spring often has increased demand for outdoor & fitness products.",
                "Summer": "Summer is strong for travel, outdoor, and seasonal goods.",
                "Fall": "Fall sees higher demand for back-to-school items and early holiday shopping."
            }
            st.write(f"**Seasonal Trend:** {seasonal_trends.get(season, 'No trend data available')}")


# Main app code
st.markdown("<h1 style='text-align: center;'>Demand Forecasting System</h1>", unsafe_allow_html=True)
st.markdown("---")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

# Load data based on uploaded file or default
data = load_data(uploaded_file)

# Run the appropriate page based on selection
if page == "Home":
    home_page(data)
elif page == "Data Visualization":
    exploration_page(data)
elif page == "Prediction":
    prediction_page(data)


# Add footer
st.markdown("---")
st.markdown("Product Demand Prediction System")