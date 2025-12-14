import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Zopper - Device Insurance Attach % Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_excel('Jumbo & Company_ Attach % .xls')
        
        # Convert monthly columns to long format for easier analysis
        months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_long = pd.melt(
            df, 
            id_vars=['Branch', 'Store_Name'], 
            value_vars=months,
            var_name='Month', 
            value_name='Attach_Percentage'
        )
        
        # Convert attach percentage to actual percentage (assuming values are in 0-1 range)
        df_long['Attach_Percentage'] = df_long['Attach_Percentage'] * 100
        
        # Create month order for proper sorting
        month_order = {'Aug': 1, 'Sep': 2, 'Oct': 3, 'Nov': 4, 'Dec': 5}
        df_long['Month_Order'] = df_long['Month'].map(month_order)
        df_long = df_long.sort_values(['Branch', 'Store_Name', 'Month_Order'])
        
        return df, df_long
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def categorize_stores(df_long):
    """Categorize stores based on performance"""
    store_stats = df_long.groupby('Store_Name').agg({
        'Attach_Percentage': ['mean', 'std', 'min', 'max']
    }).reset_index()
    store_stats.columns = ['Store_Name', 'Mean_Attach', 'Std_Attach', 'Min_Attach', 'Max_Attach']
    
    # Merge with branch info
    branch_info = df_long[['Store_Name', 'Branch']].drop_duplicates()
    store_stats = store_stats.merge(branch_info, on='Store_Name')
    
    # Categorize stores
    def categorize(mean_attach, std_attach):
        if mean_attach >= 30:
            if std_attach < 10:
                return "High Performer - Stable"
            else:
                return "High Performer - Volatile"
        elif mean_attach >= 20:
            if std_attach < 10:
                return "Medium Performer - Stable"
            else:
                return "Medium Performer - Volatile"
        elif mean_attach >= 10:
            return "Low Performer - Needs Improvement"
        else:
            return "Underperformer - Critical"
    
    store_stats['Category'] = store_stats.apply(
        lambda x: categorize(x['Mean_Attach'], x['Std_Attach']), axis=1
    )
    
    return store_stats

def prepare_prediction_data(df):
    """Prepare data for prediction model"""
    # Create features from historical data
    months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    prediction_df = df.copy()
    
    # Calculate features
    prediction_df['Mean_Attach'] = prediction_df[months].mean(axis=1)
    prediction_df['Std_Attach'] = prediction_df[months].std(axis=1)
    prediction_df['Trend'] = (prediction_df['Dec'] - prediction_df['Aug']) / 4  # Monthly trend
    prediction_df['Recent_Avg'] = (prediction_df['Nov'] + prediction_df['Dec']) / 2
    prediction_df['Volatility'] = prediction_df[months].std(axis=1)
    
    # Encode branch as numeric
    branch_encoding = {branch: idx for idx, branch in enumerate(prediction_df['Branch'].unique())}
    prediction_df['Branch_Encoded'] = prediction_df['Branch'].map(branch_encoding)
    
    return prediction_df, branch_encoding

def train_prediction_model(df):
    """Train model to predict January attach percentage"""
    months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Use Dec as target (proxy for Jan prediction)
    # In real scenario, we'd use historical Dec to predict Jan
    # Here we'll use Aug-Nov to predict Dec, then use that pattern for Jan
    
    X = []
    y = []
    
    for idx, row in df.iterrows():
        # Use Aug-Nov to predict Dec
        features = [
            row['Aug'],
            row['Sep'],
            row['Oct'],
            row['Nov'],
            np.mean([row['Aug'], row['Sep'], row['Oct'], row['Nov']]),
            np.std([row['Aug'], row['Sep'], row['Oct'], row['Nov']]),
            (row['Nov'] - row['Aug']) / 3,  # Trend
        ]
        X.append(features)
        y.append(row['Dec'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def predict_january(df, model):
    """Predict January attach percentage for each store"""
    predictions = []
    
    for idx, row in df.iterrows():
        # Use Sep-Dec to predict Jan (similar pattern as before)
        features = [
            row['Sep'],
            row['Oct'],
            row['Nov'],
            row['Dec'],
            np.mean([row['Sep'], row['Oct'], row['Nov'], row['Dec']]),
            np.std([row['Sep'], row['Oct'], row['Nov'], row['Dec']]),
            (row['Dec'] - row['Sep']) / 3,  # Trend
        ]
        
        pred = model.predict([features])[0]
        # Ensure prediction is within reasonable bounds
        pred = max(0, min(1, pred))
        predictions.append(pred * 100)  # Convert to percentage
    
    df['Jan_Predicted'] = predictions
    return df

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Zopper Device Insurance Attach % Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df, df_long = load_data()
    
    if df is None or df_long is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Sidebar
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis View",
        ["🏠 Overview Dashboard", "📈 Branch Analysis", "🏪 Store Analysis", 
         "📅 Monthly Trends", "🎯 Store Categorization", "🔮 January Prediction"]
    )
    
    # Overview Dashboard
    if page == "🏠 Overview Dashboard":
        st.header("Executive Summary")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_stores = df['Store_Name'].nunique()
        total_branches = df['Branch'].nunique()
        overall_avg = df_long['Attach_Percentage'].mean()
        recent_avg = df_long[df_long['Month'] == 'Dec']['Attach_Percentage'].mean()
        
        with col1:
            st.metric("Total Stores", total_stores)
        with col2:
            st.metric("Total Branches", total_branches)
        with col3:
            st.metric("Overall Avg Attach %", f"{overall_avg:.2f}%")
        with col4:
            st.metric("Dec Avg Attach %", f"{recent_avg:.2f}%")
        
        st.markdown("---")
        
        # Overall trend
        st.subheader("📈 Overall Monthly Trend")
        monthly_avg = df_long.groupby('Month')['Attach_Percentage'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('Month_Order')
        
        fig = px.line(
            monthly_avg, 
            x='Month', 
            y='Attach_Percentage',
            markers=True,
            title="Average Attach Percentage by Month",
            labels={'Attach_Percentage': 'Attach %', 'Month': 'Month'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=3, marker_size=10)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Branch comparison
        st.subheader("🏢 Branch Performance Comparison")
        branch_avg = df_long.groupby('Branch')['Attach_Percentage'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            branch_avg,
            x='Branch',
            y='Attach_Percentage',
            title="Average Attach % by Branch",
            labels={'Attach_Percentage': 'Attach %', 'Branch': 'Branch'},
            color='Attach_Percentage',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Complete Dataset")
        st.dataframe(df, use_container_width=True, height=400)
    
    # Branch Analysis
    elif page == "📈 Branch Analysis":
        st.header("Branch-Wise Analysis")
        
        selected_branch = st.selectbox("Select Branch", ['All'] + sorted(df['Branch'].unique().tolist()))
        
        if selected_branch == 'All':
            branch_df = df_long.copy()
        else:
            branch_df = df_long[df_long['Branch'] == selected_branch]
        
        # Branch metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stores in Branch", branch_df['Store_Name'].nunique())
        with col2:
            st.metric("Avg Attach %", f"{branch_df['Attach_Percentage'].mean():.2f}%")
        with col3:
            st.metric("Best Month", branch_df.groupby('Month')['Attach_Percentage'].mean().idxmax())
        
        # Monthly trend for branch
        st.subheader("Monthly Trend")
        branch_monthly = branch_df.groupby('Month')['Attach_Percentage'].mean().reset_index()
        branch_monthly = branch_monthly.sort_values('Month_Order')
        
        fig = px.line(
            branch_monthly,
            x='Month',
            y='Attach_Percentage',
            markers=True,
            title=f"Monthly Trend - {selected_branch}",
            labels={'Attach_Percentage': 'Attach %'}
        )
        fig.update_traces(line_color='#2ca02c', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Store performance in branch
        st.subheader("Store Performance within Branch")
        store_perf = branch_df.groupby('Store_Name')['Attach_Percentage'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            store_perf,
            x='Store_Name',
            y='Attach_Percentage',
            title="Store Performance Ranking",
            labels={'Attach_Percentage': 'Avg Attach %'},
            color='Attach_Percentage',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Store Analysis
    elif page == "🏪 Store Analysis":
        st.header("Store-Wise Analysis")
        
        selected_store = st.selectbox("Select Store", sorted(df['Store_Name'].unique().tolist()))
        
        store_data = df_long[df_long['Store_Name'] == selected_store].sort_values('Month_Order')
        store_info = df[df['Store_Name'] == selected_store].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Branch", store_info['Branch'])
        with col2:
            st.metric("Avg Attach %", f"{store_data['Attach_Percentage'].mean():.2f}%")
        with col3:
            st.metric("Best Month", store_data.loc[store_data['Attach_Percentage'].idxmax(), 'Month'])
        with col4:
            st.metric("Worst Month", store_data.loc[store_data['Attach_Percentage'].idxmin(), 'Month'])
        
        # Store trend
        st.subheader("Monthly Performance Trend")
        fig = px.line(
            store_data,
            x='Month',
            y='Attach_Percentage',
            markers=True,
            title=f"Performance Trend - {selected_store}",
            labels={'Attach_Percentage': 'Attach %'}
        )
        fig.update_traces(line_color='#ff7f0e', line_width=3, marker_size=10)
        fig.add_hline(
            y=store_data['Attach_Percentage'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {store_data['Attach_Percentage'].mean():.2f}%"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly breakdown
        st.subheader("Monthly Breakdown")
        st.dataframe(
            store_data[['Month', 'Attach_Percentage']].set_index('Month'),
            use_container_width=True
        )
    
    # Monthly Trends
    elif page == "📅 Monthly Trends":
        st.header("Monthly Trends Analysis")
        
        selected_month = st.selectbox("Select Month", ['All', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        if selected_month == 'All':
            # Heatmap
            st.subheader("Attach % Heatmap by Store and Month")
            pivot_data = df_long.pivot(index='Store_Name', columns='Month', values='Attach_Percentage')
            pivot_data = pivot_data[['Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
            
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Month", y="Store", color="Attach %"),
                aspect="auto",
                color_continuous_scale="RdYlGn",
                title="Attach Percentage Heatmap"
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
        else:
            month_data = df_long[df_long['Month'] == selected_month]
            
            # Distribution
            st.subheader(f"Distribution of Attach % - {selected_month}")
            fig = px.histogram(
                month_data,
                x='Attach_Percentage',
                nbins=30,
                title=f"Distribution of Attach % in {selected_month}",
                labels={'Attach_Percentage': 'Attach %', 'count': 'Number of Stores'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top and bottom stores
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 10 Stores")
                top_stores = month_data.nlargest(10, 'Attach_Percentage')[['Store_Name', 'Branch', 'Attach_Percentage']]
                st.dataframe(top_stores, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Bottom 10 Stores")
                bottom_stores = month_data.nsmallest(10, 'Attach_Percentage')[['Store_Name', 'Branch', 'Attach_Percentage']]
                st.dataframe(bottom_stores, use_container_width=True)
    
    # Store Categorization
    elif page == "🎯 Store Categorization":
        st.header("Store Performance Categorization")
        
        store_stats = categorize_stores(df_long)
        
        # Category distribution
        st.subheader("Store Category Distribution")
        category_counts = store_stats['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig = px.pie(
            category_counts,
            values='Count',
            names='Category',
            title="Distribution of Stores by Performance Category"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category by branch
        st.subheader("Category Distribution by Branch")
        category_branch = pd.crosstab(store_stats['Branch'], store_stats['Category'])
        
        fig = px.bar(
            category_branch.reset_index().melt(id_vars='Branch', var_name='Category', value_name='Count'),
            x='Branch',
            y='Count',
            color='Category',
            title="Store Categories by Branch",
            barmode='stack'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed categorization table
        st.subheader("Detailed Store Categorization")
        display_stats = store_stats[['Store_Name', 'Branch', 'Mean_Attach', 'Std_Attach', 'Category']].sort_values('Mean_Attach', ascending=False)
        display_stats['Mean_Attach'] = display_stats['Mean_Attach'].round(2)
        display_stats['Std_Attach'] = display_stats['Std_Attach'].round(2)
        st.dataframe(display_stats, use_container_width=True, height=600)
        
        # Filter by category
        selected_category = st.selectbox("Filter by Category", ['All'] + sorted(store_stats['Category'].unique().tolist()))
        if selected_category != 'All':
            filtered = store_stats[store_stats['Category'] == selected_category]
            st.subheader(f"Stores in Category: {selected_category}")
            st.dataframe(
                filtered[['Store_Name', 'Branch', 'Mean_Attach', 'Std_Attach']].sort_values('Mean_Attach', ascending=False),
                use_container_width=True
            )
    
    # January Prediction
    elif page == "🔮 January Prediction":
        st.header("January 2026 Attach % Prediction")
        
        st.info("Using Random Forest Regression model trained on historical data to predict January attach percentages.")
        
        # Train model
        with st.spinner("Training prediction model..."):
            model, metrics = train_prediction_model(df)
            df_pred = predict_january(df, model)
        
        # Model metrics
        st.subheader("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"{metrics['MAE']*100:.2f}%")
        with col2:
            st.metric("Root Mean Squared Error", f"{metrics['RMSE']*100:.2f}%")
        with col3:
            st.metric("R² Score", f"{metrics['R2']:.3f}")
        
        st.markdown("---")
        
        # Predictions summary
        st.subheader("Prediction Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Predicted Attach %", f"{df_pred['Jan_Predicted'].mean():.2f}%")
        with col2:
            st.metric("Max Predicted", f"{df_pred['Jan_Predicted'].max():.2f}%")
        with col3:
            st.metric("Min Predicted", f"{df_pred['Jan_Predicted'].min():.2f}%")
        with col4:
            st.metric("Stores Above 30%", f"{(df_pred['Jan_Predicted'] > 30).sum()}")
        
        # Prediction vs Recent Performance
        st.subheader("Predicted vs December Performance")
        comparison_df = pd.DataFrame({
            'Store_Name': df_pred['Store_Name'],
            'Branch': df_pred['Branch'],
            'December_Attach': df_pred['Dec'] * 100,
            'January_Predicted': df_pred['Jan_Predicted']
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison_df['December_Attach'],
            y=comparison_df['January_Predicted'],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            text=comparison_df['Store_Name'],
            name='Stores'
        ))
        fig.add_trace(go.Scatter(
            x=[comparison_df['December_Attach'].min(), comparison_df['December_Attach'].max()],
            y=[comparison_df['December_Attach'].min(), comparison_df['December_Attach'].max()],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="January Prediction vs December Performance",
            xaxis_title="December Attach %",
            yaxis_title="Predicted January Attach %",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top predicted stores
        st.subheader("Top 20 Predicted Performers for January")
        top_predicted = df_pred.nlargest(20, 'Jan_Predicted')[['Store_Name', 'Branch', 'Dec', 'Jan_Predicted']]
        top_predicted['Dec'] = (top_predicted['Dec'] * 100).round(2)
        top_predicted['Jan_Predicted'] = top_predicted['Jan_Predicted'].round(2)
        top_predicted.columns = ['Store Name', 'Branch', 'Dec Attach %', 'Jan Predicted %']
        st.dataframe(top_predicted, use_container_width=True)
        
        # Branch-wise predictions
        st.subheader("Branch-wise Average Predictions")
        branch_pred = df_pred.groupby('Branch')['Jan_Predicted'].mean().sort_values(ascending=False).reset_index()
        branch_pred['Jan_Predicted'] = branch_pred['Jan_Predicted'].round(2)
        
        fig = px.bar(
            branch_pred,
            x='Branch',
            y='Jan_Predicted',
            title="Average Predicted Attach % by Branch for January",
            labels={'Jan_Predicted': 'Predicted Attach %', 'Branch': 'Branch'},
            color='Jan_Predicted',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Complete predictions table
        st.subheader("Complete Predictions for All Stores")
        display_pred = df_pred[['Store_Name', 'Branch', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan_Predicted']].copy()
        display_pred['Aug'] = (display_pred['Aug'] * 100).round(2)
        display_pred['Sep'] = (display_pred['Sep'] * 100).round(2)
        display_pred['Oct'] = (display_pred['Oct'] * 100).round(2)
        display_pred['Nov'] = (display_pred['Nov'] * 100).round(2)
        display_pred['Dec'] = (display_pred['Dec'] * 100).round(2)
        display_pred['Jan_Predicted'] = display_pred['Jan_Predicted'].round(2)
        display_pred.columns = ['Store Name', 'Branch', 'Aug %', 'Sep %', 'Oct %', 'Nov %', 'Dec %', 'Jan Predicted %']
        st.dataframe(display_pred, use_container_width=True, height=600)
        
        # Download predictions
        csv = df_pred[['Store_Name', 'Branch', 'Jan_Predicted']].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="january_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

