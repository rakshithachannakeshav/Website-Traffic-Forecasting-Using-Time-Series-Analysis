import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import zipfile

# Set up the Streamlit app
st.set_page_config(
    page_title="Brand Ranking Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {border-radius: 5px;}
    .stSelectbox, .stSlider, .stFileUploader {margin-bottom: 20px;}
    .plot-container {background-color: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the data with encoding detection"""
    try:
        # Try common encodings if UTF-8 fails
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer for each attempt
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # If all encodings failed, try with error handling
        if 'df' not in locals():
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            st.warning("Used error replacement for problematic characters")
        
        # Data cleaning
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        # Ensure year is integer
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Try uploading a different file format or check the file encoding")
        return None

@st.cache_data
def load_zipped_data(uploaded_file):
    """Load data from a zip file"""
    try:
        with zipfile.ZipFile(uploaded_file) as z:
            csv_file = z.namelist()[0]  # get first file in zip
            with z.open(csv_file) as f:
                # Try multiple encodings for the zipped file
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        f.seek(0)
                        df = pd.read_csv(f, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if 'df' not in locals():
                    f.seek(0)
                    df = pd.read_csv(f, encoding='utf-8', errors='replace')
                    st.warning("Used error replacement for problematic characters in ZIP file")
                
                # Data cleaning
                df.dropna(inplace=True)
                df.drop_duplicates(inplace=True)
                
                if 'year' in df.columns:
                    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                
                return df
    except Exception as e:
        st.error(f"Error reading ZIP file: {e}")
        return None

def main():
    st.title("📈 Brand Ranking Analysis Dashboard")
    st.markdown("Analyze brand ranking trends over time with interactive visualizations")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data file", 
        type=["csv", "zip"],
        help=r"C:\Users\Meghana\OneDrive\Desktop\meghna\.vs studio\TrafficDashboard\trends.csv.zip"
    )
    
    if uploaded_file is not None:
        # Determine file type and load accordingly
        if uploaded_file.name.endswith('.zip'):
            df = load_zipped_data(uploaded_file)
        else:
            df = load_data(uploaded_file)
        
        if df is not None:
            st.sidebar.success("Data loaded successfully!")
            
            # Check if required columns exist
            required_columns = {'year', 'query', 'rank'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()
            
            # Sidebar controls
            st.sidebar.header("Filters")
            year_range = st.sidebar.slider(
                "Select Year Range",
                min_value=int(df['year'].min()),
                max_value=int(df['year'].max()),
                value=(int(df['year'].min()), int(df['year'].max()))
            )
            
            num_brands = st.sidebar.slider(
                "Number of Top Brands to Display",
                min_value=5,
                max_value=30,
                value=15
            )
            
            # Filter data based on selections
            filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
            top_brands = filtered_df['query'].value_counts().head(num_brands).index
            df_top = filtered_df[filtered_df['query'].isin(top_brands)]
            
            # Dashboard layout
            tab1, tab2, tab3 = st.tabs(["📈 Trends", "📊 Distributions", "🔍 Deep Dive"])
            
            with tab1:
                st.header("Brand Ranking Trends")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Rank Over Time (Top {num_brands} Brands)")
                    st.markdown("This line chart shows how the rankings of the most frequently appearing brands have changed over time.")
                    
                    # Matplotlib line plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot_df = df_top.pivot_table(index='year', columns='query', values='rank', aggfunc='mean').ffill()
                    
                    for brand in pivot_df.columns:
                        ax.plot(pivot_df.index, pivot_df[brand], label=brand)
                    
                    ax.invert_yaxis()
                    ax.set_title(f'Rank Over Time (Top {num_brands} Brands)', pad=20)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Rank')
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax.grid(True)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Download button
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    st.download_button(
                        label="Download Line Chart",
                        data=buf.getvalue(),
                        file_name="brand_rank_trend.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.subheader("Interactive Rank Trends")
                    
                    # Plotly interactive line chart
                    fig = px.line(df_top, x='year', y='rank', color='query',
                                title=f'Interactive Rank Trends - Top {num_brands} Brands',
                                labels={'rank': 'Rank', 'year': 'Year', 'query': 'Brand'},
                                height=500)
                    fig.update_yaxes(autorange='reversed')
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Average Rank Over Time (All Brands)")
                avg_rank_all = filtered_df.groupby('year')['rank'].mean()
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(avg_rank_all.index, avg_rank_all.values, marker='o', color='purple')
                ax.set_title('Average Rank of All Brands Over Time')
                ax.set_xlabel('Year')
                ax.set_ylabel('Average Rank')
                ax.invert_yaxis()
                ax.grid(True)
                st.pyplot(fig)
            
            with tab2:
                st.header("Brand Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Top {num_brands} Brands by Frequency")
                    brand_counts = filtered_df['query'].value_counts().head(num_brands)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    brand_counts.plot(kind='barh', color='teal', ax=ax)
                    ax.set_title(f'Top {num_brands} Brands by Frequency')
                    ax.set_xlabel('Appearances')
                    st.pyplot(fig)
                    
                    # Download button
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    st.download_button(
                        label="Download Bar Chart",
                        data=buf.getvalue(),
                        file_name="brand_frequency.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.subheader("Brand Frequency Distribution")
                    brand_counts_pie = brand_counts.reset_index()
                    brand_counts_pie.columns = ['Brand', 'Frequency']
                    fig_pie = px.pie(brand_counts_pie, names='Brand', values='Frequency',
                                    title=f'Top {num_brands} Brands - Frequency Distribution',
                                    hole=0.3)
                    fig_pie.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.subheader("Rank Distribution by Brand")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=df_top, x='query', y='rank', ax=ax)
                ax.set_title(f'Rank Distribution by Brand (Top {num_brands})')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.invert_yaxis()
                st.pyplot(fig)
            
            with tab3:
                st.header("Deep Dive Analysis")
                
                st.subheader("Heatmap: Average Rank by Year")
                heatmap_data = df_top.pivot_table(index='query', columns='year', values='rank', aggfunc='mean')
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(heatmap_data, cmap='mako_r', linewidths=0.5, ax=ax)
                ax.set_title(f'Heatmap: Average Rank of Top {num_brands} Brands Over Years')
                ax.set_xlabel('Year')
                ax.set_ylabel('Brand')
                st.pyplot(fig)
                
                st.subheader("Brand Presence Over Time")
                brands_per_year = filtered_df.groupby('year')['query'].nunique()
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.bar(brands_per_year.index, brands_per_year.values, color='coral')
                ax.set_title('Number of Unique Brands Per Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Brands')
                st.pyplot(fig)
                
                # Data summary
                st.subheader("Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(filtered_df))
                
                with col2:
                    st.metric("Unique Brands", filtered_df['query'].nunique())
                
                with col3:
                    st.metric("Years Covered", f"{filtered_df['year'].min()} - {filtered_df['year'].max()}")
                
                # Raw data display
                if st.checkbox("Show raw data"):
                    st.dataframe(filtered_df.sort_values(['year', 'rank']))
    
    else:
        st.info("👈 Please upload a CSV or ZIP file to get started")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                use_column_width=True, caption="Brand Ranking Analysis")

if __name__ == "__main__":
    main()