import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Fake News Detection - Model Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:5001"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📊 Fake News Detection - Model Comparison Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # Fetch available models from API
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            available_models = response.json().get("models", [])
        else:
            st.error("Failed to fetch models from API")
            available_models = []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        st.error("Please ensure the Flask API is running.")
        st.stop()
        available_models = []
    
    if available_models:
        st.success(f"✅ Connected to API - {len(available_models)} models available")
        st.markdown("---")
        st.subheader("Select Models to Compare")
        
        # Select all models by default
        selected_models = st.multiselect(
            "Choose models:",
            options=available_models,
            default=available_models,
            help="Select one or more models to compare"
        )
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model to compare")
    else:
        selected_models = []
        st.warning("No models available")

# Main content area
if selected_models:
    # Fetch comparison data from API
    with st.spinner("Fetching model comparison data..."):
        try:
            compare_response = requests.post(
                f"{API_BASE_URL}/compare",
                json={"models": selected_models},
                timeout=10
            )
            
            if compare_response.status_code == 200:
                comparison_data = compare_response.json()
                
                # Create DataFrame for easier manipulation
                metrics_df = pd.DataFrame(comparison_data).T
                metrics_df = metrics_df.reset_index()
                metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
                
                # Display metrics in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "📋 Detailed Metrics", "🏆 Rankings"])
                
                with tab1:
                    st.header("Model Performance Overview")
                    st.markdown("---")
                    
                    # Display metrics in columns
                    num_models = len(selected_models)
                    cols = st.columns(num_models)
                    
                    for idx, (model, metrics) in enumerate(comparison_data.items()):
                        with cols[idx]:
                            st.markdown(f"### {model}")
                            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                            st.metric("Precision", f"{metrics['precision']:.2%}")
                            st.metric("Recall", f"{metrics['recall']:.2%}")
                            st.metric("F1 Score", f"{metrics['f1']:.2%}")
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Summary Statistics")
                    summary_stats = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].describe()
                    st.dataframe(summary_stats.style.format("{:.4f}"), use_container_width=True)
                
                with tab2:
                    st.header("📈 Performance Visualizations")
                    st.markdown("---")
                    
                    # Bar chart comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Metric Comparison (Bar Chart)")
                        fig_bar = go.Figure()
                        
                        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                        colors = px.colors.qualitative.Set3[:len(selected_models)]
                        
                        for idx, model in enumerate(selected_models):
                            values = [
                                metrics_df[metrics_df['Model'] == model]['Accuracy'].values[0],
                                metrics_df[metrics_df['Model'] == model]['Precision'].values[0],
                                metrics_df[metrics_df['Model'] == model]['Recall'].values[0],
                                metrics_df[metrics_df['Model'] == model]['F1 Score'].values[0]
                            ]
                            fig_bar.add_trace(go.Bar(
                                name=model,
                                x=metrics_list,
                                y=values,
                                marker_color=colors[idx]
                            ))
                        
                        fig_bar.update_layout(
                            title="Model Performance Metrics Comparison",
                            xaxis_title="Metrics",
                            yaxis_title="Score",
                            yaxis=dict(range=[0, 1]),
                            barmode='group',
                            height=500,
                            showlegend=True
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        st.subheader("Radar Chart Comparison")
                        fig_radar = go.Figure()
                        
                        for idx, model in enumerate(selected_models):
                            values = [
                                metrics_df[metrics_df['Model'] == model]['Accuracy'].values[0],
                                metrics_df[metrics_df['Model'] == model]['Precision'].values[0],
                                metrics_df[metrics_df['Model'] == model]['Recall'].values[0],
                                metrics_df[metrics_df['Model'] == model]['F1 Score'].values[0]
                            ]
                            # Add first value at end for closed polygon
                            values = values + [values[0]]
                            metrics_radar = metrics_list + [metrics_list[0]]
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=metrics_radar,
                                fill='toself',
                                name=model,
                                line_color=colors[idx]
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Model Performance Radar Chart",
                            height=500,
                            showlegend=True
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Line chart for trend
                    st.subheader("Performance Trend")
                    fig_line = px.line(
                        metrics_df,
                        x='Model',
                        y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        markers=True,
                        title="Model Performance Trends",
                        labels={'value': 'Score', 'Model': 'Model Name'}
                    )
                    fig_line.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with tab3:
                    st.header("📋 Detailed Metrics Table")
                    st.markdown("---")
                    
                    # Format the DataFrame for display
                    display_df = metrics_df.copy()
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Metrics as CSV",
                        data=csv,
                        file_name="model_comparison_metrics.csv",
                        mime="text/csv"
                    )
                
                with tab4:
                    st.header("🏆 Model Rankings")
                    st.markdown("---")
                    
                    # Rank by each metric
                    ranking_cols = st.columns(4)
                    metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    
                    for idx, metric in enumerate(metrics_to_rank):
                        with ranking_cols[idx]:
                            st.subheader(f"Top by {metric}")
                            ranked = metrics_df.nlargest(len(selected_models), metric)[['Model', metric]]
                            ranked[metric] = ranked[metric].apply(lambda x: f"{x:.4f}")
                            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                                st.markdown(f"{medal} **{row['Model']}**: {row[metric]}")
                    
                    # Overall ranking (average of all metrics)
                    st.markdown("---")
                    st.subheader("🏆 Overall Ranking (Average Score)")
                    metrics_df['Average Score'] = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean(axis=1)
                    overall_ranked = metrics_df.nlargest(len(selected_models), 'Average Score')[['Model', 'Average Score']]
                    overall_ranked['Average Score'] = overall_ranked['Average Score'].apply(lambda x: f"{x:.4f}")
                    
                    for rank, (_, row) in enumerate(overall_ranked.iterrows(), 1):
                        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                        st.markdown(f"{medal} **{row['Model']}**: {row['Average Score']}")
                        st.progress(float(row['Average Score']))
            else:
                st.error(f"❌ API returned error: {compare_response.status_code}")
                st.error(compare_response.text)
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            st.error("Please ensure the Flask API is running at http://localhost:5000")
else:
    st.info("👈 Please select models from the sidebar to begin comparison")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Fake News Detection Model Comparison Dashboard | Powered by Streamlit & Flask API"
    "</div>",
    unsafe_allow_html=True
)

