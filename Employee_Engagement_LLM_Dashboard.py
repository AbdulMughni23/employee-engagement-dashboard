import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Employee Engagement LLM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# LOAD AND INITIALIZE LLM
# ===============================================================

@st.cache_resource
def load_llm_model():
    """Load a small Hugging Face model for generating insights."""
    # We're using a small model that can run locally
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================

def load_data():
    """Load the analysis data."""
    try:
        # Load survey data
        survey_data = pd.read_csv('output/data/survey_data_clean.csv')
        
        # Load synthesis results
        with open('output/synthesis_results.json', 'r') as f:
            synthesis = json.load(f)
        
        # Load executive summary and recommendations
        with open('output/executive_summary.md', 'r') as f:
            exec_summary = f.read()
        
        with open('output/recommendations.md', 'r') as f:
            recommendations = f.read()
        
        return {
            'survey_data': survey_data,
            'synthesis': synthesis,
            'executive_summary': exec_summary,
            'recommendations': recommendations
        }
    except FileNotFoundError:
        st.warning("Analysis data files not found. Running with sample data.")
        # Generate sample data if files don't exist
        from employee_engagement_analysis import generate_synthetic_survey_data, run_employee_engagement_analysis
        
        # Run the full analysis to generate all required files
        results = run_employee_engagement_analysis()
        return results

def generate_llm_response(llm_pipe, query, context):
    """Generate a response using the LLM model with context from the analysis.
    
    Parameters:
    ----------
    llm_pipe : pipeline
        Hugging Face pipeline for text generation
    query : str
        User's query
    context : str
        Context from the engagement analysis
        
    Returns:
    -------
    str
        Generated response
    """
    if llm_pipe is None:
        return "LLM model not available. Please check the error message."
    
    # Format the prompt with context and query
    prompt = f"""<|system|>
You are an AI assistant specialized in employee engagement analysis. Use the following context to answer the user's query thoughtfully.

CONTEXT:
{context}
<|endoftext|>

<|user|>
{query}
<|endoftext|>

<|assistant|>"""
    
    try:
        # Generate response
        response = llm_pipe(prompt)[0]['generated_text']
        
        # Extract just the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        return assistant_response
    except Exception as e:
        return f"Error generating response: {e}"

def create_department_context(data, dept_name):
    """Create context about a specific department for the LLM.
    
    Parameters:
    ----------
    data : dict
        Analysis data
    dept_name : str
        Department name
        
    Returns:
    -------
    str
        Context string about the department
    """
    survey_data = data['survey_data']
    dept_data = survey_data[survey_data['department'] == dept_name]
    
    engagement_cols = ['job_satisfaction', 'workload', 'autonomy', 'recognition', 
                      'career_development', 'work_life_balance', 'leadership_trust', 
                      'team_collaboration', 'resources_support', 'overall_engagement']
    
    # Calculate department metrics
    dept_metrics = {}
    for col in engagement_cols:
        dept_avg = dept_data[col].mean()
        overall_avg = survey_data[col].mean()
        diff = dept_avg - overall_avg
        dept_metrics[col] = {
            'score': dept_avg,
            'overall_avg': overall_avg,
            'difference': diff
        }
    
    # Identify top strengths and challenges
    sorted_metrics = sorted(dept_metrics.items(), key=lambda x: x[1]['difference'], reverse=True)
    strengths = sorted_metrics[:3]
    challenges = sorted_metrics[-3:]
    
    # Format context
    context = f"""
Department Profile: {dept_name}

Overall engagement score: {dept_metrics['overall_engagement']['score']:.2f} (University average: {dept_metrics['overall_engagement']['overall_avg']:.2f})

Number of respondents: {len(dept_data)}

Top Strengths:
"""
    
    for metric, values in strengths:
        metric_name = metric.replace('_', ' ').title()
        context += f"- {metric_name}: {values['score']:.2f} (University avg: {values['overall_avg']:.2f}, Difference: {values['difference']:.2f})\n"
    
    context += "\nTop Challenges:\n"
    
    for metric, values in challenges:
        metric_name = metric.replace('_', ' ').title()
        context += f"- {metric_name}: {values['score']:.2f} (University avg: {values['overall_avg']:.2f}, Difference: {values['difference']:.2f})\n"
    
    # Add recommendations context
    context += "\nGeneral recommendations:\n"
    context += "1. Focus on addressing the challenge areas identified above\n"
    context += "2. Build on existing strengths\n"
    context += "3. Implement regular feedback mechanisms\n"
    context += "4. Develop tailored interventions for specific issues\n"
    
    return context

def create_overall_context(data):
    """Create overall context from the analysis for the LLM.
    
    Parameters:
    ----------
    data : dict
        Analysis data
        
    Returns:
    -------
    str
        Context string about the overall analysis
    """
    survey_data = data['survey_data']
    synthesis = data['synthesis']
    
    # Extract key metrics
    overall_engagement = survey_data['overall_engagement'].mean()
    
    # Top factors from both methods
    top_quant_factors = synthesis['top_quantitative_factors']
    top_qual_themes = synthesis['top_qualitative_themes']
    
    # Department insights
    dept_insights = []
    for insight in synthesis['department_insights'][:3]:
        direction = "higher" if insight['difference'] > 0 else "lower"
        dept_insights.append(
            f"{insight['department']} ({insight['variable']} is {direction} at {insight['score']:.2f} vs. avg {insight['avg_score']:.2f})"
        )
    
    # Format context
    context = f"""
Overall Employee Engagement Analysis Summary:

Average engagement score: {overall_engagement:.2f}
Number of respondents: {len(survey_data)}

Top Quantitative Factors:
{', '.join(f"{factor}" for factor in top_quant_factors[:3])}

Top Qualitative Themes:
{', '.join(theme for theme in top_qual_themes[:3])}

Department Highlights:
{chr(10).join(f"- {insight}" for insight in dept_insights)}

Key Alignments and Contradictions:
- Alignments: {', '.join(align['theme'] for align in synthesis['alignments'])}
- Contradictions: {', '.join(contra['theme'] for contra in synthesis['contradictions'])}

Main Recommendations:
1. Address the top engagement factors identified
2. Develop department-specific interventions
3. Create monitoring and feedback systems
4. Implement a structured improvement plan
"""
    
    return context

def create_comparison_context(data, depts_to_compare):
    """Create context for comparing departments.
    
    Parameters:
    ----------
    data : dict
        Analysis data
    depts_to_compare : list
        List of department names to compare
        
    Returns:
    -------
    str
        Context string comparing departments
    """
    survey_data = data['survey_data']
    
    # Filter data for selected departments
    dept_data = survey_data[survey_data['department'].isin(depts_to_compare)]
    
    engagement_cols = ['job_satisfaction', 'workload', 'autonomy', 'recognition', 
                      'career_development', 'work_life_balance', 'leadership_trust', 
                      'team_collaboration', 'resources_support', 'overall_engagement']
    
    # Calculate metrics for each department
    dept_metrics = {}
    for dept in depts_to_compare:
        dept_slice = dept_data[dept_data['department'] == dept]
        dept_metrics[dept] = {}
        
        for col in engagement_cols:
            dept_metrics[dept][col] = dept_slice[col].mean()
    
    # Find largest differences between departments
    diff_metrics = []
    for col in engagement_cols:
        max_val = max(dept_metrics[dept][col] for dept in depts_to_compare)
        min_val = min(dept_metrics[dept][col] for dept in depts_to_compare)
        max_dept = [dept for dept in depts_to_compare if dept_metrics[dept][col] == max_val][0]
        min_dept = [dept for dept in depts_to_compare if dept_metrics[dept][col] == min_val][0]
        diff = max_val - min_val
        
        diff_metrics.append({
            'metric': col,
            'max_dept': max_dept,
            'max_val': max_val,
            'min_dept': min_dept,
            'min_val': min_val,
            'diff': diff
        })
    
    # Sort by largest differences
    diff_metrics.sort(key=lambda x: x['diff'], reverse=True)
    
    # Format context
    context = f"""
Department Comparison: {', '.join(depts_to_compare)}

Overall Engagement Scores:
"""
    
    for dept in depts_to_compare:
        context += f"- {dept}: {dept_metrics[dept]['overall_engagement']:.2f}\n"
    
    context += "\nLargest Differences Between Departments:\n"
    
    for i, metric in enumerate(diff_metrics[:5]):
        metric_name = metric['metric'].replace('_', ' ').title()
        context += f"{i+1}. {metric_name}: Difference of {metric['diff']:.2f}\n"
        context += f"   - Highest: {metric['max_dept']} ({metric['max_val']:.2f})\n"
        context += f"   - Lowest: {metric['min_dept']} ({metric['min_val']:.2f})\n"
    
    context += "\nRecommendations for Improvement:\n"
    context += "1. Knowledge sharing between high and low-performing departments\n"
    context += "2. Identify and transfer best practices\n"
    context += "3. Address specific gap areas through targeted interventions\n"
    context += "4. Create opportunities for cross-department collaboration\n"
    
    return context

def create_visualizations(df, quant_results, qual_results, synthesis):
    """Create visualizations from the data."""
    # Set color palette
    colors = px.colors.qualitative.Plotly
    
    # 1. Overall engagement by department
    dept_data = df.groupby('department')['overall_engagement'].mean().reset_index()
    dept_data = dept_data.sort_values('overall_engagement', ascending=False)
    
    fig1 = px.bar(
        dept_data, 
        x='department', 
        y='overall_engagement',
        title='Average Engagement by Department',
        labels={'department': 'Department', 'overall_engagement': 'Engagement Score'},
        color='overall_engagement',
        color_continuous_scale=px.colors.sequential.Viridis,
        height=500
    )
    
    fig1.add_hline(
        y=df['overall_engagement'].mean(), 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Overall Avg: {df['overall_engagement'].mean():.2f}"
    )
    
    fig1.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[1, 5],
        coloraxis_showscale=False
    )
    
    # 2. Engagement factors by role
    roles = df['role'].unique()
    engagement_cols = ['job_satisfaction', 'workload', 'autonomy', 'recognition', 
                      'career_development', 'work_life_balance', 'leadership_trust', 
                      'team_collaboration', 'resources_support']
    
    role_means = df.groupby('role')[engagement_cols].mean()
    
    # Create radar chart
    fig2 = go.Figure()
    
    for i, role in enumerate(role_means.index):
        values = role_means.loc[role].tolist()
        # Add the first value again to close the polygon
        values.append(values[0])
        
        fig2.add_trace(go.Scatterpolar(
            r=values,
            theta=[col.replace('_', ' ').title() for col in engagement_cols] + [engagement_cols[0].replace('_', ' ').title()],
            fill='toself',
            name=role,
            line_color=colors[i % len(colors)]
        ))
    
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5]
            )
        ),
        title='Engagement Factors by Role',
        height=600,
        showlegend=True
    )
    
    # 3. Theme frequencies
    theme_data = []
    theme_counts = qual_results.get('theme_counts', {})
    
    if theme_counts:
        for theme, count in theme_counts.items():
            theme_data.append({
                'Theme': theme,
                'Count': count
            })
        
        theme_df = pd.DataFrame(theme_data)
        theme_df = theme_df.sort_values('Count', ascending=False)
        
        fig3 = px.bar(
            theme_df,
            x='Theme',
            y='Count',
            title='Qualitative Theme Frequency',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig3.update_layout(
            xaxis_tickangle=-45,
            coloraxis_showscale=False
        )
    else:
        fig3 = go.Figure()
        fig3.add_annotation(
            text="Qualitative theme data not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    return fig1, fig2, fig3

# ===============================================================
# STREAMLIT UI
# ===============================================================

def main():
    """Main dashboard function."""
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #EFF6FF;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .llm-response {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.9rem;
        color: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<div class='main-header'>Employee Engagement LLM Dashboard</div>", unsafe_allow_html=True)
    
    # Load data
    try:
        data = load_data()
        survey_data = data['survey_data']
        synthesis = data['synthesis']
        recommendations = data['recommendations']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.warning("Make sure to run the analysis script first to generate the required data files.")
        return
    
    # Initialize LLM model
    with st.spinner("Loading LLM model... This may take a moment."):
        llm_pipe = load_llm_model()
    
    if llm_pipe is None:
        st.warning("LLM model could not be loaded. Some features will be limited.")
    
    # SIDEBAR
    st.sidebar.markdown("## Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Department Analysis", "Role Comparison", "LLM Insights", "Custom Query"]
    )
    
    # MAIN CONTENT
    if page == "Overview":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{survey_data['overall_engagement'].mean():.2f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Avg Engagement Score</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{len(survey_data)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Respondents</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            above_avg = (survey_data['overall_engagement'] > 3.5).sum()
            pct_above = (above_avg / len(survey_data)) * 100
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{pct_above:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Above Average Engagement</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Key insights
        st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
        
        if 'key_insights' in synthesis:
            for i, insight in enumerate(synthesis['key_insights']):
                st.markdown(f"**{i+1}.** {insight}")
        
        # Create visualizations
        fig1, fig2, fig3 = create_visualizations(
            survey_data, 
            {'regression': {'coefficients': [{'Variable': v} for v in synthesis['top_quantitative_factors']]}}, 
            {'theme_counts': {t: i for i, t in enumerate(synthesis['top_qualitative_themes'], 1)}},
            synthesis
        )
        
        # Department engagement
        st.markdown("<div class='section-header'>Department Engagement</div>", unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # LLM insights
        st.markdown("<div class='section-header'>LLM-Generated Insights</div>", unsafe_allow_html=True)
        
        if llm_pipe is not None:
            # Create context
            context = create_overall_context(data)
            
            # Generate insights
            query = "Provide 3 key insights about the employee engagement analysis results and what they might mean for the university."
            llm_response = generate_llm_response(llm_pipe, query, context)
            
            st.markdown("<div class='llm-response'>", unsafe_allow_html=True)
            st.markdown(llm_response)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("LLM model not available. Cannot generate insights.")
    
    elif page == "Department Analysis":
        st.markdown("<div class='section-header'>Department Analysis</div>", unsafe_allow_html=True)
        
        # Department selector
        departments = sorted(survey_data['department'].unique())
        selected_dept = st.selectbox("Select Department", departments)
        
        if selected_dept:
            # Filter data for the selected department
            dept_data = survey_data[survey_data['department'] == selected_dept]
            
            # Department metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{dept_data['overall_engagement'].mean():.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Dept Engagement Score</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                univ_avg = survey_data['overall_engagement'].mean()
                diff = dept_data['overall_engagement'].mean() - univ_avg
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{diff:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Vs. University Avg</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{len(dept_data)}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Respondents</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Department engagement factors
            engagement_cols = ['job_satisfaction', 'workload', 'autonomy', 'recognition', 
                              'career_development', 'work_life_balance', 'leadership_trust', 
                              'team_collaboration', 'resources_support']
            
            col_labels = [col.replace('_', ' ').title() for col in engagement_cols]
            
            # Compare department to overall average
            compare_data = []
            
            for col, label in zip(engagement_cols, col_labels):
                dept_avg = dept_data[col].mean()
                univ_avg = survey_data[col].mean()
                compare_data.append({
                    'Factor': label,
                    'Department': dept_avg,
                    'University Avg': univ_avg,
                    'Difference': dept_avg - univ_avg
                })
            
            compare_df = pd.DataFrame(compare_data)
            compare_df = compare_df.sort_values('Difference', ascending=False)
            
            # Create a horizontal bar chart showing differences
            fig = px.bar(
                compare_df,
                y='Factor',
                x='Difference',
                title=f'Engagement Factors: {selected_dept} vs. University Average',
                color='Difference',
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0,
                orientation='h'
            )
            
            fig.update_layout(
                height=500,
                xaxis_title='Difference from University Average',
                yaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Department strengths and challenges
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Department Strengths")
                strengths = compare_df.head(3)
                for i, row in strengths.iterrows():
                    st.markdown(f"**{row['Factor']}**: {row['Department']:.2f} (Univ: {row['University Avg']:.2f}, Diff: {row['Difference']:.2f})")
            
            with col2:
                st.markdown("#### Department Challenges")
                challenges = compare_df.tail(3).iloc[::-1]  # Reverse to show worst first
                for i, row in challenges.iterrows():
                    st.markdown(f"**{row['Factor']}**: {row['Department']:.2f} (Univ: {row['University Avg']:.2f}, Diff: {row['Difference']:.2f})")
            
            # LLM-powered department insights
            if llm_pipe is not None:
                st.markdown("<div class='section-header'>LLM-Generated Department Insights</div>", unsafe_allow_html=True)
                
                # Create context for the selected department
                dept_context = create_department_context(data, selected_dept)
                
                # Generate insights
                query = f"Based on the data, what are the most important insights about the {selected_dept} department, and what specific actions could improve employee engagement in this department?"
                llm_response = generate_llm_response(llm_pipe, query, dept_context)
                
                st.markdown("<div class='llm-response'>", unsafe_allow_html=True)
                st.markdown(llm_response)
                st.markdown("</div>", unsafe_allow_html=True)
    
    elif page == "Role Comparison":
        st.markdown("<div class='section-header'>Role Comparison</div>", unsafe_allow_html=True)
        
        # Role selector
        roles = sorted(survey_data['role'].unique())
        selected_roles = st.multiselect("Select Roles to Compare", roles, default=roles)
        
        if selected_roles:
            # Filter data for selected roles
            role_data = survey_data[survey_data['role'].isin(selected_roles)]
            
            # Role engagement comparison
            role_engagement = role_data.groupby('role')['overall_engagement'].mean().reset_index()
            role_engagement = role_engagement.sort_values('overall_engagement', ascending=False)
            
            fig = px.bar(
                role_engagement,
                x='role',
                y='overall_engagement',
                title='Overall Engagement by Role',
                color='overall_engagement',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            fig.add_hline(
                y=survey_data['overall_engagement'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Overall Avg: {survey_data['overall_engagement'].mean():.2f}"
            )
            
            fig.update_layout(
                height=400,
                yaxis_range=[1, 5],
                coloraxis_showscale=False,
                xaxis_title=None,
                yaxis_title='Engagement Score'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart for selected roles
            engagement_cols = ['job_satisfaction', 'workload', 'autonomy', 'recognition', 
                              'career_development', 'work_life_balance', 'leadership_trust', 
                              'team_collaboration', 'resources_support']
            
            radar_fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            role_means = role_data.groupby('role')[engagement_cols].mean()
            
            for i, role in enumerate(role_means.index):
                values = role_means.loc[role].tolist()
                # Add the first value again to close the polygon
                values.append(values[0])
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[col.replace('_', ' ').title() for col in engagement_cols] + [engagement_cols[0].replace('_', ' ').title()],
                    fill='toself',
                    name=role,
                    line_color=colors[i % len(colors)]
                ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[1, 5]
                    )
                ),
                title='Engagement Factors by Role',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Key differences between roles
            if len(selected_roles) > 1:
                st.markdown("<div class='section-header'>Key Differences Between Roles</div>", unsafe_allow_html=True)
