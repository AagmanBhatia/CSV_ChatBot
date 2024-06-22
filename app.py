import streamlit as st
import pandas as pd
import json
import tempfile
import plotly.express as px  # Import Plotly

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Ask your CSV",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.markdown(
        """
        <style>
        .main-header {
            font-size:50px;
            font-family: 'Arial', sans-serif;
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2E7D32,#A5D6A7);
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-header">ASK YOUR CSV</h1>', unsafe_allow_html=True)
    st.sidebar.header("Upload and Ask")

    user_csv = st.sidebar.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        df = pd.read_csv(user_csv)
        
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a file with data.")
        else:
            st.subheader("Preview of your data:")
            st.dataframe(df.head())

            user_question = st.text_input("Ask a question about your CSV")

            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(user_csv.getvalue())
                tmp_file_path = tmp_file.name

            llm = ChatGoogleGenerativeAI(temperature=0.3, model='gemini-pro')
            agent = create_csv_agent(llm, tmp_file_path, verbose=True, handle_parsing_errors=True)

            if user_question:
                response = agent.run(user_question)
                st.markdown("### Response")
                st.write(response)

                if 'plot' in user_question.lower() or 'visualization' in user_question.lower():
                    generate_plot(user_question, df)

def generate_plot(user_question, df):
    plot_type = None
    if 'bar' in user_question.lower():
        plot_type = "bar"
    elif 'line' in user_question.lower():
        plot_type = "line"
    elif 'histogram' in user_question.lower():
        plot_type = "histogram"
    elif 'pie' in user_question.lower():
        plot_type = "pie"

    if plot_type:
        if plot_type == "bar":
            generate_bar_plot(df)
        elif plot_type == "line":
            generate_line_plot(df)
        elif plot_type == "histogram":
            generate_histogram(df)
        elif plot_type == "pie":
            generate_pie_chart(df)

def generate_bar_plot(df):
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select the column for the bar plot:", columns, key="bar_column")
    if selected_column:
        fig = px.bar(df.head(5), x='title', y=selected_column, title=f'Bar Plot of Top 5 Movies by {selected_column}')
        st.plotly_chart(fig)
        st.experimental_rerun()

def generate_line_plot(df):
    columns = df.columns.tolist()
    x_column = st.selectbox("Select the X-axis column for the line plot:", columns, key="line_x_column")
    y_column = st.selectbox("Select the Y-axis column for the line plot:", columns, key="line_y_column")
    if x_column and y_column:
        fig = px.line(df, x=x_column, y=y_column, title=f'Line Plot of {y_column} over {x_column}')
        st.plotly_chart(fig)
        st.experimental_rerun()

def generate_histogram(df):
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select the column for the histogram:", columns, key="hist_column")
    if selected_column:
        fig = px.histogram(df, x=selected_column, title=f'Histogram of {selected_column}')
        st.plotly_chart(fig)
        st.experimental_rerun()

def generate_pie_chart(df):
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select the column for the pie chart:", columns, key="pie_column")
    if selected_column:
        fig = px.pie(df, names=selected_column, title=f'Pie Chart of {selected_column}')
        st.plotly_chart(fig)
        st.experimental_rerun()

def parse_json_column(df, column_name):
    """
    Parse a column in a dataframe that contains JSON strings.
    """
    def parse_json(x):
        try:
            return json.loads(x.replace("'", "\""))  # Ensure proper JSON format
        except (json.JSONDecodeError, TypeError):
            return []

    df[column_name] = df[column_name].apply(parse_json)
    return df

if __name__ == "__main__":
    main()













