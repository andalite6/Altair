import streamlit as st
import pandas as pd
import openai
import logging
import time
import re
import requests
import os
from datetime import datetime
import difflib
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)

# Setup OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error("Please set up OpenAI API key in secrets")
    st.stop()

# Constants
DEFAULT_KNOWLEDGE_BASE = """
Insight report writing involves analyzing data to extract meaningful patterns and actionable
information. Key aspects include:
1. Clarity: Present findings in a clear, concise manner.
2. Relevance: Focus on insights that are relevant to the business or research question.
3. Data-driven: Back up insights with data and evidence.
4. Actionable: Provide recommendations or next steps based on the insights.
5. Visual aids: Use charts, graphs, or tables to illustrate key points.
6. Structure: Organize the report with an executive summary, main findings, and conclusion.
7. Context: Explain the significance of the insights in the broader context.
8. Objectivity: Present unbiased analysis, acknowledging limitations or uncertainties.
"""

# Global Variables and Configurations
if 'VALIDATION_STRICTNESS' not in st.session_state:
    st.session_state.VALIDATION_STRICTNESS = 2
CACHE_SIZE = 128

# Citation Helper Functions
@lru_cache(maxsize=CACHE_SIZE)
def is_valid_doi_format(doi):
    pattern = r'^10.\d{4,9}/[-._;()/:A-Z0-9]+$'
    return re.match(pattern, doi, re.IGNORECASE) is not None

def retry_request(url, method='head', retries=3, timeout=5):
    for attempt in range(retries):
        try:
            if method == 'head':
                response = requests.head(url, allow_redirects=True, timeout=timeout)
            elif method == 'get':
                response = requests.get(url, allow_redirects=True, timeout=timeout)
            else:
                return None
            if response.status_code == 200:
                return response
        except requests.RequestException as e:
            logging.error(f"Network error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None

[... rest of citation helper functions from original code ...]

# Main Functions
def process_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {'User', 'Category', 'Prompt', 'Response'}
        if not required_columns.issubset(df.columns):
            st.error(f"The CSV file must contain the following columns: {required_columns}")
            return None
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None

def generate_insight(user, category, prompt_text, response_text, knowledge_base, context, temperature, max_tokens):
    system_prompt = f"{knowledge_base}\n\n{context}"
    user_prompt = f"""
    Given the following information:
    User: {user}
    Category: {category}
    Prompt: {prompt_text}
    Response: {response_text}
    Generate a concise, meaningful insight based on this information. The insight should be
    relevant, data-driven, and actionable.
    """
    
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.RateLimitError:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return "Rate limit exceeded. Please try again later."
        except Exception as e:
            logging.error(f"Error generating insight: {e}")
            return f"Error generating insight: {str(e)}"

def main():
    st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Insight Report Assistant", "Auto Citation Tool"])

    with tab1:
        st.header("Insight Report Assistant")
        
        # Configuration
        with st.sidebar:
            st.title("Configuration")
            temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
            max_tokens = st.number_input("Max Tokens:", min_value=50, max_value=2048, value=150)

        [... rest of tab1 implementation from original code ...]

    with tab2:
        st.header("Auto Citation Tool")
        
        # Citation Tool Settings
        with st.sidebar:
            st.title("Citation Settings")
            validation_level = st.radio(
                "Validation Strictness:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Strict", 2: "Moderate", 3: "Lenient"}[x]
            )
            st.session_state.VALIDATION_STRICTNESS = validation_level

        [... rest of tab2 implementation from original code ...]

if __name__ == "__main__":
    main()
