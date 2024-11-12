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

def is_valid_doi_format(doi):
    pattern = r'^10.\d{4,9}/[-._;()/:A-Z0-9]+$'
    return re.match(pattern, doi, re.IGNORECASE) is not None

def validate_doi(doi):
    if not is_valid_doi_format(doi):
        return False
    url = f"https://doi.org/{doi}"
    response = retry_request(url, method='head')
    return response is not None

def validate_url(url):
    response = retry_request(url, method='head')
    return response is not None

def is_metadata_complete(article):
    essential_fields = ['author', 'title', 'issued']
    missing_fields = [field for field in essential_fields if field not in article or not article[field]]
    if missing_fields:
        logging.warning(f"Missing fields: {missing_fields}")
    return len(missing_fields) <= st.session_state.VALIDATION_STRICTNESS

def format_authors_apa(authors):
    authors_list = []
    for author in authors:
        last_name = author.get('family', '')
        initials = ''.join([name[0] + '.' for name in author.get('given', '').split()])
        authors_list.append(f"{last_name}, {initials}")
    
    if not authors_list:
        return "Anonymous"
    elif len(authors_list) == 1:
        return authors_list[0]
    elif len(authors_list) <= 20:
        return ', '.join(authors_list[:-1]) + ', & ' + authors_list[-1]
    else:
        return ', '.join(authors_list[:19]) + ', ... ' + authors_list[-1]

def format_citation(article, style="APA"):
    if not article:
        return None
    
    authors = article.get('author', [])
    authors_str = format_authors_apa(authors)
    
    year = article.get('published-print', {}).get('date-parts', [[None]])[0][0]
    if not year:
        year = article.get('published-online', {}).get('date-parts', [[None]])[0][0]
    if not year:
        year = article.get('issued', {}).get('date-parts', [[None]])[0][0]
    if not year:
        year = 'n.d.'
        
    title = article.get('title', [''])[0]
    journal = article.get('container-title', [''])[0]
    doi = article.get('DOI', '')
    
    citation = f"{authors_str} ({year}). {title}"
    if journal:
        citation += f". {journal}"
    if doi:
        citation += f". https://doi.org/{doi}"
    
    return citation

@st.cache_data(show_spinner=False)
def search_articles(query):
    try:
        response = requests.get(
            f"https://api.crossref.org/works?query={query}&rows=10",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("items", [])
    except Exception as e:
        st.error(f"Error fetching articles: {str(e)}")
        return []

def process_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {'User', 'Category', 'Prompt', 'Response'}
        if not required_columns.issubset(df.columns):
            st.error(f"Required columns: {required_columns}")
            return None
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def generate_insight(user, category, prompt_text, response_text, knowledge_base, context, temperature, max_tokens):
    system_prompt = f"{knowledge_base}\n\n{context}"
    user_prompt = f"""
    Given the following information:
    User: {user}
    Category: {category}
    Prompt: {prompt_text}
    Response: {response_text}
    Generate a concise, meaningful insight based on this information.
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
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: Rate limit exceeded"

def main():
    st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
    
    tab1, tab2 = st.tabs(["Insight Report Assistant", "Auto Citation Tool"])

    with tab1:
        st.header("Insight Report Assistant")
        
        with st.sidebar:
            st.title("Configuration")
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
            max_tokens = st.number_input("Max Tokens:", 50, 2048, 150)

        st.subheader("Report Guidelines")
        knowledge_base_input = st.text_area(
            "Guidelines:",
            value=DEFAULT_KNOWLEDGE_BASE,
            height=200
        )

        st.subheader("Additional Context")
        context_input = st.text_area(
            "Context:",
            height=100
        )

        st.subheader("Data Upload")
        sample_data = "User,Category,Prompt,Response\nJohn Doe,Sales,How did sales perform?,Sales increased 15%"
        
        st.download_button(
            "Download Sample CSV",
            sample_data,
            "sample.csv",
            "text/csv"
        )

        uploaded_file = st.file_uploader("Upload CSV", type="csv")

        if uploaded_file:
            df = process_csv(uploaded_file)
            if df is not None:
                st.success("CSV uploaded successfully!")
                st.dataframe(df)

                if st.button("Generate Insights"):
                    insights = []
                    progress = st.progress(0)
                    status = st.empty()

                    for idx, row in df.iterrows():
                        status.text(f"Processing {idx + 1}/{len(df)}")
                        insight = generate_insight(
                            row['User'],
                            row['Category'],
                            row['Prompt'],
                            row['Response'],
                            knowledge_base_input,
                            context_input,
                            temperature,
                            max_tokens
                        )
                        insights.append(insight)
                        progress.progress((idx + 1) / len(df))

                    df['Generated Insight'] = insights
                    status.text("Processing complete!")
                    st.success("Insights generated!")
                    st.dataframe(df)

                    # Export options
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False).encode('utf-8'),
                        "insights_report.csv",
                        "text/csv"
                    )

    with tab2:
        st.header("Auto Citation Tool")
        
        with st.sidebar:
            validation_level = st.radio(
                "Validation Level:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Strict", 2: "Moderate", 3: "Lenient"}[x]
            )
            st.session_state.VALIDATION_STRICTNESS = validation_level

        query = st.text_input("Search Query:")
        citation_style = st.selectbox("Citation Style:", ["APA", "MLA", "Chicago"])

        if st.button("Search"):
            if query:
                with st.spinner('Searching...'):
                    articles = search_articles(query)
                    if articles:
                        st.write("### Results:")
                        selected_articles = []
                        
                        for idx, article in enumerate(articles):
                            title = article.get('title', ['No title'])[0]
                            st.write(f"**{idx+1}. {title}**")
                            
                            if st.checkbox("Include", key=f"include_{idx}"):
                                citation = format_citation(article, citation_style)
                                if citation:
                                    st.write(citation)
                                    selected_articles.append(citation)
                                else:
                                    st.error(f"Couldn't generate citation for: {title}")

                        if selected_articles:
                            st.write("### Bibliography:")
                            for cite in selected_articles:
                                st.write(cite)
                    else:
                        st.info("No results found")
            else:
                st.write("Please enter a search query")

if __name__ == "__main__":
    main()
