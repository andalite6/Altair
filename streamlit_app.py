import streamlit
import pandas as pd
import openai
import logging
import time
import re
import requests
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import psutil
import difflib
from functools import lru_cache

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY') or "your_openai_api_key_here"

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DEFAULT_KNOWLEDGE_BASE = """
Insight report writing involves analyzing data to extract meaningful patterns and actionable information. Key aspects include:
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
VALIDATION_STRICTNESS = 2
CACHE_SIZE = 128

# Security Manager
class SecurityManager:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY') or "your-generated-key-here"
        if not self.key:
            raise ValueError("ENCRYPTION_KEY environment variable is not set.")
        self.cipher_suite = Fernet(self.key)

    def encrypt_data(self, data: bytes) -> bytes:
        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        return self.cipher_suite.decrypt(encrypted_data)

# Metrics Collector
class MetricsCollector:
    def __init__(self):
        pass  # Placeholder for metrics collection logic

    def update_memory_usage(self):
        memory = psutil.virtual_memory().used
        logging.info(f"Memory usage: {memory} bytes")

# Continual Learning LLM
class ContinualLearningLLM:
    def __init__(self, base_model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        self.learning_rate = 1e-5
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def preprocess_input(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return tokens.to(self.device)

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.preprocess_input(prompt)
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# AI Quality Control Bot
class AIQualityControlBot:
    def __init__(self):
        pass  # Placeholder for AI quality control logic

# Smart Citation Manager
class SmartCitationManager:
    def __init__(self):
        self.citations = {}
        self.citation_patterns = {}
        self.load_citations()

    def load_citations(self):
        try:
            with open('citation_database.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                self.citations = saved_data.get('citations', {})
                self.citation_patterns = saved_data.get('patterns', {})
        except FileNotFoundError:
            pass

    def save_citations(self):
        with open('citation_database.pkl', 'wb') as f:
            pickle.dump({'citations': self.citations, 'patterns': self.citation_patterns}, f)

    def get_citation_data(self, query: str) -> dict:
        try:
            response = requests.get(f"https://api.crossref.org/works?query={query}&rows=5", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("message", {}).get("items"):
                items = data["message"]["items"]
                scored_items = [(self._calculate_relevance(query, item), item) for item in items]
                best_score, item = max(scored_items, key=lambda x: x[0])
                return item
            return None
        except Exception as e:
            logging.error(f"Citation fetch error: {str(e)}")
            return None

    def _calculate_relevance(self, query: str, item: dict) -> float:
        score = 0.0
        if "title" in item:
            title = item["title"][0].lower()
            query_terms = query.lower().split()
            score += sum(term in title for term in query_terms) / len(query_terms)
        year = item.get("created", {}).get("date-parts", [[0]])[0][0]
        if year:
            score += min(1.0, (float(year) - 2000) / 25) * 0.3
        completeness = sum(1 for field in ["author", "DOI", "abstract"] if field in item) / 3
        score += completeness * 0.2
        return score

# Auto Citation Tool Functions

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

ESSENTIAL_FIELDS = ['author', 'title', 'issued']

def is_metadata_complete(article):
    missing_fields = [field for field in ESSENTIAL_FIELDS if field not in article or not article[field]]
    if missing_fields:
        logging.warning(f"Missing fields: {missing_fields}")
    # Allow some flexibility based on validation strictness
    if VALIDATION_STRICTNESS == 1:  # Strict
        return not missing_fields
    elif VALIDATION_STRICTNESS == 2:  # Moderate
        return len(missing_fields) <= 1
    else:  # Lenient
        return True

@lru_cache(maxsize=CACHE_SIZE)
def cross_verify_article(doi, original_title):
    if not doi:
        return False
    try:
        response = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        response.raise_for_status()
        crossref_article = response.json().get('message', {})
        if not crossref_article:
            return False
        crossref_title = crossref_article.get('title', [''])[0]
        # Normalize titles
        original_title_normalized = re.sub(r'\s+', ' ', original_title.lower().strip())
        crossref_title_normalized = re.sub(r'\s+', ' ', crossref_title.lower().strip())
        # Use fuzzy matching
        similarity = difflib.SequenceMatcher(None, original_title_normalized, crossref_title_normalized).ratio()
        logging.info(f"Title similarity: {similarity}")
        # Adjust threshold based on validation strictness
        threshold = 0.9 if VALIDATION_STRICTNESS == 1 else 0.7
        return similarity >= threshold
    except Exception as e:
        logging.error(f"Cross-verification error for DOI {doi}: {e}")
        return False

def is_title_relevant(title, query):
    title_tokens = set(re.sub(r'\W+', ' ', title.lower()).split())
    query_tokens = set(re.sub(r'\W+', ' ', query.lower()).split())
    common_tokens = title_tokens & query_tokens
    # Calculate relevance score
    relevance_score = len(common_tokens) / len(query_tokens)
    # Adjust threshold based on validation strictness
    threshold = 0.6 if VALIDATION_STRICTNESS == 1 else 0.3
    return relevance_score >= threshold

@lru_cache(maxsize=CACHE_SIZE)
def is_publisher_reputable(publisher):
    reputable_publishers = [
        'springer', 'elsevier', 'ieee', 'acm', 'nature', 'science',
        'wiley', 'taylor & francis', 'sage publications',
        'oxford university press', 'cambridge university press'
    ]
    publisher_lower = publisher.lower()
    for reputable in reputable_publishers:
        if reputable in publisher_lower:
            return True
    return VALIDATION_STRICTNESS >= 3  # Allow if lenient

def is_arxiv_article(url):
    return 'arxiv.org' in url

@lru_cache(maxsize=CACHE_SIZE)
def validate_arxiv_id(url):
    arxiv_id_match = re.search(r'arxiv\.org/abs/([^\s]+)', url)
    if arxiv_id_match:
        arxiv_id = arxiv_id_match.group(1)
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = retry_request(api_url, method='get')
        return response is not None
    return False

# Author formatting functions for APA, MLA, and Chicago styles

def format_authors_apa(authors):
    authors_list = []
    for author in authors:
        last_name = author.get('family', '')
        initials = ''.join([name[0] + '.' for name in author.get('given', '').split()])
        authors_list.append(f"{last_name}, {initials}")
    if not authors_list:
        return "Anonymous"
    elif len(authors_list) <= 20:
        if len(authors_list) == 1:
            return authors_list[0]
        else:
            return ', '.join(authors_list[:-1]) + ', & ' + authors_list[-1]
    else:
        first_19 = ', '.join(authors_list[:19])
        last_author = authors_list[-1]
        return f"{first_19}, ..., {last_author}"

def format_authors_mla(authors):
    authors_list = []
    for i, author in enumerate(authors):
        last_name = author.get('family', '')
        given_names = author.get('given', '')
        if i == 0:
            authors_list.append(f"{last_name}, {given_names}")
        else:
            authors_list.append(f"{given_names} {last_name}")
    if not authors_list:
        return "Anonymous"
    elif len(authors_list) == 1:
        return authors_list[0]
    elif len(authors_list) <= 3:
        return ', and '.join(authors_list)
    else:
        return f"{authors_list[0]}, et al."

def format_authors_chicago(authors):
    authors_list = []
    for author in authors:
        last_name = author.get('family', '')
        given_names = author.get('given', '')
        authors_list.append(f"{given_names} {last_name}")
    if not authors_list:
        return "Anonymous"
    elif len(authors_list) == 1:
        return authors_list[0]
    elif len(authors_list) <= 10:
        return ', '.join(authors_list[:-1]) + ', and ' + authors_list[-1]
    else:
        return authors_list[0] + ' et al.'

# Citation formatting functions for APA, MLA, and Chicago styles

def format_apa_citation(article):
    authors = article.get('author', [])
    authors_str = format_authors_apa(authors)

    # Publication Year
    publication_year = 'n.d.'
    if 'published-print' in article:
        publication_year = article['published-print']['date-parts'][0][0]
    elif 'published-online' in article:
        publication_year = article['published-online']['date-parts'][0][0]
    elif 'issued' in article:
        publication_year = article['issued']['date-parts'][0][0]

    # Title
    title = article.get('title', [''])[0]
    if not title:
        title = "No title available"

    # Journal or Conference Name
    container_title = article.get('container-title', [''])[0]

    # Volume and Issue
    volume = article.get('volume', '')
    issue = article.get('issue', '')

    if volume and issue:
        volume_issue = f"*{volume}*({issue})"
    elif volume:
        volume_issue = f"*{volume}*"
    else:
        volume_issue = ''

    # Pages
    pages = article.get('page', '')

    # DOI or URL
    doi = article.get('DOI', '')
    url = article.get('URL', '')

    if doi:
        doi_str = f"https://doi.org/{doi}"
    elif url:
        doi_str = f"Retrieved from {url}"
    else:
        doi_str = ''

    # Construct Citation
    citation = f"{authors_str} ({publication_year}). {title}."

    if container_title:
        citation += f" *{container_title}*"
        if volume_issue:
            citation += f", {volume_issue}"
        if pages:
            citation += f", {pages}"
        citation += "."
    else:
        citation += "."

    if doi_str:
        citation += f" {doi_str}"

    return citation.strip()

def format_mla_citation(article):
    authors = article.get('author', [])
    authors_str = format_authors_mla(authors)

    # Publication Year
    publication_year = 'n.d.'
    if 'published-print' in article:
        publication_year = article['published-print']['date-parts'][0][0]
    elif 'published-online' in article:
        publication_year = article['published-online']['date-parts'][0][0]
    elif 'issued' in article:
        publication_year = article['issued']['date-parts'][0][0]

    # Title
    title = article.get('title', [''])[0]
    if not title:
        title = "No title available"

    # Container Title
    container_title = article.get('container-title', [''])[0]

    # Volume and Issue
    volume = article.get('volume', '')
    issue = article.get('issue', '')
    volume_issue = ''
    if volume and issue:
        volume_issue = f", vol. {volume}, no. {issue}"
    elif volume:
        volume_issue = f", vol. {volume}"
    elif issue:
        volume_issue = f", no. {issue}"

    # Pages
    pages = article.get('page', '')
    if pages:
        pages = f", pp. {pages}"

    # DOI or URL
    doi = article.get('DOI', '')
    url = article.get('URL', '')

    if doi:
        access_info = f", doi:{doi}"
    elif url:
        access_info = f". Accessed from {url}"
    else:
        access_info = ''

    # Citation
    citation = f"{authors_str}. \"{title}.\" {container_title}{volume_issue}{pages}, {publication_year}{access_info}."
    return citation.strip()

def format_chicago_citation(article):
    authors = article.get('author', [])
    authors_str = format_authors_chicago(authors)

    # Publication Year
    publication_year = 'n.d.'
    if 'published-print' in article:
        publication_year = article['published-print']['date-parts'][0][0]
    elif 'published-online' in article:
        publication_year = article['published-online']['date-parts'][0][0]
    elif 'issued' in article:
        publication_year = article['issued']['date-parts'][0][0]

    # Title
    title = article.get('title', [''])[0]
    if not title:
        title = "No title available"

    # Container Title
    container_title = article.get('container-title', [''])[0]

    # Volume and Issue
    volume = article.get('volume', '')
    issue = article.get('issue', '')
    volume_issue = ''
    if volume and issue:
        volume_issue = f" {volume}, no. {issue}"
    elif volume:
        volume_issue = f" {volume}"
    elif issue:
        volume_issue = f" no. {issue}"

    # Pages
    pages = article.get('page', '')
    if pages:
        pages = f": {pages}"

    # DOI or URL
    doi = article.get('DOI', '')
    url = article.get('URL', '')

    if doi:
        access_info = f". doi:{doi}"
    elif url:
        access_info = f". Accessed {url}"
    else:
        access_info = ''

    # Citation
    citation = f"{authors_str}. \"{title}.\" {container_title}{volume_issue} ({publication_year}){pages}{access_info}."
    return citation.strip()

def format_citation(article, style, query):
    title = article.get('title', ['No title'])[0]
    doi = article.get('DOI', '')
    url = article.get('URL', '')
    publisher = article.get('publisher', 'Unknown Publisher')

    # Metadata Completeness Check
    if not is_metadata_complete(article):
        logging.warning(f"Incomplete metadata for article: {title}")
        if VALIDATION_STRICTNESS == 1:
            return None

    # Publisher Reputation Check
    if not is_publisher_reputable(publisher):
        logging.warning(f"Unreputable publisher for article: {title}")
        if VALIDATION_STRICTNESS <= 2:
            return None

    # Title Relevance Check
    if not is_title_relevant(title, query):
        logging.warning(f"Title not relevant for article: {title}")
        if VALIDATION_STRICTNESS == 1:
            return None

    # Cross-Verification
    if doi and not cross_verify_article(doi, title):
        logging.warning(f"Cross-verification failed for article: {title}")
        if VALIDATION_STRICTNESS <= 2:
            return None

    # DOI or URL Validation
    doi_valid = validate_doi(doi) if doi else False
    url_valid = validate_url(url) if url else False
    arxiv_valid = validate_arxiv_id(url) if is_arxiv_article(url) else False

    if not (doi_valid or url_valid or arxiv_valid):
        logging.warning(f"Invalid DOI/URL for article: {title}")
        if VALIDATION_STRICTNESS <= 2:
            return None

    # Proceed to format citation
    if style == "APA":
        return format_apa_citation(article)
    elif style == "MLA":
        return format_mla_citation(article)
    elif style == "Chicago":
        return format_chicago_citation(article)
    else:
        return "Unsupported citation style."

@st.cache_data(show_spinner=False)
def search_articles(query):
    try:
        response = requests.get(f"https://api.crossref.org/works?query={query}&rows=10", timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("message", {}).get("items", [])
        return articles
    except Exception as e:
        st.error(f"An error occurred while fetching articles: {e}")
        logging.error(f"Article search error: {e}")
        return []

# Insight Report Assistant Functions

@st.cache_data
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

Generate a concise, meaningful insight based on this information. The insight should be relevant, data-driven, and actionable.

Insight:
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
            time.sleep(2 ** attempt)
            continue
        except openai.error.OpenAIError as e:
            st.error(f"An error occurred: {e}")
            return "Error generating insight."
    st.error("Failed to generate insight after multiple attempts due to rate limiting.")
    return "Error generating insight."

# Main Application Function

def main():
    st.set_page_config(page_title="Advanced Research Assistant", page_icon=":microscope:", layout="wide")

    # Access Control
    if 'locked' not in st.session_state:
        st.session_state.locked = True
    if 'permanent_unlock' not in st.session_state:
        st.session_state.permanent_unlock = False
    security_manager = SecurityManager()

    # Access control logic
    if st.session_state.locked:
        st.title("Secured Access")
        st.write("This application requires authorization to access.")
        passkey = st.text_input("Enter passkey:", type="password", key="passkey")
        if st.button("Unlock"):
            if passkey == "Flostradamus":
                st.session_state.locked = False
                st.success("Access Granted. Welcome!")
            else:
                st.error("Incorrect passkey. Access denied.")
        release_key = st.text_input("Enter release phrase:", type="password", key="release_key")
        if st.button("Release"):
            if release_key == "Release for Jedi Trials":
                st.session_state.locked = False
                st.session_state.permanent_unlock = True
                st.success("Locks permanently disabled.")
            else:
                st.error("Incorrect release phrase.")
    else:
        st.title("Advanced Research Assistant")
        if st.session_state.permanent_unlock:
            st.info("System is operating in public mode.")
        else:
            st.info("System is operating in developer mode.")

        # Initialize managers
        citation_manager = SmartCitationManager()
        llm = ContinualLearningLLM("gpt2")  # You can replace "gpt2" with a different model if preferred
        quality_control_bot = AIQualityControlBot()
        metrics_collector = MetricsCollector()

        # Placeholder for app functionality
        st.write("System is ready for operation.")

        # Include the Insight Report Writing Assistant and Auto Citation Tool
        # Tabs for separating functionalities
        tab1, tab2 = st.tabs(["Insight Report Assistant", "Auto Citation Tool"])

        with tab1:
            st.header("Insight Report Writing Assistant")

            # OpenAI Parameters
            st.sidebar.title("Configuration")
            temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
            max_tokens = st.sidebar.number_input("Max Tokens:", min_value=50, max_value=2048, value=150)

            # Knowledge Base Input
            st.subheader("Step 1: Provide Insight Report Writing Guidelines")
            knowledge_base_input = st.text_area("Enter the guide on how to write an insight report:", value=DEFAULT_KNOWLEDGE_BASE, height=200)

            # Context Input
            st.subheader("Step 2: Provide Up-to-Date Context")
            context_input = st.text_area("Enter any additional context for the insight report:", height=100)

            # CSV Upload
            st.subheader("Step 3: Upload Your CSV File")
            sample_csv = "User,Category,Prompt,Response\nJohn Doe,Sales,\"How did sales perform this quarter?\",\"Sales increased by 15%.\""
            st.download_button(
                "Download Sample CSV",
                sample_csv,
                "sample.csv",
                "text/csv",
                key='download-sample-csv'
            )

            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file is not None:
                df = process_csv(uploaded_file)
                if df is not None:
                    st.write("CSV file uploaded successfully!")
                    st.dataframe(df)

                    if st.button("Generate Insights"):
                        insights = []
                        with st.spinner("Generating insights..."):
                            for index, row in df.iterrows():
                                logging.info(f"Generating insight for User: {row['User']}, Category: {row['Category']}")
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
                        df['Generated Insight'] = insights
                        st.success("Insights generated successfully!")
                        st.dataframe(df)

                        # Download options
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download CSV with Insights",
                            csv,
                            "insights_report.csv",
                            "text/csv",
                            key='download-csv'
                        )

                        # For Excel download
                        towrite = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
                        df.to_excel(towrite, index=False)
                        towrite.close()
                        with open('temp.xlsx', 'rb') as f:
                            excel_data = f.read()
                        st.download_button(
                            label="Download Excel with Insights",
                            data=excel_data,
                            file_name='insights_report.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )

            # Question Answering
            st.subheader("Ask a Question About Insight Report Writing")
            question = st.text_input("Enter your question:")
            if question:
                prompt = f"{knowledge_base_input}\n\n{context_input}\n\nQuestion: {question}\n\nAnswer:"
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": knowledge_base_input + "\n\n" + context_input},
                            {"role": "user", "content": f"Question: {question}\n\nAnswer:"}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    answer = response['choices'][0]['message']['content'].strip()
                    st.write("Answer:", answer)
                except openai.error.OpenAIError as e:
                    st.error(f"An error occurred: {e}")

        with tab2:
            st.header("Auto Citation Tool")

            st.sidebar.title("Auto Citation Tool Settings")
            validation_level = st.sidebar.radio(
                "Validation Strictness Level:",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Strict", 2: "Moderate", 3: "Lenient"}[x],
                key='validation_level'
            )

            global VALIDATION_STRICTNESS
            VALIDATION_STRICTNESS = validation_level

            query = st.text_input("Enter your search query:")
            citation_style = st.selectbox("Select Citation Style:", ["APA", "MLA", "Chicago"], key='citation_style')

            if st.button("Search", key='search_button'):
                if query:
                    with st.spinner('Searching...'):
                        articles = search_articles(query)
                    if articles:
                        st.write("### Search Results and Citations:")
                        selected_articles = []
                        for idx, article in enumerate(articles):
                            title = article.get('title', ['No title available'])[0]
                            st.write(f"**{idx+1}. {title}**")
                            include = st.checkbox("Include in Bibliography", key=f'include_{idx}')
                            if include:
                                citation = format_citation(article, citation_style, query)
                                if citation:
                                    st.write(citation)
                                    selected_articles.append(citation)
                                else:
                                    st.error(f"Failed to generate citation for article: {title}")
                        if selected_articles:
                            if st.button("Generate Bibliography", key='generate_bibliography'):
                                st.write("### Bibliography:")
                                for cite in selected_articles:
                                    st.write(cite)
                        else:
                            st.info("No articles selected for bibliography.")
                    else:
                        st.write("No results found.")
                else:
                    st.write("Please enter a search query.")

        # Help Section
        st.sidebar.markdown("### How to Use")
        st.sidebar.markdown("""
1. Navigate between tabs to use different tools.
2. For the Insight Report Assistant:
   - Provide your guidelines and context.
   - Upload a CSV file with the required columns.
   - Adjust OpenAI parameters if needed.
   - Click 'Generate Insights' to process the data.
3. For the Auto Citation Tool:
   - Enter your search query.
   - Select the citation style.
   - Adjust validation strictness level.
   - Click 'Search' to retrieve articles and generate citations.
""")

        # Monitor system metrics
        metrics_collector.update_memory_usage()

if __name__ == "__main__":
    main()


