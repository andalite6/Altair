# Filename: full_application.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import csv
import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
pip install --upgrade pip

# Initialize the environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Deadbolt Lock State
if 'locked' not in st.session_state:
    st.session_state['locked'] = True
    st.session_state['permanent_unlock'] = False

# Deadbolt Lock Functions
def unlock_app():
    if st.session_state.passkey == "Flostradamus":
        st.session_state['locked'] = False
        st.success("Access Granted. Welcome!")
    else:
        st.error("Incorrect passkey. Access denied.")

def release_app():
    if st.session_state.release_key == "Release for Jedi Trials":
        st.session_state['locked'] = False
        st.session_state['permanent_unlock'] = True
        st.success("Locks permanently disabled. App is now open for public usage.")
    else:
        st.error("Incorrect release phrase. Try again.")

# Citation Dataclass
@dataclass
class Citation:
    author: str
    year: int
    title: str
    url: str
    doi: str
    relevance_score: float
    usage_count: int
    last_used: datetime

class SmartCitationManager:
    def __init__(self):
        self.citations = {}
        self.citation_patterns = {}
        self.load_citations()
        
    def load_citations(self):
        """Load saved citations from disk"""
        try:
            with open('citation_database.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                self.citations = saved_data.get('citations', {})
                self.citation_patterns = saved_data.get('patterns', {})
        except FileNotFoundError:
            pass

    def save_citations(self):
        """Save citations to disk"""
        with open('citation_database.pkl', 'wb') as f:
            pickle.dump({
                'citations': self.citations,
                'patterns': self.citation_patterns
            }, f)

    def get_citation_data(self, query: str) -> Tuple[str, int, str, str, str, float]:
        """Fetch citation data from CrossRef API"""
        url = f"https://api.crossref.org/works?query={query}&rows=5"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("message", {}).get("items"):
                items = data["message"]["items"]
                scored_items = []
                for item in items:
                    relevance_score = self._calculate_relevance(query, item)
                    scored_items.append((relevance_score, item))
                best_match = max(scored_items, key=lambda x: x[0])
                item = best_match[1]
                author = item.get("author", [{"family": "Unknown Author"}])[0].get("family")
                year = item.get("created", {}).get("date-parts", [[0]])[0][0]
                title = item.get("title", ["Unknown Title"])[0]
                doi = item.get("DOI", "No DOI Available")
                url = f"https://doi.org/{doi}"
                return author, year, title, url, doi, best_match[0]
        except Exception as e:
            logging.error(f"Citation fetch error: {str(e)}")
            return None, None, None, None, None, 0.0

    def _calculate_relevance(self, query: str, item: Dict) -> float:
        score = 0.0
        title = item.get("title", [""])[0].lower()
        query_terms = query.lower().split()
        score += sum(term in title for term in query_terms) / len(query_terms)
        year = item.get("created", {}).get("date-parts", [[0]])[0][0]
        recency_score = min(1.0, (float(year) - 2000) / 25) if year else 0
        score += recency_score * 0.3
        completeness = sum(1 for field in ["author", "DOI", "abstract"] if field in item) / 3
        score += completeness * 0.2
        return score

    def add_citation(self, query: str, analysis_context: str) -> Optional[Citation]:
        author, year, title, url, doi, relevance = self.get_citation_data(query)
        if not all([author, year, title, url]):
            return None
        citation = Citation(
            author=author,
            year=year,
            title=title,
            url=url,
            doi=doi,
            relevance_score=relevance,
            usage_count=1,
            last_used=datetime.now()
        )
        key = hashlib.md5(f"{author}{year}{title}".encode()).hexdigest()
        self.citations[key] = citation
        self._update_citation_patterns(query, analysis_context, key)
        self.save_citations()
        return citation

    def _update_citation_patterns(self, query: str, context: str, citation_key: str):
        keywords = set(query.lower().split() + context.lower().split())
        for keyword in keywords:
            if keyword not in self.citation_patterns:
                self.citation_patterns[keyword] = {}
            if citation_key not in self.citation_patterns[keyword]:
                self.citation_patterns[keyword][citation_key] = 1
            else:
                self.citation_patterns[keyword][citation_key] += 1

    def suggest_citations(self, context: str, n: int = 3) -> List[Citation]:
        keywords = context.lower().split()
        citation_scores = {}
        for keyword in keywords:
            if keyword in self.citation_patterns:
                for cit_key, count in self.citation_patterns[keyword].items():
                    citation_scores[cit_key] = citation_scores.get(cit_key, 0) + count
        sorted_citations = sorted(citation_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [self.citations[key] for key, _ in sorted_citations]

class InsightReportGenerator:
    def __init__(self, csv_file: str = "insight_report_data.csv"):
        self.csv_file = csv_file
        self.fields = ["Timestamp", "Category", "Evidence", "Insight", "Benchmark", "TestMethod", "MetaData"]
        self.ensure_csv_exists()
        
    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.fields)

    def collect_user_input(self):
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'category': st.text_input("Category (e.g., Performance, Security):"),
            'evidence': st.text_area("Evidence or test case:"),
            'insight': st.text_area("Initial insight:"),
            'benchmark': st.text_input("Benchmark data:"),
            'test_method': st.text_input("Testing methodology:"),
            'metadata': st.text_area("Additional metadata:")
        }

    def save_to_csv(self, data: Dict):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([data.get(field.lower(), "") for field in self.fields])

# Streamlit UI
def create_streamlit_ui():
    st.title("Full Application with Deadbolt Lock, Citation, and Insight Reporting")

    if st.session_state['locked']:
        st.subheader("🔒 Application is Locked")
        st.text_input("Enter passkey to unlock:", type="password", key="passkey")
        st.button("Unlock", on_click=unlock_app)
        st.text_input("Enter release phrase for permanent unlock:", type="password", key="release_key")
        st.button("Release", on_click=release_app)
    else:
        bot = SmartCitationManager()
        report_gen = InsightReportGenerator()
        st.subheader("🔓 Application Access Granted")
        
        st.write("### Insight Report Generation")
        report_data = report_gen.collect_user_input()
        if st.button("Save Report"):
            report_gen.save_to_csv(report_data)
            st.success("Report saved successfully.")
        
        st.write("### Citation Suggestions")
        context = report_data.get('category', "data analysis")
        suggested_citations = bot.suggest_citations(context)
        for citation in suggested_citations:
            st.write(f"- {citation.author} ({citation.year}). *{citation.title}*. [DOI: {citation.doi}]({citation.url})")
        
        if st.session_state.get('permanent_unlock', False):
            st.write("The application has been permanently unlocked for public usage.")
        else:
            st.write("The application is temporarily unlocked for developer access.")

if __name__ == "__main__":
    create_streamlit_ui()
streamlit run full_application.py
