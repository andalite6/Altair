import streamlit as st
import pandas as pd
import openai
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

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

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = "your_openai_api_key_here"

# Caching the CSV processing
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

def main():
    st.title("Insight Report Writing Assistant")

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

    # Help Section
    st.sidebar.markdown("### How to Use")
    st.sidebar.markdown("""
1. Provide your insight report guidelines.
2. Enter any additional context.
3. Upload your CSV file containing the required columns.
4. Adjust the OpenAI parameters if needed.
5. Click 'Generate Insights' to process the data.
""")

if __name__ == "__main__":
    main()

