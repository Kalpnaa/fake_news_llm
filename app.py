import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st 


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


MODEL_NAME = 'models/gemini-1.5-flash-latest' # Using the more lenient model
model = genai.GenerativeModel(MODEL_NAME)


def get_fake_news_detection_prompt(article_text):
    prompt = f"""
    You are an AI assistant specialized in identifying potential misinformation and biased content in news articles. Your goal is to critically analyze the provided article for signs of fakery, sensationalism, or misleading information.

    Consider the following aspects during your analysis:
    1.  **Factual Claims & Verifiability:** Does the article make strong factual claims? Are these claims supported by evidence or verifiable sources?
    2.  **Source Credibility:** Are the sources cited (if any) generally considered reputable? Is the source of the article itself known for factual reporting or for spreading misinformation?
    3.  **Tone and Language:** Is the language used objective and neutral, or is it highly emotional, inflammatory, or overly opinionated? Look for buzzwords, hyperbolic statements, or appeals to emotion rather than logic.
    4.  **Headline-Body Consistency:** Does the headline accurately reflect the content of the article, or is it clickbait designed to mislead?
    5.  **Logical Coherence:** Is the narrative consistent? Are there contradictions or illogical jumps in reasoning?
    6.  **Grammar and Style:** Are there unusual grammatical errors, typos, or an unprofessional writing style that might indicate a lack of editorial oversight?

    Based on your analysis, provide:
    -   A 'Likelihood of Fake News' assessment (choose one: "Very Low", "Low", "Medium", "High", "Very High").
    -   A confidence score (0-100%) for your assessment.
    -   A concise explanation (2-3 sentences) detailing the *primary reasons* for your assessment, referencing the criteria above.
    -   Any specific examples from the article that support your reasoning.

    ---
    News Article to Analyze:
    {article_text}
    ---

    Your analysis:
    """
    return prompt


st.set_page_config(page_title="Zero-Shot Fake News Detector", layout="wide")

st.title("📰 Zero-Shot Fake News Detector")
st.markdown("Enter a news article below to get an AI-powered assessment of its likelihood of being fake.")
st.markdown("---")

# Text area for user input
article_input = st.text_area("Paste News Article Here:", height=300,
                             placeholder="Paste the full text of the news article you want to analyze...")

# Button to trigger analysis
if st.button("Analyze Article"):
    if article_input:
        with st.spinner("Analyzing... Please wait."):
            try:
                full_prompt = get_fake_news_detection_prompt(article_input)
                response = model.generate_content(full_prompt)

                # Display the LLM's analysis
                st.markdown("### AI Analysis Results:")
                st.write(response.text) # Display the raw text output from the LLM

                st.success("Analysis Complete!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.warning("Please ensure your API key is correct and you have an active internet connection.")
                st.info("If you encounter a quota error (429), wait a minute or two and try again. "
                        "Consider using 'gemini-1.5-flash-latest' for higher free-tier limits.")
    else:
        st.warning("Please paste a news article into the text area to analyze.")

st.markdown("---")
st.caption("Powered by Google Gemini LLM & Streamlit | This tool provides an AI-powered assessment and should not be considered definitive fact-checking.")
