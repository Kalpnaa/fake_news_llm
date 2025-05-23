
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# --- Choose the LLM Model ---
# Using 'gemini-1.5-flash-latest' for good performance and higher free-tier limits
MODEL_NAME = 'models/gemini-1.5-flash-latest'
model = genai.GenerativeModel(MODEL_NAME)

# --- Define the core prompt for fake news detection ---
# We will refine this prompt iteratively!
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

# --- Example Usage (We will replace this with real articles soon!) ---
example_article = """
The Reserve Bank of India's Monetary Policy Committee announced on Friday that it would keep the key repo rate unchanged at 6.5%, citing persistent inflationary pressures and global economic uncertainties. Governor Shaktikanta Das emphasized the central bank's commitment to achieving its inflation target while supporting sustainable growth. Analysts had widely anticipated the decision, with many noting the need for continued vigilance.

"""

print(f"Analyzing an example article...\n")

full_prompt = get_fake_news_detection_prompt(example_article)

try:
    response = model.generate_content(full_prompt)
    print("LLM's Fake News Analysis:")
    print(response.text)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This could be due to quota limits, network issues, or an invalid model name.")
    print("If it's a quota error (429), wait a minute or two and try again, or consider using a different model like 'gemini-1.5-flash-latest' if not already.")

