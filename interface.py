import os
import re
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

# Constants
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
HF_MODEL_REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"

PROMPT_TEMPLATE = """
You are an expert in Monte Carlo methods, answering questions based on the provided context from a course document. The context may contain LaTeX math expressions (e.g., \\mathrm{{E}}[h(X)], \\sqrt{{n}}). Answer in plain text, avoiding markdown or LaTeX syntax (e.g., no $...$ or $$...$$). Instead, use ASCII-friendly notation to represent math clearly for console output. For example:
- Use 'E[h(X)]' for expectation instead of \\mathrm{{E}}[h(X)].
- Use 'Sum_k=1^K' for summation instead of \\sum_{{k=1}}^K.
- Use 'sigma_k^2' for variance instead of \\sigma_k^2.
- Use '1/n * Sum_i=1^n' for averages instead of \\frac{{1}}{{n}} \\sum_{{i=1}}^n.
Do not modify the mathematical structure or content from the context; use the exact formulas provided, converting them to plain text. If the question requires a mathematical explanation, derive or explain using these conventions, ensuring clarity and accuracy. For stratification, use the standard weighted sum estimator and variance formulas from the document, such as:
- Estimator: mu_strat = Sum_k=1^K w_k * (1/n_k * Sum_i=1^n_k h(X_k,i))
- Variance: Var(mu_strat) = Sum_k=1^K w_k^2 * sigma_k^2 / n_k

Context:
{context}

---

Question: {question}

Answer in plain text with clear, ASCII-friendly math notation, ensuring formulas are accurate and based on the document.
"""

# Custom embedding class for Hugging Face Inference API
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=api_key)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.feature_extraction(text)
            embeddings.append(response[0].tolist() if isinstance(response, list) else response.tolist())
        return embeddings

    def embed_query(self, text):
        response = self.client.feature_extraction(text)
        return response[0].tolist() if isinstance(response, list) else response.tolist()

# Custom LLM class for Hugging Face Inference API
class HFInferenceLLM:
    def __init__(self, model_repo, token, temperature=0.5, max_tokens=512):
        self.client = InferenceClient(model=model_repo, token=token)
        self.model = model_repo
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

def escape_latex_braces(text):
    """
    Escape curly braces in LaTeX expressions to prevent string format errors.
    Replaces { with {{ and } with }} outside of format placeholders.
    """
    def replace_braces(match):
        content = match.group(0)
        return content.replace('{', '{{').replace('}', '}}')

    pattern = r'\\[a-zA-Z]+{[^}]*}|\{[^}]*\}'
    escaped_text = re.sub(pattern, replace_braces, text)
    return escaped_text

def query_monte_carlo(query_text):
    """Query the Chroma database and LLM with the given text."""
    COMMON_GREETINGS = ["hi", "hello", "bonjour", "salut", "how are you", "what's up", "yo"]
    if any(greet in query_text.lower() for greet in COMMON_GREETINGS):
        return (
            "👋 Hey there! I'm here to help you understand Monte Carlo simulation topics.\n"
            "💡 Try asking something like: 'What is the Law of Large Numbers?' or 'How does stratification reduce variance?'",
            []
        )

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN

    # Initialize embedding function
    embedding_function = HuggingFaceAPIEmbeddings(
        model_name=EMBEDDING_MODEL,
        api_key=HF_API_TOKEN
    )

    # Load Chroma database
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant documents
    results = vectordb.similarity_search_with_relevance_scores(query_text, k=3)
    if not results or results[0][1] < 0.5:
        return (
            "⚠️ No relevant content found in the Monte Carlo course.\n"
            "💡 This assistant only answers technical questions related to Monte Carlo simulation (in the PDF).",
            []
        )

    # Prepare context and prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_text = escape_latex_braces(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query LLM
    llm = HFInferenceLLM(model_repo=HF_MODEL_REPO, token=HF_API_TOKEN)
    answer = llm.invoke(prompt)

    return answer.strip(), results

def main():
    st.set_page_config(page_title="Monte Carlo Methods Q&A", page_icon="📚")
    st.title("Monte Carlo Methods Question-Answering System")
    st.markdown("Ask questions about Monte Carlo methods based on the course document. Answers include mathematical formulas in plain text for clarity.")

    # Query input
    query_text = st.text_input("Enter your question:", placeholder="e.g., How does stratification reduce variance?")
    
    if st.button("Submit Query"):
        if query_text:
            with st.spinner("Processing your query..."):
                answer, results = query_monte_carlo(query_text)
                
                # Display answer
                st.subheader("Answer")
                st.text(answer)
                
                # Display source documents
                if results:
                    st.subheader("Source Documents")
                    for i, (doc, score) in enumerate(results):
                        page = doc.metadata.get("page", f"Doc {i+1}")
                        st.write(f"• Page {page} (Score: {score:.2f})")

                
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()