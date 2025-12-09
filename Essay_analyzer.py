from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ---------------------------------------------------------
# Streamlit Cloud Secrets
# ---------------------------------------------------------

if "api_key" not in st.secrets:
    st.error("Missing API key. Add it in your Streamlit secrets.")
    st.stop()

API_KEY = st.secrets["api_key"]

# ---------------------------------------------------------
# LLM + Parsers
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=API_KEY,
    temperature=0.3
)

parser = StrOutputParser()

# ---------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------

essay_prompt = PromptTemplate(
    template='''
Evaluate the essay provided in the input. Work in three sections only.

Structural assessment
Identify weaknesses in organization, argument flow, clarity of thesis, and paragraph coherence.

Content assessment
Point out gaps in reasoning, unsupported claims, factual inconsistencies, and missing evidence.

Revision directives
List concrete, actionable changes that would improve clarity, argument strength, and overall quality.
Avoid rewriting the essay. Focus only on weaknesses and what should be fixed.

Base your analysis strictly on the text.
{essay}
''',
    input_variables=["essay"]
)

weakness_highlighter = PromptTemplate(
    template='''
Identify every weakness present in the following text.
Focus only on logical gaps, unclear reasoning, missing evidence,
structural issues, or inaccurate claims. Do not restate strengths.
Do not summarize the full text. Return only the weaknesses with no added explanation.

Text:
{analysis_output}
''',
    input_variables=["analysis_output"]
)

suggestion_generator = PromptTemplate(
    template='''
Provide precise, high-value suggestions based on the following text.
Identify the core issue, state what matters, list exact actions that fix it.
Do not add commentary, questions, or motivational tone.

{analysis_output}
''',
    input_variables=["analysis_output"]
)

final_prompt = PromptTemplate(
    template='''
Create a clear, well-structured analysis report with the following sections:

MAIN WEAKNESSES:
{Weakness}

SUGGESTIONS FOR IMPROVEMENT:
{Suggestions}

OVERVIEW:
{summary}

Format each section clearly without markdown tables or complex formatting.
''',
    input_variables=["summary", "Weakness", "Suggestions"]
)

# ---------------------------------------------------------
# Chains
# ---------------------------------------------------------

chain1 = RunnableParallel(
    {
        "summary": essay_prompt | llm | parser,
        "Weakness": essay_prompt | llm | parser | weakness_highlighter | llm | parser,
        "Suggestions": essay_prompt | llm | parser | suggestion_generator | llm | parser,
    }
)

final_chain = chain1 | final_prompt | llm | parser

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Essay Analyzer", layout="wide")

st.title("Essay Analysis Tool")
st.write("Paste your essay below to receive detailed analytical feedback.")

col1, col2 = st.columns([2, 1])

with col1:
    essay_input = st.text_area(
        "Your Essay:",
        height=400,
        placeholder="Paste your essay text here..."
    )

with col2:
    if essay_input:
        st.subheader("Document Info")
        words = len(essay_input.split())
        paras = len([p for p in essay_input.split('\n\n') if p.strip()])
        sentences = len([s for s in essay_input.replace('?', '.').replace('!', '.').split('.') if s.strip()])

        st.write(f"**Word count:** {words}")
        st.write(f"**Paragraphs:** {paras}")
        st.write(f"**Sentences:** {sentences}")

if st.button("Analyze Essay", type="primary"):
    if not essay_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):
            try:
                raw = final_chain.invoke({"essay": essay_input})

                clean = raw.replace('|', '').replace('---', '').replace('-------', '')
                sections = clean.split('\n\n')

                st.subheader("Analysis Report")

                for section in sections:
                    block = section.strip()
                    if not block:
                        continue

                    header = block.split('\n')[0].strip()

                    if header.upper().startswith("MAIN WEAKNESSES"):
                        st.write("---")
                        st.subheader("Main Weaknesses")
                        for line in block.split('\n')[1:]:
                            if line.strip():
                                st.write(f"• {line.strip()}")

                    elif header.upper().startswith("SUGGESTIONS"):
                        st.write("---")
                        st.subheader("Suggestions for Improvement")
                        for line in block.split('\n')[1:]:
                            if line.strip():
                                st.write(f"• {line.strip()}")

                    elif header.upper().startswith("OVERVIEW"):
                        st.write("---")
                        st.subheader("Overview")
                        content = ' '.join(block.split('\n')[1:])
                        st.write(content)

                st.download_button(
                    "Download Analysis Report",
                    data=raw,
                    file_name="essay_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {e}")
