from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("api_key"),
    temperature=0.3
)

parser = StrOutputParser()

essay_prompt = PromptTemplate(
    template='''Evaluate the essay provided in the input. Work in three sections only.

Structural assessment
Identify weaknesses in organization, argument flow, clarity of thesis, and paragraph coherence.

Content assessment
Point out gaps in reasoning, unsupported claims, factual inconsistencies, and missing evidence.

Revision directives
List concrete, actionable changes that would improve clarity, argument strength, and overall quality.
Avoid rewriting the essay. Focus only on weaknesses and what should be fixed.

Base your analysis strictly on the text.
{essay}''',
    input_variables=["essay"]
)

weakness_highlighter = PromptTemplate(
    template='''Identify every weakness present in the following text. 
Focus only on logical gaps, unclear reasoning, missing evidence,
structural issues, or inaccurate claims. Do not restate strengths.
Do not summarize the full text. Return only the weaknesses with no added explanation.

Text:
{analysis_output}''',
    input_variables=["analysis_output"]
)

suggestion_generator = PromptTemplate(
    template='''Provide precise, high-value suggestions based on the following text.
Identify the core issue, state what matters, list exact actions that fix it. Do not add commentary, questions, or motivational tone.
{analysis_output}''',
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

chain1 = RunnableParallel(
    {
        "summary": essay_prompt | llm | parser,
        "Weakness": essay_prompt | llm | parser | weakness_highlighter | llm | parser,
        "Suggestions": essay_prompt | llm | parser | suggestion_generator | llm | parser,
    }
)

final_chain = chain1 | final_prompt | llm | parser

st.set_page_config(page_title="Essay Analyzer", layout="wide")

st.title("Essay Analysis Tool")
st.write("Paste your essay below to receive detailed feedback on structure, content, and areas for improvement.")

col1, col2 = st.columns([2, 1])

with col1:
    essay_input = st.text_area(
        "Your Essay:",
        height=400,
        placeholder="Paste your essay text here...",
        key="essay_input"
    )

with col2:
    if essay_input:
        st.subheader("Document Info")
        word_count = len(essay_input.split())
        para_count = len([p for p in essay_input.split('\n\n') if p.strip()])
        sentence_count = len([s for s in essay_input.replace('?', '.').replace('!', '.').split('.') if s.strip()])
        
        st.write(f"**Word count:** {word_count}")
        st.write(f"**Paragraphs:** {para_count}")
        st.write(f"**Sentences:** {sentence_count}")

if st.button("Analyze Essay", type="primary"):
    if not essay_input.strip():
        st.warning("Please enter an essay to analyze.")
    else:
        with st.spinner("Analyzing your essay... This may take a moment."):
            try:
                result = final_chain.invoke({"essay": essay_input})
                
                st.subheader("Analysis Report")
                
                # Clean up the result by removing any markdown table artifacts
                clean_result = result.replace('|', '').replace('---', '').replace('-------', '')
                
                # Split into sections
                sections = clean_result.split('\n\n')
                
                for section in sections:
                    if not section.strip():
                        continue
                    
                    lines = section.strip().split('\n')
                    section_title = lines[0].strip()
                    
                    if section_title.upper().startswith('MAIN WEAKNESSES'):
                        st.write("---")
                        st.subheader("Main Weaknesses")
                        for line in lines[1:]:
                            if line.strip() and not line.isspace():
                                st.write(f"• {line.strip()}")
                    
                    elif section_title.upper().startswith('SUGGESTIONS'):
                        st.write("---")
                        st.subheader("Suggestions for Improvement")
                        for line in lines[1:]:
                            if line.strip() and not line.isspace():
                                st.write(f"• {line.strip()}")
                    
                    elif section_title.upper().startswith('OVERVIEW'):
                        st.write("---")
                        st.subheader("Overview")
                        overview_content = ' '.join(lines[1:]) if len(lines) > 1 else ' '.join(lines)
                        st.write(overview_content)
                    
                    else:
                        # For any other content, display it as is
                        if len(lines) > 1:
                            st.write(section)
                
                # Download button
                st.download_button(
                    label="Download Analysis Report",
                    data=result,
                    file_name="essay_analysis.txt",
                    mime="text/plain",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")