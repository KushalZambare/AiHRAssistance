from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from pydantic import BaseModel

# LLM
llm = ChatOllama(model="qwen3:4b")

# Load Vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.load_local(
    "HrVectorDb", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever()

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def experience_calculator(start_year: str) -> str:
    """Calculate a candidate's total years of experience given the year they started working."""
    return str(datetime.now().year - int(start_year))


@tool
def eligibility_checker(skills: str) -> str:
    """Check whether a candidate is eligible based on their skills (comma-separated list)."""
    required = {"python", "sql", "git"}
    candidate = {skill.strip().lower() for skill in skills.split(",")}
    missing = required - candidate

    if not missing:
        return "Eligible"
    return "Not Eligible. Missing: " + ", ".join(missing)


@tool
def interview_question(skills: str) -> str:
    """Generate 5 interview questions for the given skills."""
    prompt = f"Generate 5 interview questions for a candidate with the following skills: {skills}"
    return llm.invoke(prompt).content

@tool
def score_experience_user(experience: int) -> int:
    """Calculate candidate score based on years of experience."""
   
    if experience > 5:
        score = 10
    elif experience > 3:
        score = 7
    else:
        score = 2
 
    return score
 


@tool
def company_policy_search(question: str) -> str:
    """Search company HR policy documents to answer questions about leave, notice period,
    working hours, WFH policy, job descriptions, or company benefits."""
    docs = retriever.invoke(question)
    content = "\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Answer only from the provided content.\n"
        f"Context:\n{content}\n\n"
        f"Question: {question}"
    )
    return llm.invoke(prompt).content


# ── Agent ─────────────────────────────────────────────────────────────────────

tools = [
    experience_calculator,
    eligibility_checker,
    interview_question,
    company_policy_search,
]

SYSTEM_PROMPT = """
You are an HR Recruitment Assistant. Use tools whenever required.

If the user asks about any of the following topics, ALWAYS use the company_policy_search tool:
- leave policy
- notice period
- working hours
- work from home policy
- job description
- company benefits
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# ── Structured output for resume parsing ──────────────────────────────────────

class Candidate(BaseModel):
    name: str
    start_year: int
    skills: List[str]

structured_llm = llm.with_structured_output(Candidate)

# ── Chat loop ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("         HR RECRUITMENT ASSISTANT")
print("=" * 60)
print("Type 'exit' to quit.")
print("Prefix your message with 'resume:' to parse a resume.\n")

while True:
    user_input = input("You : ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Resume parsing mode
    if user_input.lower().startswith("resume:"):
        resume_text = user_input[len("resume:"):].strip()
        candidate = structured_llm.invoke(
            f"Extract the following fields from the resume:\n"
            f"- name\n- start_year (the year they started their career)\n- skills (list)\n\n"
            f"Resume:\n{resume_text}"
        )
        print("\n── Candidate Details ──────────────────────────")
        print(f"  Name       : {candidate.name}")
        print(f"  Start Year : {candidate.start_year}")
        print(f"  Skills     : {', '.join(candidate.skills)}")
        print("───────────────────────────────────────────────\n")
        continue

    # Agent mode
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]}
    )

    # The last message in the response is the AI's final answer
    final_message = response["messages"][-1]
    print(f"\nAssistant: {final_message.content}\n")
