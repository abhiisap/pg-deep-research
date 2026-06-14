"""Run the deep research agent and save the report. Usage: python run.py"""

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

from src.deep_researcher import graph

QUESTION = (
    "What are the emerging brands that compete with P&G products and "
    "why are they popular with consumers?"
)

if __name__ == "__main__":
    print("=" * 70)
    print("Deep research agent starting. This takes a few minutes on the")
    print("free tier (it pauses between steps to respect rate limits).")
    print("=" * 70)
    result = graph.invoke(
        {"messages": [HumanMessage(content=QUESTION)]},
        config={"configurable": {}, "recursion_limit": 100},
    )
    report = result["final_report"]
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70 + "\n")
    print(report)
    with open("results/RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n\n[done] Live report saved to results/RESULTS.md")
