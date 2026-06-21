from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from .rag import get_query_engine, RAGUnavailable

DEMO_QUESTIONS = [
    "Which countries generated the most nuclear electricity in 2025, and how dependent are they on it?",
    "How has France's nuclear dependency changed over time?",
    "Compare China and France in nuclear generation and grid dependence.",
    "Which countries have the highest decarbonisation scores?",
    "Which European countries are most dependent on nuclear energy?",
]


@require_http_methods(["GET", "POST"])
def analyst(request):
    answer = None
    question = ""
    error = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        if question:
            try:
                engine = get_query_engine()
                answer = str(engine.query(question))
            except RAGUnavailable as e:
                error = str(e)
            except Exception as e:
                error = f"Query failed: {e}"

    return render(request, "insights/analyst.html", {
        "question": question,
        "answer": answer,
        "error": error,
        "demo_questions": DEMO_QUESTIONS,
    })

