from search import search

###########################################################

DEFAULT_K = 5
MAX_K = 8

###########################################################

def formatResult(n,r):
    """One search hit as a numbered, citable block. """
    return f"[{n}] {r["source"]} p.{r["page"]} · {r["section"]} (score {r["score"]})\n {r["text"]}"


def searchDocs(query, k=DEFAULT_K):
    """search the indexed papers and return the closest chunks as text"""
    if not query.strip():
        return "err: empty query.."
    k = max(1,min(k,MAX_K))      ## restricted number of k
    results = search(query,k)
    return "\n\n".join(formatResult(n,r) for n,r in enumerate(results,start=1))


###########################################################
"""Tools schema"""

SEARCH_DOCS_TOOL = {
    "name": "searchDocs",
    "description": (
        "Searches a local collection of three research papers: ReAct (reasoning and acting "
        "in language models), RAG (retrieval-augmented generation) and Toolformer (language "
        "models that learn to use tools). Use it for any question about these papers or their "
        "methods, results and examples. Returns the most similar text chunks, each with a number, "
        "source file, page, section and a similarity score. Scores above 0.45 are usually "
        "relevant; below 0.35 the papers probably do not cover the question. Search is by "
        "meaning, so short, specific queries work best. If the results do not answer the "
        "question, search again with different words."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Short search query about one idea. Example: ReAct hallucination compared to chain-of-thought",
            },
            "k": {
                "type": "integer",
                "description": "Number of results, 1 to 8. Default: 5.",
                "minimum": 1,
                "maximum": 8,
            },
        },
        "required": ["query"],
    },
}

###########################################################
"""Tool Variables"""

TOOLS = [SEARCH_DOCS_TOOL]
TOOL_FUNCTIONS = {"searchDocs": searchDocs}