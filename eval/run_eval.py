import json,re,sys,time
from pathlib import Path 

##########################################################

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))    # to import core from eval/

from core import runAgent,MODEL
from logger import calcCost

##########################################################

QUESTİON_FILE = ROOT / "eval" / "questions.json"
RESULT_DIR = ROOT / "eval" / "results"
HIT = re.compile(r"^\[\d+\] (\S+) p\.(\d+)", re.MULTILINE)

##########################################################

class ToolRecorder:
    """Collect the queries and retrieved pages of one question """
    def __init__(self):
        self.queries = []
        self.pages = set()
    
    def record(self, name, toolInput, result):
        self.queries.append(toolInput.get("query",""))
        collectPages(result,self.pages)


##########################################################

def collectPages(result,pages):
    """add every found in one serachDocs result to pages """
    pages.update((src,int(p)) for src,p in HIT.findall(result))


def checkRetrieval(expected,pages,searched):
    """ Was at least one expected page retrieved? If none expected, no search should happen. """
    if not expected:
        return (not searched,"searched but should not have" if searched else "")
    wanted = {(e["source"],e["page"]) for e in expected}
    ok = bool(wanted & pages)

    return ok, "" if ok else f"expected one of {sorted(wanted)}"


def checkAnswer(answer,mustInclude, mustNotInclude):
    """Required words present, forbidden words absent (case-insensitive)."""
    text = answer.lower()
    missing = [w for w in mustInclude if w.lower() not in text]
    leaked = [w for w in mustNotInclude if w.lower() in text]
    reasons = ([f"missing {missing}"] if missing else []) + ([f"forbidden {leaked}"] if leaked else []) 

    return not reasons, "; ".join(reasons)   


def scoreCase(case,answer,rec:ToolRecorder):
    """score retrieval and answe seperately for one question."""
    retOk, retWhy = checkRetrieval(case["expected_pages"],rec.pages,bool(rec.queries))
    ansOk, ansWhy = checkAnswer(answer,case["must_include"], case["must_not_include"])

    return {"retrieval_ok": retOk, "answer_ok": ansOk, 
            "reason":"; ".join(r for r in [retWhy,ansWhy] if r) }


def runCase(case):
    """Ask the agent one question and score retrieval and answer separately."""
    rec = ToolRecorder()
    answer, turns = runAgent(case["question"], onTool= rec.record)
    tokensIn = sum(t["input_tokens"] for t in turns)
    tokensOut = sum(t["output_tokens"] for t in turns)

    return { "id": case["id"], "category": case["category"],
        **scoreCase(case, answer, rec),
        "queries": rec.queries, "turns": len(turns),
        "cost_usd": calcCost(MODEL, tokensIn, tokensOut), "answer": answer,}


def mark(ok):
    """ ok or bad for progress line."""

    return "OK " if ok else "BAD"


def printRow(i,total,r:dict):
    """One progress line per question."""
    print(f"[{i:2}/{total}] ret {mark(r['retrieval_ok'])} ans {mark(r['answer_ok'])}  "
          f"{r['id']}  {r['reason']}", flush=True)


def printSummary(results:list[dict]):
    """scores and total cost for the whole run """
    retScore = sum(r["retrieval_ok"] for r in results)
    ansScore = sum(r["answer_ok"] for r in results)
    totalCost = sum(r["cost_usd"] for r in results)
    print(f"\n Model : {MODEL}")
    print(f"Retrieval: {retScore}/{len(results)}    Answer: {ansScore}/{len(results)}")
    print(f"Total Cost: ${totalCost:.4f}")


def saveResults(results:list[dict]):
    """ write the full results to a timestamped JSON file """
    RESULT_DIR.mkdir(exist_ok=True)
    outFile = RESULT_DIR/f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    outFile.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    
    return outFile

def main():
    cases = json.loads(QUESTİON_FILE.read_text(encoding="utf-8"))
    results = []
    for i, case in enumerate(cases,start=1):
        results.append(runCase(case))
        printRow(i,len(cases),results[-1])
    printSummary(results)
    print(f"saved to .. {saveResults(results).relative_to(ROOT)}")


###########################################################

if __name__ == "__main__":
    main()