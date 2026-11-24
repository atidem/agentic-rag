from core import runAgent,MODEL
from logger import saveLog

###########################################################

toolCalls = []

###########################################################

def showTool(name,toolInput,result):
    """Print each tool call and remember it for the log."""
    topHit = result.split("\n",1)[0]
    toolCalls.append({"tool": name, "input": toolInput, "top_hit": topHit})
    print(f"\n >> {name} ({toolInput})")
    print(" ==>>  "+ topHit + " ... ")

if __name__ == "__main__":
    question = input("Question:  ")
    answer, turns = runAgent(question,onTool=showTool)
    print(f"\n  {answer}" )
    print(f"\n [{len(turns)} turns, {sum(t["input_tokens"]for t in turns)} input tokens]")
    saveLog(question, answer, turns, MODEL, toolCalls)