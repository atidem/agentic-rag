from core import runAgent

###########################################################

def showTool(name,toolInput,result):
    """Print each tool call so we can match the agent work."""
    print(f"\n >> {name} ({toolInput})")
    print(" ==>>  "+result[:200].replace("\n"," ")+ "...")

if __name__ == "__main__":
    question = input("Question:  ")
    answer, turns = runAgent(question,onTool=showTool)
    print(f"\n  {answer}" )
    print(f"\n [{len(turns)} turns, {sum(t["input_tokens"]for t in turns)} input tokens]")