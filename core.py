import anthropic
import os 
from dotenv import load_dotenv
from tools import TOOLS,TOOL_FUNCTIONS
from datetime import datetime
from time import perf_counter

#######################################################

MODEL = "claude-haiku-5-5"
load_dotenv()
claudeKEy = os.getenv("claudeKey")

#######################################################

client = anthropic.Anthropic(api_key=claudeKEy)

#######################################################

def generateResponse(messages,turns):
    """ contact with agent """
    start = perf_counter()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=TOOLS,
        messages=messages,
        system = build_system_prompt()
    )
    end = perf_counter()

    ## logging each steps 
    toolNames = [b.name for b in response.content if b.type == "tool_use"]
    log = {
        "step": len(turns) + 1,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "duration_sec": round(end - start, 2),
        "tools": toolNames,
    }
    
    turns.append(log)

    return response

def build_system_prompt():
    today = datetime.now().strftime("%A, %Y-%m-%d")
    return (
        f"Today's date is {today}.\n\n"
        "You are a research assistant for three papers: ReAct, RAG and Toolformer.\n\n"
        "Searching:\n"
        "- For any question about these papers, use searchDocs before answering.\n"
        "- If the results do not answer the question, search again with different words, "
        "up to 3 searches in total.\n"
        "- Do not search for greetings or questions unrelated to the papers.\n\n"
        "Answering:\n"
        "- Answer only from the retrieved text. Do not add facts from your own knowledge.\n"
        "- Cite each claim as (source, page), e.g. (react.pdf, p.3).\n"
        "- If the papers do not cover the question, say so clearly.\n"
        "- Keep answers concise, but answer every part of the question."
    )

def toolUsage(usrPrompt,respond,turns,N=10, onTool=None):
    """agent helper to use tools agents have"""

    messages = []
    endTurn = False
    messages.append({"role":"user","content":usrPrompt})

    while N>0 and not endTurn:
        N = N-1
        toolResults = []

        messages.append({"role":"assistant","content":respond.content})

        ### scans blocks which have to use tools..
        for block in respond.content:
            if block.type != "tool_use":
                continue
            func = TOOL_FUNCTIONS.get(block.name)
            if func is not None:
                ## call the function from tool pool.
                res = func(**block.input)
                isErr = res.startswith("Error")
            else:
                res,isErr = f"Err: ! Tool: {block.name}",True
            toolResults.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": res,
                "is_error": isErr,
            })
            
            ## visualize to call a tool on streamlit
            if onTool:
                 onTool(block.name, block.input, res)

        messages.append({"role": "user", "content": toolResults})
        respond = generateResponse(messages,turns)

        ## stop mechanism
        if respond.stop_reason == "end_turn":
            endTurn = True
        else :
            mssg = "".join(b.text for b in respond.content if b.type=="text")
            if respond.stop_reason == "max_tokens":
                return f"{mssg} \n\n !!! Max Token !!!"
            if N == 0:
                return f"{mssg} \n\n !!! MAX ITR 10 is completed.. "
        
    return respond

def runAgent(question, onTool=None):
    """Run the agent for one question and return the answer text and turn logs."""
    turns = []
    messages = [{"role": "user", "content": question}]
    response = generateResponse(messages, turns)

    if response.stop_reason == "tool_use":
        result = toolUsage(question, response, turns, onTool=onTool)
    else:
        result = response

    if isinstance(result, str):
        answerText = result
    else:
        answerText = "".join(b.text for b in result.content if b.type == "text")

    return answerText, turns