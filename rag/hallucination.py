LOW_CONFIDENCE_THRESHOLD = 0.40

NO_CONTEXT_REPLY = (
    "I don't have enough information in the provided 3GPP documents to answer this question."

)

def check_hallucination_risk(retrieved_docs:list[dict], answer:str) -> dict:
    if not retrieved_docs:
        return {"answer":NO_CONTEXT_REPLY , "confidence":"none","grounded":False,"sources":[]}

    top_score = retrieved_docs[0]["score"]
    avg_score = round(sum(d["score"] for d in retrieved_docs) / len(retrieved_docs), 4)

    if top_score < LOW_CONFIDENCE_THRESHOLD:
        return {"answer":NO_CONTEXT_REPLY , "confidence":"low","grounded":False,"sources":[]}


    return {
        "answer":answer,
        "confidence":"high" if top_score >= 0.65 else "medium",
        "grounded":NO_CONTEXT_REPLY.lower() not in answer.lower(),
        "sources":list({d["metadata"].get("source","unknown") for d in retrieved_docs}),
        "top_score":round(top_score,4),
        "avg_score":avg_score
    }