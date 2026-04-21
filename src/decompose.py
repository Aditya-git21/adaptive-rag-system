import re

SPLIT_WORDS = ["and also", "also", " and ", "as well as", 
               "difference between", "compare", "versus", "vs"]

def needs_decomposition(query):
    q = query.lower()
    for word in SPLIT_WORDS:
        if word in q:
            return True
    return False

def decompose(query):
    q = query.lower().strip()
    
    # try splitting on "and also" first
    if "and also" in q:
        parts = q.split("and also")
    elif " and " in q:
        parts = q.split(" and ")
    elif "as well as" in q:
        parts = q.split("as well as")
    else:
        parts = [q]
    
    # clean up each part
    parts = [p.strip() for p in parts if len(p.strip()) > 4]
    return parts
