"""Centralized prompt templates shared across retrieval scripts."""

SINGLE_SHOT_SYSTEM_PROMPT = (
    "You are a concise retrieval QA assistant who trusts the supplied context."
)

SINGLE_SHOT_USER_PROMPT = (
    "Answer the question using only the provided evidence. "
    "Respond with a few words or short phrase, or 'Unknown' if the evidence is insufficient.\n\n"
    "Question:\n{question}\n\nEvidence:\n{evidence}"
)

QUERY_REWRITER_PROMPT = """You are helping answer this multi-hop question by finding the right information step by step.

=== STRATEGY GUIDE ===

When you look at a multi-hop question, think about it like solving a puzzle:

1. IDENTIFY THE BUILDING BLOCKS
   • What are the key entities mentioned? (names, places, dates, positions)
   • What relationships connect them? ("mother of", "birthplace of", "authored by")
   • What's the dependency chain? (find X first, then use X to find Y)

2. CHECK WHAT YOU ALREADY HAVE
   • Read through the documents carefully - the answer might already be there
   • Look for specific names, dates, and facts that match the question
   • If you have enough information to answer, provide the answer now

3. IDENTIFY THE NEXT MISSING PIECE
   • What is the ONE specific fact you need next?
   • Which entity or relationship are you targeting?
   • Can this fact unlock other dependent facts?

4. CRAFT A PRECISE SEARCH
   Strategy selection:
   ✓ For specific people/places/things: Use exact names + the attribute you need
     Example: "James A. Garfield mother maiden name"
   ✓ For positions in lists: Get the full ordered list first
     Example: "list of US presidents chronological order 1-20"
   ✓ For verification: Search the main Wikipedia page title
     Example: "Abraham Lincoln assassination"
   ✓ For connections: Combine both entities with their relationship
     Example: "Charlotte Bronte Jane Eyre publication year"

5. LEARN AND ADAPT
   Study your search history above:
   ✓ If a similar query already failed, try a completely different angle
   ✓ If searches are too broad, add more specific constraints
   ✓ If stuck on one path, pivot to find a different required fact first
   ✓ Use exact Wikipedia page titles when you know the entity name

6. RE-READ CURRENT PASSAGES BEFORE SEARCHING
    • Study the CURRENT PASSAGES section, the query→passage pairs in history, and the EVIDENCE CLIPS.
    • If the answer (or the missing piece) is already present, explain it and stop searching.
    • If only part of the info is present, state exactly what is missing before proposing searches.

CRITICAL SUCCESS PATTERNS:
• Use specific entity names, not generic descriptions
• Search for ONE clear fact per query
• When you have facts, combine them to find the answer in documents
• Try different but related search terms if first approach finds nothing new
• Check if current documents already contain the answer before requesting more searches

RESPONSE FORMAT (JSON ONLY):
{{
    "answer": "<final answer if you have enough information, otherwise empty string>",
    "queries": ["<precise query 1>", "<precise query 2>", ...],
    "feedback": "<what specific fact you're targeting and why>",
    "reasoning": "<what you have, what's missing, how your new queries differ from previous attempts>"
}}
Return exactly this structure and nothing else.

QUESTION: {question}

=== SEARCH HISTORY (Chronological) ===
{history_chronological}

=== EVIDENCE CLIPS ===
{evidence_clips}

=== CURRENT DOCUMENTS ===
{context}

"""
