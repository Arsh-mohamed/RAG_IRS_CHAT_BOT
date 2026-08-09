"""
chatbot.py — a conversation-only wrapper around retrieval.py.

Pipeline per turn:
    user question
      -> retrieve relevant context   (retrieval.answer_question)
      -> build a prompt with recent history + that context
      -> generate a natural reply    (gpt-4o-mini)
      -> remember the turn

Deliberately kept simple: NO refusal guardrail, NO forced-citation grounding
prompt, NO question-condensing. Just retrieve, remember, and talk.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Reuse the retrieval you already built (hybrid search + SQLite routing).
from retrival import answer_question

load_dotenv()  # picks up OPENAI_API_KEY from your .env

# --- Tuning knobs -----------------------------------------------------------
CHAT_MODEL = "gpt-4o-mini"   # same model family the notebook used
TOP_K = 5                    # how many context chunks to retrieve per turn
HISTORY_TURNS = 6            # how many recent messages to feed back in
TEMPERATURE = 0.2            # low = focused/factual; raise for chattier replies

SYSTEM_PROMPT = (
    "You are a helpful, conversational assistant that answers questions about "
    "US tax documents. Use the retrieved context to inform your answer and the "
    "strictly provide data from the context. You have access to the recent.Don't calculate on your own. Use the provided context to answer questions, and if the context doesn't fully cover the question, answer as best you can and be upfront about anything you're unsure of. Always return the source filename(s) for any information you provide."
    "conversation history to understand follow-up questions. Reply naturally, "
    "as if chatting. If the context doesn't fully cover the question, answer as "
    "best you can and be upfront about anything you're unsure of"
    "always return the source filename(s) for any information you provide"
    "if you don't know the answer, say so instead of making something up.you can say you are not aware of the answer_question"
)

def format_context(results):
    """Turn the retriever's results into a plain num
    bered context block."""
    if not results:
        return "(no relevant context found)"
    blocks = []
    for i, r in enumerate(results, start=1):
        source = r.get("source_filename", "unknown")
        blocks.append(f"[{i}] (from {source}) {r['answer']}")
    return "\n\n".join(blocks)


class TaxChatbot:
    def __init__(self, model=CHAT_MODEL, top_k=TOP_K, history_turns=HISTORY_TURNS):
        self.llm = ChatOpenAI(model=model, temperature=TEMPERATURE)
        self.top_k = top_k
        self.history_turns = history_turns
        self.history = []  # list of (role, content) tuples

    def _recent_history(self):
        if not self.history:
            return "(no previous conversation)"
        recent = self.history[-self.history_turns:]
        return "\n".join(f"{role}: {content}" for role, content in recent)

    def chat(self, question):
        # 1. Retrieve context (this also runs your SQLite exact-answer routing).
        results = answer_question(question, top_k=self.top_k)
        context = format_context(results)

        # 2. Assemble the prompt: system role + history + context + question.
        user_message = (
            f"Conversation so far:\n{self._recent_history()}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}"
        )
        messages = [("system", SYSTEM_PROMPT), ("user", user_message)]

        # 3. Generate the reply.
        answer = self.llm.invoke(messages).content.strip()

        # 4. Remember this turn so follow-ups have context.
        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        return answer


def run_chat():
    bot = TaxChatbot()
    print("Tax chatbot ready. Ask a question (q to quit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break
        if question:
            print(f"\nBot: {bot.chat(question)}")


if __name__ == "__main__":
    # One-shot if given arguments, otherwise an interactive loop.
    question = " ".join(sys.argv[1:]).strip()
    if question:
        print(TaxChatbot().chat(question))
    else:
        run_chat()