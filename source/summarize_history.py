import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_history(history):

    prompt = f"""
Résume la conversation suivante.
Ne garde que :
- Le but général de l’utilisateur
- Les infos importantes
- Les contraintes
- Les solutions déjà explorées

Conversation :
{history}

Résumé :
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

