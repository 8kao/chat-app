import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve_context
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(context_items, conv_history, question):

    #contexte
    context_text = ""
    for item in context_items:
        src = item["metadata"].get("source", "inconnu")
        dist = item["distance"]
        content = item["content"]
        context_text += f"\n\n[Source: {src} | Score: {dist:.3f}]\n{content}"

    #on garde l'historique de la conv pour l'ajouter au prompt
    history_text = ""
    for turn in conv_history:
        role = turn["role"]
        msg = turn["content"]
        history_text += f"{role.upper()}: {msg}\n"

    prompt = f"""
Tu es un assistant expert et rigoureux. Tu utilises EXCLUSIVEMENT les informations
du contexte pour répondre. Tu prends aussi en compte l'historique de la conversation
pour comprendre le fil de la discussion.

Si la réponse n'existe PAS dans le contexte :
- tu dis clairement : "Je ne trouve pas cette information dans les documents."
- tu n'inventes rien.
- tu n'ajoutes aucune spéculation.

===== CONTEXTE =====
{context_text}

===== HISTORIQUE =====
{history_text}

===== CONSIGNES =====
1. Utilise uniquement le contexte fourni.
2. Ne mélange pas des connaissances externes.
3. Cite les sources entre crochets comme ceci : [source: path/to/file].
4. Si plusieurs sources donnent des infos contradictoires, dis-le.
5. Si la question est vague, demande une précision.
6. Répond de manière claire, organisée, pédagogique.

==== QUESTION ====
{question}

==== RÉPONSE ====
"""
    return prompt

def chat():

    conv_history = []

    print("Comment puis-je vous aider ?\n(Tapez 'exit' pour quitter le chat).")

    while True:
        user_input = input("\nVous >> ")
    
        if user_input.lower() == "exit":
            break
        
        conv_history.append({"role": "utilisateur", "content": user_input})
        
        context = retrieve_context(user_input)

        prompt = build_prompt(context, conv_history, user_input)

        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        conv_history.append({"role": "assistant", "content": llm_response})

        print("\nAssistant >>", llm_response)

if __name__ == "__main__":
    chat()
