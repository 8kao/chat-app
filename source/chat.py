import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve_context
from decompose import decompose_query
from summarize_history import summarize_history

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class bcolors:
    WHITE = "\033[37m"
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'

def build_prompt(context_items, conv_summary, conv_history_raw, question):

    #contexte
    context_text = ""
    for item in context_items:
        src = item["metadata"].get("source", "inconnu")
        dist = item["distance"]
        content = item["content"]
        context_text += f"\n\n[Source: {src} | Score: {dist:.3f}]\n{content}"

    #on garde l'historique de la conv pour l'ajouter au prompt
    history_text = ""

    if conv_summary:
        history_text += f"--- RÉSUMÉ ---\n{conv_summary}\n"

    # derniers échanges (2 max)
    history_text += "\n--- DERNIERS ÉCHANGES ---\n"
    for m in conv_history_raw:
        history_text += f"{m['role'].upper()}: {m['content']}\n"


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

    conv_history_raw = []
    conv_summary = ""
    history_counter =0

    print(f"\n{bcolors.OKGREEN}Assistant >>{bcolors.WHITE} Comment puis-je vous aider ?\n(Tapez 'exit' pour quitter le chat). ")
    while True:

        user_input = input(f"\n{bcolors.OKCYAN}Vous >>{bcolors.WHITE} ")
    
        if user_input.lower() == "exit":
            break
        
        conv_history_raw.append({"role": "utilisateur", "content": user_input})
        history_counter += 1
        #on decoupe la question pour savoir si il y a plusieurs taches et on les traite une par une (ex: explique-moi comment fonctionne geometry ET compare avec physics)
        subtasks = decompose_query(user_input)
        contexts = []
        for task in subtasks:
            context = retrieve_context(task, k=5)
            contexts.append({"task": task, "ctx": context})

        #tous les 10 tours on fait appel à un llm pour résumer l'historique de la conversation afin d'alléger le prompt envoyé au llm chat 
        if (history_counter >= 10):
            full_history_text = ""
            for m in conv_history_raw:
                full_history_text += f"{m['role']}: {m['content']}\n"

            # resumer
            new_summary = summarize_history(full_history_text)

            # ajouter dans le resume global
            if conv_summary == "":
                conv_summary = new_summary
            else:
                conv_summary += "\n" + new_summary

            #garder seulement les 2 derniers messages
            conv_history_raw = conv_history_raw[-2:]

            history_counter = 0

            prompt = build_prompt(
                context_items=contexts,
                conv_summary=conv_summary,
                conv_history_raw=conv_history_raw,
                question=user_input
            )

        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        conv_history_raw.append({"role": "assistant", "content": llm_response})
        
        print(f"\n{bcolors.OKGREEN}Assistant >>{bcolors.WHITE} ", llm_response)

if __name__ == "__main__":
    chat()
