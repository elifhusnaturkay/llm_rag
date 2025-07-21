from utils import get_chroma
from ollama import chat, ChatResponse

conversation_history = []

def answer_question(query):
    collection = get_chroma()

    results = collection.query(query_texts=[query], n_results=5)
    context = "\n---\n".join(results["documents"][0])

    messages = [
        {
            'role': 'system',
            'content': (
                'sana verilen ilgili bilgilerden yola cikarak kullanicinun sorusunu cevapla, '
                'icerikte ilgili soru ile alakali bilgi gelmez ise bu konu hakkinda bilgim yok cevabini ver. '
                'aksi takdirde kullaniciya mevzuattan yararlanarak cevaplarini ver. '
                'kisa cevaplar vermen yeterli. '
                'Konuşmanın geçmişini göz önünde bulundurarak cevap ver.'
            )
        }
    ]

    messages.append({'role': 'system', 'content': f'### İLGİLİ BİLGİLER ###\n{context}'})
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': f'### SORU ###\n{query}'})

    response: ChatResponse = chat(
        model='deepseek-llm:7b',
        messages=messages 
    )

    llm_response_content = response["message"]["content"]

    conversation_history.append({'role': 'user', 'content': f'### SORU ###\n{query}'})
    conversation_history.append({'role': 'assistant', 'content': llm_response_content})

    print("\nYANIT:")
    print(llm_response_content)

if __name__ == "__main__":
    print("Sohbet başladı. Çıkmak için 'q', 'quit' veya 'exit' yazın.")
    while True:
        soru = input("\nSoru: ")
        if soru.lower() in ["q", "quit", "exit"]:
            break
        answer_question(soru)
