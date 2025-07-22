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
                'Sana verilen ilgili bilgilerden yola çıkarak kullanıcının sorusunu cevapla. '
                'İçerikte soru ile alakalı bilgi gelmez ise "Bu konu hakkında bilgim yok." cevabını ver. '
                'Aksi takdirde kullanıcıya mevzuattan yararlanarak cevaplarını ver. '
                'Eğer cevap bir tablo içeriyorsa, tabloyu referans göstermek yerine içeriğini metin olarak açıkla. '
                'Kısa ve net cevaplar vermen yeterli. '
                'Konuşmanın geçmişini göz önünde bulundurarak cevap ver.'
            )
        }
    ]

    messages.append({'role': 'system', 'content': f'### İLGİLİ BİLGİLER ###\n{context}'})
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': f'### SORU ###\n{query}'})

    response: ChatResponse = chat(
        model='gemma3:12b', #muadili olarak gemma3:12b-instruct
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
