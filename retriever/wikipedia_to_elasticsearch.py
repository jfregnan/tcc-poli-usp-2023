from datasets import load_dataset

wikipedia_pt = load_dataset("graelo/wikipedia", "20230901.pt")
wikipedia_pt['train'].to_csv('wikipedia-pt.csv')

from haystack.document_stores import ElasticsearchDocumentStore
import nltk
import re
from tqdm import tqdm
import pandas as pd

# A função abaixo tem o objetivo de eliminar caracteres de quebra de linha (\n)
# e espaçamentos extras que "poluem" os textos do dump. 
def replace_extra_spaces(input_string, delimiter=' - '):    
    result_string = re.sub(r' {2,}', delimiter, input_string.replace('\n', ' '))
    return result_string

# Converte grandes quantidades de texto em uma lista de trechos com 
# aproximadamente 100 palavras sem dividir frases ao meio. 
def split_article(article, max_words=100):
    sentences = nltk.sent_tokenize(article)
    result = []
    current_string = ''
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if len(current_string.split()) + len(words) <= max_words:
            current_string += ' ' + sentence if current_string != '' else sentence
        else:
            if current_string:
                result.append(current_string)
            current_string = sentence
    if current_string:
        result.append(current_string)    
    return result

# Inicialização da DocumentStore
document_store = ElasticsearchDocumentStore()

# Remove todos os documentos antigos da DocumentStore
document_store.delete_documents()

progress_bar = tqdm(total=1107946, desc="Processing")

# Carrega o CSV contento o dump da Wikipedia em português. O arquivo foi divido
# em partes (chunks) de 50 linhas, processados no laço for abaixo. 
wikipedia_pt = pd.read_csv('wikipedia-pt.csv', chunksize=50)
i = 0

for chunk in wikipedia_pt:
    dicts = []    
    try:    
        for index, artigo in chunk.iterrows():
            text = artigo['text']
            title = artigo['title']
            sentences = split_article(text)
            for sentence in sentences:    
                dicts.append({
                                'content': replace_extra_spaces(sentence), 
                                'meta': {'title': title}
                            })
        # Insere uma lista de documentos (dicts) na DocumentStore
        document_store.write_documents(dicts, duplicate_documents='overwrite')
    except Exception as e:
        print(f'    -> Erro ao processar o chunk {i}: {e}')
    finally:
        progress_bar.update(50)
        i += 1
progress_bar.close()