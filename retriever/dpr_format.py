import json
from haystack.nodes import BM25Retriever
from haystack.document_stores import ElasticsearchDocumentStore
from tqdm import tqdm

base_dir = 'datasets/'
train_dir = base_dir + 'train/'

nq_squad = json.load(open(train_dir + 'nq-squad-train.json', 'r', encoding='utf-8-sig'))
nq_dpr = []

document_store = ElasticsearchDocumentStore()
retriever = BM25Retriever(document_store)
progress_bar = tqdm(total=len(nq_squad), desc="Processing")

for qa in nq_squad:    
    try:    
        qas = qa['qas']
        pergunta = qas['question']
        resposta = qas['answers'][0]        
                
        dpr_qa = {
            'question': pergunta,
            'answers': [resposta],
            'positive_ctxs': [{'title': '', 'text': qa['context'], 'passage_id': ''}],
            'negative_ctxs': []
        }
        
        candidatos_hard_negative_ctx = retriever.retrieve(query=pergunta, top_k=5)
        
        for document in candidatos_hard_negative_ctx:            
            if resposta.lower() not in document.content.lower():            
                title = document.meta['title']                
                dpr_qa['hard_negative_ctxs'] = [{'title': title, 
                                                 'text': document.content, 
                                                 'passage_id': ''}]
                
                candidatos_titulo = retriever.retrieve(query=qa['context'], top_k=1)        
                dpr_qa['positive_ctxs'][0]['title'] = candidatos_titulo[0].meta['title']
                
                nq_dpr.append(dpr_qa)
                break

    except Exception as e:
        print(f'   -> Erro: {e}.')
        print()
    
    finally:    
        progress_bar.update(1)    
   
progress_bar.close()

with open(train_dir + 'nq-dpr-train.json', 'w', encoding='utf-8') as fout:
    json.dump(nq_dpr, fout, ensure_ascii=False)



# [
#     {
#         "question": "....",
#         "answers": ["...", "...", "..."],
#         "positive_ctxs": [{
#             "title": "...",
#             "text": "...."
#         }],
#         "negative_ctxs": ["..."],
#         "hard_negative_ctxs": ["..."]
#     },
#     ...
# ]