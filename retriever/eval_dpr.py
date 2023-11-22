import json
from haystack.nodes import BM25Retriever
from haystack.document_stores import ElasticsearchDocumentStore

base_dir = 'C:/Users/jfregnan/OneDrive - EQUITAS/USP/TCC 2/datasets/'
# train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'

test_dataset = json.load(open(test_dir + 'nq-dpr-dev.json', 'r', encoding='utf-8'))
document_store = ElasticsearchDocumentStore()
# document_store.delete_documents()
dicts = []
total_perguntas = len(test_dataset)
passage_id = 0

for qa in test_dataset:
    positive_ctx = qa['positive_ctxs'][0]    
    
    documento_inserido = next((documento for documento in dicts if documento['content'] == positive_ctx['text']), None)
    
    if documento_inserido == None:
        positive_ctx['passage_id'] = passage_id
        dicts.append({'content': positive_ctx['text'], 'meta': {'title': positive_ctx['title'], 'passage_id': positive_ctx['passage_id']}})
        passage_id += 1
    
    else:
        positive_ctx['passage_id'] = documento_inserido['meta']['passage_id']

document_store.write_documents(dicts, duplicate_documents='overwrite')

retriever = BM25Retriever(document_store)
acertos = 0
count = 1

somatoria_mrr = 0
top_k = 20

for qa in test_dataset:
    print(f'Avaliando pergunta {count}/{total_perguntas}')
    candidatos = retriever.retrieve(query=qa['question'], top_k=top_k)
    gold_document_id = qa['positive_ctxs'][0]['passage_id']  
    
    posicao_gold_document = 1
    
    for candidato in candidatos:
        if candidato.meta['passage_id'] == gold_document_id:
            acertos += 1
            somatoria_mrr += 1/posicao_gold_document
            break
        posicao_gold_document += 1
    
    count += 1

recall = acertos/total_perguntas
mrr = (1/total_perguntas)*somatoria_mrr







