import json
from haystack.nodes import BM25Retriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from tqdm import tqdm

base_dir = 'path/to/'
test_dir = base_dir + 'test/'
model_dir = base_dir + "saved_models/dpr_bertimbau_squad"

test_dataset = json.load(open(test_dir + 'dpr-test.json', 'r', encoding='utf-8'))
document_store = ElasticsearchDocumentStore()
document_store.delete_documents()
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

# retriever = BM25Retriever(document_store)
retriever = DensePassageRetriever.load(load_dir=model_dir, document_store=document_store)
document_store.update_embeddings(retriever)

acertos = 0
somatoria_mrr = 0
top_k = 20

progress_bar = tqdm(total=total_perguntas, desc="Avaliando")

for qa in test_dataset:    
    candidatos = retriever.retrieve(query=qa['question'], top_k=top_k)
    gold_document_id = qa['positive_ctxs'][0]['passage_id']
    posicao_gold_document = 1

    for candidato in candidatos:
        if candidato.meta['passage_id'] == gold_document_id:
            acertos += 1
            somatoria_mrr += 1/posicao_gold_document
            break

        posicao_gold_document += 1
    progress_bar.update(1)

progress_bar.close()

print(f'\nrecall (top_k={top_k}) = {acertos/total_perguntas}')
print(f'mrr (top_k={top_k}) = {(1/total_perguntas)*somatoria_mrr}')







