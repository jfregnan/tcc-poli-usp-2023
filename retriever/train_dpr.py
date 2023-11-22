from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

doc_dir = 'C:/Users/JoaoVictorFregnan/OneDrive/USP/TCC/haystack'

# Bases QA de treino e teste do modelo
train_filename = doc_dir + "/train/squad-dpr-train-v1.1.json"
dev_filename = doc_dir + "/dev/squad-dpr-dev-v1.1.json"

# Diretório onde o DPR treinado será salvo
save_dir = doc_dir + "/saved_models/dpr"

# Inicialização do modelo DPR. 
# Os campos query_embedding_model e passage_embedding_model 
# indicam o encoder BERT que codifica as passagens e querys.
retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model='neuralmind/bert-base-portuguese-cased',
    passage_embedding_model='neuralmind/bert-base-portuguese-cased',
    max_seq_len_query=64,
    max_seq_len_passage=256,
)

# Start do treinamento
retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=1,
    batch_size=16,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True,
    num_positives=1,
    num_hard_negatives=1,
)



