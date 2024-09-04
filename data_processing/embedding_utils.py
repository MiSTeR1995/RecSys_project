import torch
from sentence_transformers import SentenceTransformer

# Определение устройства (CUDA или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sbert_model(config):
    """
    Загружает модель Sentence-BERT на основе конфигурации.
    :param config: Конфигурационный словарь, содержащий название модели.
    :return: Загруженная модель Sentence-BERT.
    """
    model_name = config.get('sbert_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # Дефолтная модель
    sbert_model = SentenceTransformer(model_name)
    sbert_model.to(device)
    return sbert_model

def embed_text(text, sbert_model):
    """
    Преобразование текста в эмбеддинг с использованием Sentence-BERT.
    :param text: Текст для преобразования в эмбеддинг.
    :param sbert_model: Модель Sentence-BERT для получения эмбеддингов.
    :return: Эмбеддинг текста.
    """
    with torch.no_grad():
        embeddings = sbert_model.encode(text, convert_to_tensor=True, device=device)
    return embeddings
