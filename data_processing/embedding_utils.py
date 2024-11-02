import os
import torch
from sentence_transformers import SentenceTransformer
from utils.logger import info, warning
from safetensors.torch import save_file, load_file  # Импорт для работы с safetensors

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

def save_embeddings_to_file(embeddings, file_path):
    """
    Сохраняет эмбеддинги в файл.

    :param embeddings: Список эмбеддингов (тензоров).
    :param file_path: Путь к файлу, куда будут сохранены эмбеддинги.
    """
    save_file({"embeddings": embeddings}, file_path)
    info(f"Эмбеддинги сохранены в формате Safetensors в файл {file_path}.")

def load_embeddings_from_file(file_path):
    """
    Загружает эмбеддинги из файла.

    :param file_path: Путь к файлу, откуда будут загружены эмбеддинги.
    :return: Список эмбеддингов (тензоров).
    """
    if os.path.isfile(file_path):
        embeddings = load_file(file_path)["embeddings"].to(device)
        info(f"Эмбеддинги загружены из файла {file_path}.")
        return embeddings
    else:
        return None

def prepare_embeddings(data, sbert_model, embeddings_folder=None, force_load=False):
    """
    Подготавливает эмбеддинги для всех дисциплин и сохраняет их в файл, если указана папка.
    :param data: Список дисциплин с их информацией.
    :param sbert_model: Модель Sentence-BERT для получения эмбеддингов.
    :param embeddings_folder: Папка для сохранения эмбеддингов (если указана).
    :param force_load: Флаг, указывающий, нужно ли принудительно пересчитывать эмбеддинги.
    :return: Тензор с эмбеддингами.
    """
    embeddings = None

    # Определяем путь к файлу с эмбеддингами, если указана папка
    if embeddings_folder:
        embeddings_file_path = os.path.join(embeddings_folder, "discipline_embeddings.safetensors")

        # Проверяем, нужно ли загружать существующие эмбеддинги или пересчитывать
        if not force_load and os.path.isfile(embeddings_file_path):
            embeddings = load_embeddings_from_file(embeddings_file_path)
            if embeddings is not None:
                return embeddings
            else:
                warning("Не удалось загрузить эмбеддинги. Будут созданы новые.")
        else:
            info("Принудительная загрузка включена или файл с эмбеддингами отсутствует. Начинаем вычисление эмбеддингов.")

    # Если эмбеддинги не загружены, вычисляем их
    embeddings = []
    for item in data:
        full_info = item.get('Full_Info')
        if full_info:
            discipline_embedding = embed_text(full_info, sbert_model)
            embeddings.append(discipline_embedding)

    embeddings = torch.stack(embeddings)

    # Сохраняем эмбеддинги, если указана папка
    if embeddings_folder:
        os.makedirs(embeddings_folder, exist_ok=True)
        save_embeddings_to_file(embeddings, embeddings_file_path)
        info(f"Эмбеддинги сохранены в файл {embeddings_file_path}.")

    return embeddings
