import os
import torch
import pandas as pd
from tqdm import tqdm
from data_processing.embedding_utils import load_sbert_model, embed_text, save_embeddings_to_file, load_embeddings_from_file
from data_processing.vacancy_loader import load_csv_files
from utils.logger import info, warning

def compute_and_save_vacancy_embeddings(config):
    """
    Вычисляет и сохраняет эмбеддинги для вакансий на основе полей Name, Description, KeySkills.
    :param config: Конфигурационный словарь.
    """
    # Загрузка модели
    sbert_model = load_sbert_model(config)

    # Путь для сохранения эмбеддингов
    embeddings_folder = config.get("embeddings_folder", "./embeddings")
    embeddings_file_path = os.path.join(embeddings_folder, "vacancy_embeddings.safetensors")
    parquet_folder = config.get("parquet_data_folder", "./data/parquets/")
    parquet_file_path = os.path.join(parquet_folder, "vacancies.parquet")

    # Проверка на существование файла с эмбеддингами вакансий
    if not config.get("force_load_vacancy_embeddings", False) and (os.path.isfile(embeddings_file_path) and os.path.isfile(parquet_file_path)):
        info(f"Эмбеддинги вакансий и их Parquet файл уже существуют, пропускаем их формирование.")
        return load_embeddings_from_file(embeddings_file_path)  # Загружаем эмбеддинги из файла

    # Если файл не найден или требуется пересчет, пересчитываем эмбеддинги
    vacancies_data = []  # Список для хранения данных вакансий (без эмбеддингов)
    embeddings = []
    csv_files = load_csv_files(config)

    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=";")

        # Добавляем прогресс-бар для обработки строк DataFrame
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Обработка {file_path}"):
            # Проверка наличия столбцов и замена NaN значений на пустую строку
            vacancy_id = int(row.get('ID', 0))  # Предполагаем, что ID — целое число
            name = str(row.get('Name', '') or '')
            description = str(row.get('Description', '') or '')
            key_skills = str(row.get('KeySkills', '') or '')
            professional_roles = str(row.get('ProfessionalRoles', '') or '')

            # Формируем текст, объединяя Name, Description, KeySkills и ProfessionalRoles
            text = f"{name}\n{description}\n{key_skills}\n{professional_roles}"

            # Проверяем, если текст пустой, пропускаем вакансию
            if not text.strip():
                print("Пропуск вакансии из-за отсутствия данных.\n")
                continue

            vacancy_embedding = embed_text(text, sbert_model)
            embeddings.append(vacancy_embedding)

            # Сохраняем данные вакансии (без эмбеддинга)
            vacancies_data.append({
                'ID': vacancy_id,
                'Name': name,
                'Description': description,
                'KeySkills': key_skills,
                'ProfessionalRoles': professional_roles
            })

    # Сохранение эмбеддингов в формате .safetensors
    embeddings = torch.stack(embeddings)
    os.makedirs(embeddings_folder, exist_ok=True)
    save_embeddings_to_file(embeddings, embeddings_file_path)

    # Создаем DataFrame и сохраняем его в формате Parquet
    os.makedirs(parquet_folder, exist_ok=True)
    vacancies_df = pd.DataFrame(vacancies_data)

    # Проверка на существование файла и флаг force_load

    vacancies_df.to_parquet(parquet_file_path, index=False)
    info(f"Вакансии сохранены в формате Parquet в файл {parquet_file_path}.")

    return embeddings
