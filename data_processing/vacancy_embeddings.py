import os
import torch
import pandas as pd
import polars as pl
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
    parquet_file_path = os.path.join(parquet_folder, "vacancies_cleaned.parquet")

    # Проверка на существование файла с эмбеддингами вакансий
    if not config.get("force_load_vacancy_embeddings", False) and (os.path.isfile(embeddings_file_path)):
        info(f"Эмбеддинги вакансий уже существуют, пропускаем их формирование.")
        return load_embeddings_from_file(embeddings_file_path)  # Загружаем эмбеддинги из файла

    # Загрузка списка CSV-файлов
    csv_files = load_csv_files(config)
    if not csv_files:
        info("Список CSV-файлов пуст, обработка прекращена.")
        return

    # Проверка на существование Parquet-файла
    if not config.get("force_load_vacancies", False) and os.path.isfile(parquet_file_path):
        info(f"Очищенный Parquet-файл уже существует: {parquet_file_path}")
    else:
        # Объединяем все данные из файлов
        all_data = []
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=";")
            all_data.append(df)

        # Создаём общий DataFrame из всех файлов
        combined_df = pd.concat(all_data, ignore_index=True)

        # Удаляем дубликаты по 'Name' и 'Description'
        unique_df = combined_df.drop_duplicates(subset=['Name', 'Description'])

        # Сохраняем уникальные данные в формате Parquet
        os.makedirs(parquet_folder, exist_ok=True)
        unique_df.to_parquet(parquet_file_path, index=False)

        info(f"Очищенные данные сохранены в формате Parquet: {parquet_file_path}")

    # Загружаем очищенный Parquet-файл с использованием Polars
    df = pl.read_parquet(parquet_file_path)
    info(f"Parquet-файл с вакансиями загружен. Количество строк: {df.shape[0]}")

    #Генерация эмбедингов
    embeddings = []
    for row in tqdm(df.iter_rows(named=True), total=df.shape[0], desc="Обработка вакансий"):
        # Извлекаем данные из строки
        name = row.get('Name', '')
        description = row.get('Description', '')
        key_skills = row.get('KeySkills', '')
        professional_roles = row.get('ProfessionalRoles', '')

        # Формируем текст, объединяя Name, Description, KeySkills и ProfessionalRoles
        text = f"{name}\n{description}\n{key_skills}\n{professional_roles}"

        # Проверяем, если текст пустой, пропускаем вакансию
        if not text.strip():
            print("Пропуск вакансии из-за отсутствия данных.\n")
            continue

        # Создаём эмбеддинг
        vacancy_embedding = embed_text(text, sbert_model)
        embeddings.append(vacancy_embedding)

    # Сохранение эмбеддингов в формате .safetensors
    embeddings = torch.stack(embeddings)
    os.makedirs(embeddings_folder, exist_ok=True)
    save_embeddings_to_file(embeddings, embeddings_file_path)

    return embeddings
