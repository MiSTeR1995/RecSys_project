import os
from sentence_transformers import util
from tqdm import tqdm
import pandas as pd
from data_processing.vacancy_loader import filter_rows_by_mode
from data_processing.rec_generation import generate_recommendations
from utils.logger import info, success, warning, error, highlight, bright
from data_processing.embedding_utils import load_sbert_model, embed_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def filter_unique_items_with_minimum(sorted_items, top_n):
    """
    Фильтрует список элементов, оставляя уникальные значения до тех пор, пока их количество не достигнет top_n.
    Если после первого прохода уникальных элементов меньше чем top_n, продолжаем добавлять уникальные элементы до достижения top_n.
    """
    unique_items = []
    seen_items = set()

    for item, similarity in sorted_items:
        if item not in seen_items:
            seen_items.add(item)
            unique_items.append((item, similarity))
            if len(unique_items) >= top_n:
                break

    if len(unique_items) < top_n:
        for item, similarity in sorted_items[len(unique_items):]:
            if item not in seen_items:
                seen_items.add(item)
                unique_items.append((item, similarity))
                if len(unique_items) >= top_n:
                    break

    return unique_items



def find_top(vacancy_description, data, top_n, sbert_model=None, mode="discipline_only", top_disciplines_per_faculty=None, method="sbert"):
    """
    Универсальная функция для поиска топ-N элементов.
    :param vacancy_description: Описание вакансии.
    :param data: Либо список всех дисциплин, либо список факультетов с дисциплинами.
    :param top_n: Количество топовых элементов, которые нужно вернуть.
    :param sbert_model: Модель Sentence-BERT для получения эмбеддингов (необязательно для TF-IDF).
    :param mode: Режим работы ("discipline_only", "faculty_with_disciplines").
    :param top_disciplines_per_faculty: Количество топовых дисциплин для каждого факультета (необязательно).
    :param method: Метод для получения эмбеддингов (sbert или tfidf).
    """
    try:
        all_top_items = []

        if method == "sbert":
            # Используем Sentence-BERT для получения эмбеддингов
            vacancy_embedding = embed_text(vacancy_description, sbert_model)

        elif method == "tfidf":
            # Используем TF-IDF для получения эмбеддингов
            vectorizer = TfidfVectorizer()
            corpus = [vacancy_description] + [item.get('Full_Info', '') for item in data]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            vacancy_embedding = tfidf_matrix[0:1]

        if mode == "discipline_only":
            similarities = []

            for idx, item in enumerate(data):
                discipline_name = item.get('Русскоязычное название дисциплины')
                full_info = item.get('Full_Info')

                if full_info:
                    if method == "sbert":
                        discipline_embedding = embed_text(full_info, sbert_model)
                        similarity = util.pytorch_cos_sim(vacancy_embedding, discipline_embedding).item()
                    elif method == "tfidf":
                        discipline_embedding = tfidf_matrix[idx + 1:idx + 2]
                        similarity = cosine_similarity(vacancy_embedding, discipline_embedding)[0][0]

                    similarities.append((discipline_name, similarity))

            sorted_disciplines = sorted(similarities, key=lambda x: x[1], reverse=True)
            unique_disciplines = filter_unique_items_with_minimum(sorted_disciplines, top_n)

            highlight(f"Отсортированные топовые дисциплины по косинусному сходству ({method}):")
            for rank, (discipline, similarity) in enumerate(unique_disciplines, 1):
                success(f"{rank}. {discipline} с косинусным сходством: {similarity:.4f}")

            all_top_items.extend([disc[0] for disc in unique_disciplines])

        elif mode == "faculty_with_disciplines":
            faculty_similarities = []

            for idx, faculty_data in enumerate(data):
                faculty_name = faculty_data.get('Факультет кафедры, предлагающей дисциплину')
                subjects_descriptions = faculty_data.get('Full_Info')

                if subjects_descriptions:
                    if method == "sbert":
                        faculty_embedding = embed_text(" ".join(subjects_descriptions), sbert_model)
                        faculty_similarity = util.pytorch_cos_sim(vacancy_embedding, faculty_embedding).item()
                    elif method == "tfidf":
                        faculty_embedding = tfidf_matrix[idx + 1:idx + 2]
                        faculty_similarity = cosine_similarity(vacancy_embedding, faculty_embedding)[0][0]

                    faculty_similarities.append((faculty_name, faculty_similarity, subjects_descriptions))

            sorted_faculties = sorted(faculty_similarities, key=lambda x: x[1], reverse=True)[:top_n]

            for faculty_name, faculty_similarity, subjects_descriptions in sorted_faculties:
                highlight(f"Факультет '{faculty_name}' с косинусным сходством ({method}): {faculty_similarity:.4f}")

                discipline_similarities = []

                for description in subjects_descriptions:
                    if method == "sbert":
                        discipline_embedding = embed_text(description, sbert_model)
                        discipline_similarity = util.pytorch_cos_sim(vacancy_embedding, discipline_embedding).item()
                    elif method == "tfidf":
                        discipline_embedding = vectorizer.transform([description])
                        discipline_similarity = cosine_similarity(vacancy_embedding, discipline_embedding)[0][0]

                    discipline_similarities.append((description.split('\n')[0], discipline_similarity))

                sorted_disciplines = sorted(discipline_similarities, key=lambda x: x[1], reverse=True)

                if top_disciplines_per_faculty:
                    unique_disciplines = filter_unique_items_with_minimum(sorted_disciplines, top_disciplines_per_faculty)
                else:
                    unique_disciplines = filter_unique_items_with_minimum(sorted_disciplines, top_n)

                bright(f"Отсортированные топовые дисциплины для факультета '{faculty_name}' по косинусному сходству ({method}):")
                for rank, (discipline, similarity) in enumerate(unique_disciplines, 1):
                    success(f"{rank}. {discipline} с косинусным сходством: {similarity:.4f}")

                all_top_items.extend([disc[0] for disc in unique_disciplines])

        return all_top_items

    except Exception as e:
        error(f"Произошла ошибка при выполнении поиска топ-{top_n} ({method}): {e}")
        return []



def process_vacancies(config, csv_files, grouped_df, df_cleaned, model, tokenizer):
    # Загружаем модель Sentence-BERT на основе конфигурации
    sbert_model = load_sbert_model(config) if config.get("method", "sbert") == "sbert" else None

    # Указываем папку для сохранения результатов
    output_folder = config.get("output_folder", "results")
    os.makedirs(output_folder, exist_ok=True)

    total_rows = sum([len(pd.read_csv(file_path, sep=";", on_bad_lines="skip")) for file_path in csv_files])

    with tqdm(total=total_rows, desc="Общий прогресс обработки", unit="строка") as pbar:
        for file_path in csv_files:
            info(f"Чтение файла: {file_path}")
            try:
                df_e = pd.read_csv(file_path, sep=";")
                if df_e.empty:
                    info(f"Файл {file_path} пустой, пропускаем.")
                    continue

                df_filtered = filter_rows_by_mode(df_e, config)
                base_name = os.path.basename(file_path)
                output_file = os.path.join(output_folder, base_name)

                # Проверяем, существует ли уже файл с результатами
                file_exists = os.path.isfile(output_file)

                for index, row in df_filtered.iterrows():
                    columns_name = df_e.columns
                    parts = [str(row[col]).strip() for col in columns_name[:4] if isinstance(row[col], str) and row[col].strip()]

                    vacancy_description = ". ".join(parts)
                    info(f"Описание вакансии для строки {index}: {vacancy_description}")

                    method = config.get('method', 'sbert')

                    if config['analysis_mode'] == 'faculty_based':
                        top_disciplines = find_top(
                            vacancy_description,
                            grouped_df.to_dict('records'),
                            top_n=config['top_faculties'],
                            sbert_model=sbert_model,
                            mode="faculty_with_disciplines",
                            top_disciplines_per_faculty=config['top_disciplines_per_faculty'],
                            method=method
                        )
                    else:
                        disciplines_data = df_cleaned.to_dict('records')
                        top_disciplines = find_top(
                            vacancy_description,
                            disciplines_data,
                            top_n=config['top_disciplines'],
                            sbert_model=sbert_model,
                            mode="discipline_only",
                            method=method
                        )

                    if top_disciplines:
                        disciplines_after_sbert = "; ".join(top_disciplines)

                        # Проверяем, нужно ли использовать LLaMA для генерации рекомендаций
                        if config.get('use_llm', False):
                            recommendations = generate_recommendations(vacancy_description, top_disciplines, model, tokenizer, config)
                            df_e.at[index, "SBERT_plus_LLM_Recommendations"] = recommendations
                            highlight(f"Сгенерированные рекомендации для строки {index}: {recommendations}")
                        else:
                            df_e.at[index, "SBERT_plus_LLM_Recommendations"] = "LLM не используется"

                        df_e.at[index, "SBERT_Disciplines"] = disciplines_after_sbert

                    else:
                        warning(f"Нет подходящих дисциплин для строки {index}.")
                        df_e.at[index, "SBERT_Disciplines"] = "Нет дисциплин"
                        df_e.at[index, "SBERT_plus_LLM_Recommendations"] = "Нет рекомендаций"

                    # Сохраняем строку в файл: если файл уже существует, добавляем строку без заголовка
                    df_e.loc[[index]].to_csv(
                        output_file,
                        mode='a' if file_exists else 'w',  # 'a' для дозаписи, 'w' для записи с заголовком
                        header=not file_exists,  # Пишем заголовок только если файл не существует
                        index=False,
                        encoding="utf-8-sig",
                        sep=";"
                    )

                    # После первой итерации файла мы будем добавлять без заголовка
                    file_exists = True

                    info(f"Строка {index} сохранена в файл {output_file}.")
                    pbar.update(1)

            except Exception as e:
                error(f"Ошибка при обработке файла {file_path}: {e}")
