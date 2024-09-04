import os
import torch
from sentence_transformers import util
from tqdm import tqdm
import pandas as pd
from data_processing.vacancy_loader import filter_rows_by_mode
from data_processing.rec_generation import generate_recommendations
from utils.logger import info, success, warning, error, highlight, bright, get_plural_form
from data_processing.embedding_utils import load_sbert_model, embed_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_processed_ids(csv_files, output_folder, enable_id_check=True):
    """
    Получает уже обработанные ID из выходных файлов.

    :param csv_files: Список входных файлов (CSV).
    :param output_folder: Папка для сохранения результатов.
    :param enable_id_check: Если True, выполняется проверка уже обработанных ID.
    :return: Словарь, где ключи — имена файлов, а значения — множества с обработанными ID.
    """
    processed_ids_dict = {}

    if not enable_id_check:
        info("Проверка ID отключена. Все строки будут обработаны.")
        return processed_ids_dict  # Возвращаем пустой словарь, если проверка отключена

    for file_path in csv_files:
        base_name = os.path.basename(file_path)
        output_file = os.path.join(output_folder, base_name)
        if os.path.isfile(output_file):
            try:
                df_output = pd.read_csv(output_file, sep=";", usecols=[0])  # Предполагается, что ID — первый столбец
                processed_ids = set(df_output.iloc[:, 0].astype(str).tolist())
                count = len(processed_ids)
                info(f"Найдено совпадение по ID {count} ранее {get_plural_form(count, 'обработанной строки', 'обработанных строк', 'обработанных строк')} в файле {output_file}.")
                processed_ids_dict[base_name] = processed_ids
            except Exception as e:
                warning(f"Не удалось прочитать файл {output_file} для загрузки обработанных ID: {e}")
                processed_ids_dict[base_name] = set()
        else:
            processed_ids_dict[base_name] = set()
            info(f"Файл {output_file} не найден. Начнем обработку с нуля.")

    return processed_ids_dict

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

def find_top(vacancy_description, data, top_n, sbert_model=None, mode="discipline_only", top_disciplines_per_faculty=None, method="sbert", embeddings=None):
    """
    Универсальная функция для поиска топ-N элементов.
    :param vacancy_description: Описание вакансии.
    :param data: Либо список всех дисциплин, либо список факультетов с дисциплинами.
    :param top_n: Количество топовых элементов, которые нужно вернуть.
    :param sbert_model: Модель Sentence-BERT для получения эмбеддингов (необязательно для TF-IDF).
    :param mode: Режим работы ("discipline_only", "faculty_with_disciplines").
    :param top_disciplines_per_faculty: Количество топовых дисциплин для каждого факультета (необязательно).
    :param method: Метод для получения эмбеддингов (sbert или tfidf).
    :param embeddings: Повторно используемые эмбеддинги (если есть), чтобы избежать повторных вычислений.
    :return: Список топовых элементов и эмбеддинги для повторного использования.
    """
    try:
        # Используем CosineSimilarity от PyTorch для вычисления сходства
        sim = torch.nn.CosineSimilarity()
        create_new_embeddings = False

        # Если эмбеддинги еще не были вычислены, инициализируем их
        if embeddings is None:
            embeddings = []
            create_new_embeddings = True

        all_top_items = []

        if method == "sbert":
            # Используем Sentence-BERT для получения эмбеддингов вакансии
            vacancy_embedding = embed_text(vacancy_description, sbert_model)

        elif method == "tfidf":
            # Используем TF-IDF для получения эмбеддингов
            vectorizer = TfidfVectorizer()
            corpus = [vacancy_description] + [item.get('Full_Info', '') for item in data]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            vacancy_embedding = tfidf_matrix[0:1]

        if mode == "discipline_only":
            names = []

            # Обрабатываем каждый элемент данных
            for idx, item in enumerate(data):
                discipline_name = item.get('Русскоязычное название дисциплины')
                full_info = item.get('Full_Info')

                if full_info:
                    if method == "sbert":
                        if create_new_embeddings:  # Если эмбеддинги создаются впервые
                            discipline_embedding = embed_text(full_info, sbert_model)
                            embeddings.append(discipline_embedding)  # Добавляем эмбеддинг в список
                    elif method == "tfidf":
                        discipline_embedding = tfidf_matrix[idx + 1:idx + 2]
                        similarity = cosine_similarity(vacancy_embedding, discipline_embedding)[0][0]

                    names.append(discipline_name)

            # Если эмбеддинги были созданы впервые, преобразуем их в тензор
            if create_new_embeddings:
                embeddings = torch.stack(embeddings)

            # Вычисляем косинусное сходство
            with torch.no_grad():
                similarities = sim(vacancy_embedding, embeddings).cpu().tolist()
                similarities = [(i, j) for i, j in zip(names, similarities)]

            # Сортируем по сходству
            sorted_disciplines = sorted(similarities, key=lambda x: x[1], reverse=True)
            unique_disciplines = filter_unique_items_with_minimum(sorted_disciplines, top_n)

            highlight(f"Отсортированные топовые дисциплины по косинусному сходству ({method}):")
            for rank, (discipline, similarity) in enumerate(unique_disciplines, 1):
                success(f"{rank}. {discipline} с косинусным сходством: {similarity:.4f}")

            all_top_items.extend([disc[0] for disc in unique_disciplines])

        elif mode == "faculty_with_disciplines":
            faculty_names = []
            faculty_embeddings = [] if create_new_embeddings else embeddings  # Используем или создаём эмбеддинги факультетов

            # Обрабатываем каждый факультет
            for idx, faculty_data in enumerate(data):
                faculty_name = faculty_data.get('Факультет кафедры, предлагающей дисциплину')
                subjects_descriptions = faculty_data.get('Full_Info')

                if subjects_descriptions:
                    if method == "sbert":
                        if create_new_embeddings:  # Если эмбеддинги создаются впервые
                            faculty_embedding = embed_text(" ".join(subjects_descriptions), sbert_model)
                            faculty_embeddings.append(faculty_embedding)  # Добавляем эмбеддинг в список
                    elif method == "tfidf":
                        faculty_embedding = tfidf_matrix[idx + 1:idx + 2]
                        faculty_similarity = cosine_similarity(vacancy_embedding, faculty_embedding)[0][0]

                    faculty_names.append(faculty_name)

            # Если эмбеддинги были созданы впервые, сохраняем их для факультетов
            if create_new_embeddings:
                embeddings = torch.stack(faculty_embeddings)

            # Вычисляем косинусное сходство для факультетов
            with torch.no_grad():
                faculty_similarities = sim(vacancy_embedding, embeddings).cpu().tolist()
                faculty_similarities = [(i, j) for i, j in zip(faculty_names, faculty_similarities)]

            # Сортируем по сходству и выбираем топовые факультеты
            sorted_faculties = sorted(faculty_similarities, key=lambda x: x[1], reverse=True)[:top_n]

            for faculty_name, faculty_similarity in sorted_faculties:
                highlight(f"Факультет '{faculty_name}' с косинусным сходством ({method}): {faculty_similarity:.4f}")

                discipline_similarities = []

                # Сравниваем дисциплины внутри факультета
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

        # Возвращаем список топовых дисциплин и эмбеддинги для повторного использования
        return all_top_items, embeddings

    except Exception as e:
        # Обработка ошибок и исключений
        error(f"Произошла ошибка при выполнении поиска топ-{top_n} ({method}): {e}")
        return [], embeddings

def process_vacancies(config, csv_files, grouped_df, df_cleaned, model, tokenizer):
    try:
        # Загружаем модель Sentence-BERT на основе конфигурации
        sbert_model = load_sbert_model(config) if config.get("method", "sbert") == "sbert" else None

        # Указываем папку для сохранения результатов
        output_folder = config.get("output_folder", "results")
        os.makedirs(output_folder, exist_ok=True)

        # Получаем уже обработанные ID
        enable_id_check = config.get('processing', {}).get('enable_id_check', False)
        processed_ids_dict = get_processed_ids(csv_files, output_folder, enable_id_check)

        # Инициализация переменной для эмбеддингов (используется для повторного использования)
        embeddings = None

        # Создаем итераторы для каждой строки в каждом файле
        iterators = []
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=";")
            df_filtered = filter_rows_by_mode(df, config)
            iterators.append((file_path, iter(df_filtered.iterrows())))

        total_rows = sum(len(pd.read_csv(file_path, sep=";")) for file_path in csv_files)
        with tqdm(total=total_rows, desc="Общий прогресс обработки", unit="строка") as pbar:
            while iterators:
                for file_path, iterator in iterators[:]:  # копия списка, чтобы изменять его во время итерации
                    try:
                        index, row = next(iterator)
                        # Обработка строки
                        embeddings = process_row(file_path, index, row, processed_ids_dict, sbert_model, grouped_df, df_cleaned, model, tokenizer, output_folder, config, embeddings)
                        pbar.update(1)
                    except StopIteration:
                        iterators.remove((file_path, iterator))  # Удаляем итератор, если файл завершен
                    except Exception as e:
                        error(f"Ошибка при обработке файла {file_path} на строке {index}: {e}")

    except Exception as e:
        error(f"Ошибка при инициализации процесса обработки вакансий: {e}")

def process_row(file_path, index, row, processed_ids_dict, sbert_model, grouped_df, df_cleaned, model, tokenizer, output_folder, config, embeddings):
    try:
        base_name = os.path.basename(file_path)
        processed_ids = processed_ids_dict.get(base_name, set())

        # Предполагается, что первый столбец — ID
        vacancy_id = str(row.iloc[0]).strip()
        if config.get('processing', {}).get('enable_id_check', False) and vacancy_id in processed_ids:
            return embeddings  # Пропускаем уже обработанные строки

        parts = [str(row.iloc[col]).strip() for col in range(1, 5) if isinstance(row.iloc[col], str) and row.iloc[col].strip()]
        vacancy_description = ". ".join(parts)
        bright(f"Описание вакансии {base_name} (ID: {vacancy_id}) для строки {index}: {vacancy_description}")

        method = config.get('method', 'sbert')
        if config['analysis_mode'] == 'faculty_based':
            top_disciplines, embeddings = find_top(
                vacancy_description,
                grouped_df.to_dict('records'),
                top_n=config['top_faculties'],
                sbert_model=sbert_model,
                mode="faculty_with_disciplines",
                top_disciplines_per_faculty=config['top_disciplines_per_faculty'],
                method=method,
                embeddings=embeddings  # Передаем и обновляем эмбеддинги
            )
        else:
            disciplines_data = df_cleaned.to_dict('records')
            top_disciplines, embeddings = find_top(
                vacancy_description,
                disciplines_data,
                top_n=config['top_disciplines'],
                sbert_model=sbert_model,
                mode="discipline_only",
                method=method,
                embeddings=embeddings  # Передаем и обновляем эмбеддинги
            )

        if top_disciplines:
            disciplines_after_sbert = "; ".join(top_disciplines)
            if config.get('use_llm', False):
                recommendations = generate_recommendations(vacancy_description, top_disciplines, model, tokenizer, config)
                row["SBERT_plus_LLM_Recommendations"] = recommendations
                highlight(f"Сгенерированные рекомендации для строки {index} (ID: {vacancy_id}): {recommendations}")
            else:
                row["SBERT_plus_LLM_Recommendations"] = "LLM не используется"
            row["SBERT_Disciplines"] = disciplines_after_sbert
        else:
            warning(f"Нет подходящих дисциплин для строки {index} (ID: {vacancy_id}).")
            row["SBERT_Disciplines"] = "Нет дисциплин"
            row["SBERT_plus_LLM_Recommendations"] = "Нет рекомендаций"

        output_file = os.path.join(output_folder, base_name)
        row.to_frame().T.to_csv(output_file, mode='a', header=not os.path.isfile(output_file), index=False, encoding="utf-8-sig", sep=";")
        processed_ids.add(vacancy_id)
        info(f"Строка {index} (ID: {vacancy_id}) сохранена в файл {output_file}.")

        return embeddings  # Возвращаем обновленные эмбеддинги

    except Exception as e:
        error(f"Ошибка при обработке строки {index} (ID: {vacancy_id}) в файле {file_path}: {e}")
        return embeddings
