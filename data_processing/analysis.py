import os
import torch
from sentence_transformers import util
from tqdm import tqdm
import pandas as pd
from data_processing.vacancy_loader import filter_rows_by_mode
from data_processing.rec_generation import generate_recommendations
from utils.logger import info, success, warning, error, highlight, bright, get_plural_form
from data_processing.embedding_utils import load_sbert_model, embed_text, prepare_embeddings
from data_processing.sorting_utils import sort_sbert_disciplines, sort_recommendations
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

def find_top(vacancy_description, data, top_n, sbert_model=None, method="sbert", embeddings=None, config=None):
    """
    Универсальная функция для поиска топ-N элементов.

    :param vacancy_description: Описание вакансии.
    :param data: Либо список всех дисциплин, либо список факультетов с дисциплинами.
    :param top_n: Количество топовых элементов, которые нужно вернуть.
    :param sbert_model: Модель Sentence-BERT для получения эмбеддингов (необязательно для TF-IDF).
    :param method: Метод для получения эмбеддингов (sbert или tfidf).
    :param embeddings: Повторно используемые эмбеддинги (если есть), чтобы избежать повторных вычислений.
    :param config: Конфигурационный словарь, который может содержать дополнительные параметры.

    :return: Список топовых элементов, список имен топовых дисциплин и эмбеддинги для повторного использования.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        sim = torch.nn.CosineSimilarity()

        all_top_items = []
        top_discipline_names = []

        # Вычисляем эмбеддинг для описания вакансии
        if method == "sbert":
            vacancy_embedding = embed_text(vacancy_description, sbert_model)
        elif method == "tfidf":
            vectorizer = TfidfVectorizer()
            corpus = [vacancy_description] + [item.get('Full_Info', '') for item in data]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            vacancy_embedding = tfidf_matrix[0:1]
        else:
            error("Поддерживаемые методы: 'sbert' и 'tfidf'.")
            return [], [], embeddings

        names = [item.get('Русскоязычное название дисциплины') for item in data]

        # Вычисляем косинусное сходство
        if method == "sbert":
            with torch.no_grad():
                similarities = sim(vacancy_embedding, embeddings).cpu().tolist()
                similarities = [(i, j) for i, j in zip(names, similarities)]
        elif method == "tfidf":
            similarities = [(names[idx], similarity) for idx, similarity in enumerate(cosine_similarity(vacancy_embedding, tfidf_matrix[1:])[0])]

        sorted_disciplines = sorted(similarities, key=lambda x: x[1], reverse=True)
        unique_disciplines = filter_unique_items_with_minimum(sorted_disciplines, top_n)

        highlight(f"Отсортированные топовые дисциплины по косинусному сходству ({method}):")
        for rank, (discipline, similarity) in enumerate(unique_disciplines, 1):
            success(f"{rank}. {discipline} с косинусным сходством: {similarity:.4f}")
            match = next((item for item in data if item.get('Русскоязычное название дисциплины') == discipline), None)

            if match:
                discipline_id = match.get('ID дисциплины БУП ППК (АСАВ)', '-')
                campus = match.get('Кампус кафедры, предлагающей дисциплину', '-')
                faculty = match.get('Факультет кафедры, предлагающей дисциплину', '-')
                department = match.get('Кафедра, предлагающая дисциплину', '-')
                level = match.get('Уровень обучения', '-')
                period = match.get('Период изучения дисциплины', '-')
                coverage = match.get('Охват аудитории', '-')
                form = match.get('Формат изучения', '-')

                formatted_discipline = f"{discipline_id} | {discipline} | CS={similarity:.4f} | {campus} | {faculty} | {department} | {level} | {period} | {coverage} | {form}"
                all_top_items.append(formatted_discipline)
            else:
                all_top_items.append(f"- | {discipline} | CS={similarity:.4f}")

        top_discipline_names.extend([disc[0] for disc in unique_disciplines])

        return all_top_items, top_discipline_names, embeddings

    except Exception as e:
        error(f"Произошла ошибка при выполнении поиска топ-{top_n} ({method}): {e}")
        return [], [], embeddings

def process_vacancies(config, csv_files, grouped_df, df_cleaned, model, tokenizer):
    try:
        # Загружаем модель Sentence-BERT на основе конфигурации
        sbert_model = load_sbert_model(config) if config.get("method", "sbert") == "sbert" else None
        embeddings_folder = config.get("embeddings_folder", None)
        embeddings = None

        if config.get("method") == "sbert" and embeddings_folder:
            # Подготавливаем эмбеддинги (пересчет или загрузка в зависимости от флага)
            force_load = config.get("force_load_embeddings", False)
            embeddings = prepare_embeddings(df_cleaned.to_dict('records'), sbert_model, embeddings_folder, force_load=force_load)
        else:
            info("Метод не является 'sbert'. Эмбеддинги не будут пересчитаваться или загружаться из файла.")

        # Указываем папку для сохранения результатов
        output_folder = config.get("output_folder", "results")
        os.makedirs(output_folder, exist_ok=True)

        # Получаем уже обработанные ID
        enable_id_check = config.get('processing', {}).get('enable_id_check', False)
        processed_ids_dict = get_processed_ids(csv_files, output_folder, enable_id_check)

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

        # Преобразование описания вакансии
        parts = [str(row.iloc[col]).strip() for col in range(1, 5) if isinstance(row.iloc[col], str) and row.iloc[col].strip()]
        vacancy_description = ". ".join(parts)
        bright(f"Описание вакансии {base_name} (ID: {vacancy_id}) для строки {index}: {vacancy_description}")

        method = config.get('method', 'sbert')

        disciplines_data = df_cleaned.to_dict('records')
        full_top_discipline_info, top_discipline_names, embeddings = find_top(
            vacancy_description,
            disciplines_data,
            top_n=config['top_disciplines'],
            sbert_model=sbert_model,
            method=method,
            embeddings=embeddings,
            config=config
        )

        if full_top_discipline_info:
            sorted_disciplines_str = sort_sbert_disciplines("; ".join(full_top_discipline_info))
            row["SBERT_Disciplines"] = sorted_disciplines_str

            if config.get('use_llm', False):

                recommendations = generate_recommendations(vacancy_description, top_discipline_names, model, tokenizer, config)

                sorted_recommendations_str = sort_recommendations(recommendations, full_top_discipline_info)

                row["SBERT_plus_LLM_Recommendations"] = sorted_recommendations_str
                highlight(f"Сгенерированные и отсортированные рекомендации для строки {index} (ID: {vacancy_id}): {sorted_recommendations_str}")

            else:
                row["SBERT_plus_LLM_Recommendations"] = "LLM не используется"
        else:
            warning(f"Нет подходящих дисциплин для строки {index} (ID: {vacancy_id}).")
            row["SBERT_Disciplines"] = "Нет дисциплин"
            row["SBERT_plus_LLM_Recommendations"] = "Нет рекомендаций"

        # Запись в файл
        output_file = os.path.join(output_folder, base_name)
        row.to_frame().T.to_csv(output_file, mode='a', header=not os.path.isfile(output_file), index=False, encoding="utf-8-sig", sep=";")
        processed_ids.add(vacancy_id)
        info(f"Строка {index} (ID: {vacancy_id}) сохранена в файл {output_file}.")

        return embeddings

    except Exception as e:
        error(f"Ошибка при обработке строки {index} (ID: {vacancy_id}) в файле {file_path}: {e}")
        return embeddings
