import os
import glob
import pandas as pd
from pathlib import Path
from utils.logger import info, warning
import re

def load_vacancy_data(config):
    path_to_files = config['data_path']

    # Чтение всех файлов Excel и объединение их в один DataFrame
    xlsx_files = glob.glob(os.path.join(path_to_files, "*.xlsx"))
    df = pd.DataFrame()

    for file in xlsx_files:
        df_temp = pd.read_excel(file)
        df = pd.concat([df, df_temp], ignore_index=True)

    # Извлечение года из периода изучения дисциплины и замена NaN на '0000/0000'
    # df['year'] = df['Период изучения дисциплины'].str.extract(r'(\d{4}/\d{4})').fillna('0000/0000')

    # Извлечение всех дат и выбор последней
    # Преобразуем только строки, исключив NaN
    df['year'] = df['Период изучения дисциплины'].fillna('').astype(str).str.findall(r'(\d{4}/\d{4})').apply(lambda x: x[-1] if x else '0000/0000')

    # Проверка наличия строк без года изучения (значение '0000/0000')
    missing_years = df[df['year'] == '0000/0000']
    if not missing_years.empty:
        warning(f"Обнаружены строки без информации о периоде изучения: {len(missing_years)} записей.")


    # Сортировка по 'Русскоязычное название дисциплины' и 'year', чтобы оставить самые актуальные дисциплины
    df_sorted = df.sort_values(by=['Русскоязычное название дисциплины', 'year'], ascending=[True, False])

    # Удаление дубликатов, сохраняя только самые актуальные дисциплины (с последним периодом изучения)
    df_cleaned = df_sorted.drop_duplicates(subset=['Русскоязычное название дисциплины', 'Факультет кафедры, предлагающей дисциплину'], keep='first')

    # Формирование столбца Full_Info для объединения текстовых столбцов
    df_cleaned['Full_Info'] = (
        df_cleaned['Русскоязычное название дисциплины'] +
        '\nАннотация: ' + df_cleaned['Аннотация'].fillna('') +
        '\nСписок разделов: ' + df_cleaned['Список разделов (названия и описания)'].fillna('') +
        '\nСписок планируемых результатов обучения: ' + df_cleaned['Список планируемых результатов обучения РПУДа'].fillna('')
    )

    # Группировка по факультету и сбор объединенных текстов в список дисциплин
    grouped_df = df_cleaned.groupby('Факультет кафедры, предлагающей дисциплину')['Full_Info'].apply(list).reset_index()

    return df_cleaned, grouped_df


def load_csv_files(config):
    vacancies_path = Path(config['vacancies_path'])

    # Функция для извлечения числа из имени файла
    def extract_number(path):
        match = re.search(r'part_(\d+)', path.name)
        return int(match.group(1)) if match else 0

    # Чтение всех CSV файлов и сортировка по номерам
    csv_files = sorted(list(vacancies_path.rglob("*.csv")), key=extract_number)
    info(f"Количество CSV-файлов для обработки: {len(csv_files)}")

    return csv_files

def filter_rows_by_mode(df, config):
    """
    Фильтрует строки DataFrame в зависимости от режима обработки, заданного в конфигурации.

    :param df: DataFrame с данными.
    :param config: Конфигурационный словарь с параметрами обработки.
    :return: DataFrame с отфильтрованными строками.
    """
    mode = config['processing']['mode']
    max_rows_per_file = config.get('max_rows_per_file')

    if mode == 'all':
        # Если max_rows_per_file установлено в 'ALL', возвращаем весь DataFrame
        if max_rows_per_file == 'ALL':
            return df
        else:
            return df.head(int(max_rows_per_file))

    elif mode == 'solo':
        solo_index = config['processing']['solo_index']
        info(f"Обработка единственной строки с индексом {solo_index}.")
        return df.iloc[[solo_index]]

    elif mode == 'random':
        if max_rows_per_file == 'ALL':
            info("Обработка всех строк случайным образом.")
            return df.sample(frac=1)
        else:
            info(f"Обработка {max_rows_per_file} случайных строк.")
            return df.sample(n=min(int(max_rows_per_file), len(df)))

    else:
        warning(f"Неизвестный режим обработки: {mode}. Будет обработан весь DataFrame.")
        return df
