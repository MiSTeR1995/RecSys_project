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

    # Удаление дубликатов и объединение текстовых столбцов
    df_cleaned = df.drop_duplicates(subset=['Русскоязычное название дисциплины', 'Факультет кафедры, предлагающей дисциплину']).copy()
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
    max_rows_per_file = config.get('max_rows_per_file', 10)

    if mode == 'all':
        # info("Обработка строк в режиме ALL")
        return df if max_rows_per_file is None else df.head(max_rows_per_file)

    elif mode == 'solo':
        solo_index = config['processing']['solo_index']
        info(f"Обработка единственной строки с индексом {solo_index}.")
        return df.iloc[[solo_index]]

    elif mode == 'random':
        if max_rows_per_file:
            info(f"Обработка {max_rows_per_file} случайных строк.")
            return df.sample(n=min(max_rows_per_file, len(df)))
        else:
            info("Обработка всех строк случайным образом.")
            return df.sample(frac=1)

    else:
        warning(f"Неизвестный режим обработки: {mode}. Будет обработан весь DataFrame.")
        return df
