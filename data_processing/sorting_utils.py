import re

# Приоритеты для сортировки по категориям
priority = {
    "Бакалавриат": 0,
    "Специалитет": 1,
    "Бакалавриат, Специалитет": 2,
    "Бакалавриат, Магистратура": 3,
    "Магистратура": 4,
    "Аспирантура": 5,
    "-": 6,
    "nan": 7  # если встречается nan
}

def extract_numbers(value):
    """Функция для извлечения числовых значений модулей."""
    # Ищем только числа перед словом "модуль"
    numbers = re.findall(r'(\d+)\s*модуль', value)
    return [int(num) for num in numbers] if numbers else None

def custom_sort_key(r):
    """Кастомная функция сортировки для дисциплин."""
    l = r.strip().split("|")

    # Извлекаем категорию обучения
    category = l[6].strip() if len(l) > 6 else "nan"

    # Проверяем приоритет для комбинаций категорий
    category_priority = priority.get(category, priority["nan"])

    # Извлекаем модули для сортировки
    numbers = extract_numbers(l[7].strip()) if len(l) > 7 else None
    if numbers is None:
        first_module, last_module = float('inf'), float('inf')
    else:
        first_module = numbers[0]
        last_module = numbers[-1]  # Учитываем последний модуль для сортировки

    # Возвращаем ключ для сортировки:
    # 1. По приоритету категории (включая комбинации категорий)
    # 2. По первому модулю (меньший модуль выше)
    # 3. По последнему модулю (чем раньше завершение, тем выше)
    return (category_priority, first_module, last_module)


def sort_sbert_disciplines(disciplines_str):
    """Функция для сортировки дисциплин."""
    # Разбиваем строку дисциплин на список
    disciplines = [d.strip() for d in disciplines_str.split(";")]

    # Сортируем дисциплины по кастомному ключу
    sorted_disciplines = sorted(disciplines, key=custom_sort_key)

    # Объединяем отсортированные дисциплины обратно в строку
    sorted_discipline_str = "; ".join(sorted_disciplines)

    return sorted_discipline_str

def parse_recommendations(recommendations_str):
    """Функция для парсинга рекомендаций."""
    recommendations_str = re.sub(r'^\d+\.\s*', '', recommendations_str.strip())
    recommendations = [rec.strip() for rec in re.split(r';|\n', recommendations_str) if rec.strip()]
    recommendations = list(dict.fromkeys(recommendations))  # Убираем дубликаты
    return recommendations

def sort_recommendations(recommendations_str, sorted_disciplines):
    """Функция для сортировки рекомендаций по тому же принципу, что и дисциплины."""
    recommendations = parse_recommendations(recommendations_str)

    # Извлекаем названия дисциплин и сопоставляем их с полными строками
    discipline_name_to_full_row = {s.split("|")[1].strip(): s.strip() for s in sorted_disciplines}

    # Формат строки, если дисциплина не найдена
    def generate_not_found_row(rec):
        return f"------ | {rec} | ------ | ------ | ------ | ------ | ------ | ------"

    # Сортируем рекомендации, если они найдены, применяя `custom_sort_key`
    sorted_recommendations = sorted(
        [discipline_name_to_full_row.get(rec, generate_not_found_row(rec)) for rec in recommendations],
        key=lambda r: custom_sort_key(r)
    )

    return "; ".join(sorted_recommendations)
