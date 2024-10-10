from utils.logger import info, error, highlight, warning
import torch

def generate_recommendations(vacancy_description, disciplines_list, model, tokenizer, config):
    """
    Используем LLaMA для генерации рекомендаций на основе топовых дисциплин.
    """
    info(f"Начинаем генерацию рекомендаций на основе {len(disciplines_list)} дисциплин")

    # Получаем параметры генерации из конфигурации
    generation_params = config['generation_params']

    # Формируем список дисциплин в виде строки
    disciplines_text = "\n".join(disciplines_list)

    # Формируем промпт для модели
    prompt = f"""
    На основе этого описания вакансии выберите наиболее подходящие дисциплины из списка ниже:

    Описание вакансии:
    {vacancy_description}

    Список возможных дисциплин:
    {disciplines_text}

    Требования к ответу:
    1. Выберите **минимум 8 дисциплин**, которые соответствуют требованиям вакансии.
    2. Дисциплины должны быть взяты **только** из предложенного списка, без повторений.
    3. Ответ должен содержать **только названия дисциплин**, без пояснений и комментариев
    4. В ответе **должны** содержаться только **подходящие** дисциплины из предложенного списка.
    5. Перечислите выбранные дисциплины через `;`, без нумерации и без использования специальных символов.

    ### START
    """
    info(f"Длина промпта: {len(prompt)} символов")

    # Преобразуем текст в токены
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        try:
            # Генерация текста на основе промпта
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_params['max_new_tokens'],
                num_return_sequences=generation_params['num_return_sequences'],
                do_sample=generation_params['do_sample'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )
            info("Генерация прошла успешно.")
            torch.cuda.empty_cache()

        except Exception as e:
            error(f"Ошибка при генерации текста: {e}")
            return None

    # Декодирование сгенерированного текста
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    info(f"Сгенерированный текст: {generated_text.strip()}")

    # Применение фильтра для извлечения данных между метками
    extracted_content = extract_generated_content(generated_text)
    # highlight(f"Извлеченное содержимое: {extracted_content}")

    return extracted_content


def extract_generated_content(generated_text):
    """
    Извлекает текст, заключенный между метками '### START' и '### END'.
    Если метка '### END' отсутствует, извлекает текст от '### START' до конца.
    Если метка '### START' отсутствует, возвращает весь текст.
    Если текст между '### START' и '### END' пустой, ищет следующий блок.
    """
    if "### START" in generated_text:
        # Если есть метка START, начинаем извлечение после неё
        extracted_content = generated_text.split("### START", 1)[1].strip()

        # Если есть метка END, обрезаем текст до неё
        if "### END" in extracted_content:
            extracted_content = extracted_content.split("### END", 1)[0].strip()

        # Проверяем, пусто ли между метками
        if not extracted_content:
            warning("Текст между метками '### START' и '### END' пустой. Ищем следующий блок текста.")

            # Ищем следующий блок текста после первого '### END'
            next_part = generated_text.split("### END", 1)[1].strip() if "### END" in generated_text else ''
            if "### START" in next_part:
                return extract_generated_content(next_part)  # Рекурсивно ищем следующий блок

        return extracted_content

    # Если нет метки START, возвращаем весь текст
    return generated_text
