from utils.config_loader import load_config
from data_processing.vacancy_loader import load_vacancy_data, load_csv_files
from data_processing.analysis import process_vacancies
from data_processing.model_loader import load_model

def main():
    # Загрузка конфигурации
    config = load_config("config.yaml")

    # Загрузка данных вакансий и факультетов
    df_cleaned, grouped_df = load_vacancy_data(config)

    # Проверка, нужно ли использовать модель LLaMA
    if config.get('use_llm', False):
        # Загрузка модели LLaMA
        model_path = config['model_path']
        model, tokenizer = load_model(model_path)

        # Проверка на успешную загрузку модели и токенизатора
        if model is None or tokenizer is None:
            raise RuntimeError("Не удалось загрузить модель или токенизатор.")
    else:
        model, tokenizer = None, None  # Если не нужно использовать модель, оставляем пустыми

    # Загрузка CSV файлов с вакансиями
    csv_files = load_csv_files(config)

    # Обработка вакансий и генерация рекомендаций для каждого файла
    process_vacancies(config, csv_files, grouped_df, df_cleaned, model, tokenizer)

if __name__ == "__main__":
    main()
