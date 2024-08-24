import yaml
from utils.logger import info, error, success

def load_config(config_file):
    """
    Загрузка конфигурации из YAML файла.

    :param config_file: Путь к конфигурационному файлу.
    :return: Словарь с конфигурацией.
    """
    try:

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        success("Файл конфигурации успешно загружен.")
        return config

    except FileNotFoundError:
        error(f"Файл конфигурации {config_file} не найден.")
        return {}

    except yaml.YAMLError as e:
        error(f"Ошибка при чтении YAML файла: {e}")
        return {}

    except Exception as e:
        error(f"Неизвестная ошибка при загрузке конфигурации: {e}")
        return {}
