from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from utils.logger import info, success, error

def load_model(model_path, quantization=True):
    """
    Загрузка модели и токенизатора с применением квантования и ускорения на GPU.

    :param model_path: Путь к модели.
    :param quantization: Использовать ли квантование (по умолчанию 8-битное).
    :return: Загруженная модель и токенизатор.
    """
    try:
        info(f"Загрузка модели и токенизатора из: {model_path}")

        # Создание конфигурации квантования
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=quantization,  # Используем 8-битный режим по умолчанию
        )

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        info("Токенизатор успешно загружен.")

        # Загрузка модели с использованием конфигурации
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            # quantization_config=quantization_config,
            torch_dtype=torch.float16,  # Использование FP16 для ускорения на GPU
            # attn_implementation="flash_attention_2"
        )

        model.eval()
        success("Модель успешно загружена и готова к использованию.")

        return model, tokenizer

    except FileNotFoundError:
        error(f"Модель по пути {model_path} не найдена.")
        return None, None

    except Exception as e:
        error(f"Ошибка при загрузке модели или токенизатора: {e}")
        return None, None
