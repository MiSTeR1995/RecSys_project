from colorama import Fore, Style, init
import shutil

# Инициализация colorama для корректного отображения цветов в консоли
init(autoreset=True)

def info(message):
    print(Fore.CYAN + "[INFO] " + message + Style.RESET_ALL)

def success(message):
    print(Fore.GREEN + "[SUCCESS] " + message + Style.RESET_ALL)

def warning(message):
    print(Fore.YELLOW + "[WARNING] " + message + Style.RESET_ALL)

def error(message):
    print(Fore.RED + "[ERROR] " + message + Style.RESET_ALL)

def bright(message):
    print(Style.BRIGHT + Fore.WHITE + message + Style.RESET_ALL)

def highlight(message):
    """
    Функция для выделения важной информации. Использует яркий белый цвет.
    """
    print(Style.BRIGHT + Fore.MAGENTA + message + Style.RESET_ALL)

def get_progress_bar_width(percentage=0.7):
    """
    Возвращает ширину прогресс-бара на основе процента от ширины терминала.

    :param percentage: Доля от общей ширины терминала, которую должен занимать прогресс-бар.
    :return: Ширина прогресс-бара в символах.
    """
    terminal_width, _ = shutil.get_terminal_size()
    return int(terminal_width * percentage)

def get_plural_form(n, singular, few, many):
    if 11 <= n % 100 <= 19:
        return many
    if n % 10 == 1:
        return singular
    if 2 <= n % 10 <= 4:
        return few
    return many
