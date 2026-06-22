import os
import yaml
import logging
import logging.handlers
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SETTINGS_PATH = ROOT_DIR / "settings.yaml"

# Carga .env si existe (desarrollo local). En producción las vars vienen del sistema.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env", override=False)
except ImportError:
    pass


def _bool(val: str) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró settings.yaml en {SETTINGS_PATH}. "
            "Asegúrate de tener el archivo de configuración."
        )
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Variables de entorno sobreescriben el YAML (útil en Docker / EasyPanel)
    overrides = {
        ("exchange", "api_key"):      os.getenv("EXCHANGE_API_KEY"),
        ("exchange", "api_secret"):   os.getenv("EXCHANGE_API_SECRET"),
        ("exchange", "testnet"):      os.getenv("EXCHANGE_TESTNET"),
        ("telegram", "token"):        os.getenv("TELEGRAM_TOKEN"),
        ("telegram", "chat_id"):      os.getenv("TELEGRAM_CHAT_ID"),
        ("telegram", "enabled"):      os.getenv("TELEGRAM_ENABLED"),
        ("trading", "mode"):          os.getenv("TRADING_MODE"),
        ("trading", "capital"):       os.getenv("TRADING_CAPITAL"),
        ("trading", "symbol"):        os.getenv("TRADING_SYMBOL"),
    }
    for (section, key), val in overrides.items():
        if val is None:
            continue
        if key in ("testnet", "enabled"):
            val = _bool(val)
        elif key == "capital":
            val = float(val)
        cfg[section][key] = val

    return cfg


def setup_logging(settings: dict) -> None:
    log_cfg = settings.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = ROOT_DIR / log_cfg.get("file", "logs/trading.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # Archivo con rotación (máximo 5MB, guarda 3 archivos)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


# Carga global al importar el módulo
settings = load_settings()
setup_logging(settings)

# Accesos directos a las secciones más usadas
TRADING = settings["trading"]
RISK = settings["risk"]
MODEL = settings["model"]
EXCHANGE = settings["exchange"]
TELEGRAM = settings["telegram"]
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
