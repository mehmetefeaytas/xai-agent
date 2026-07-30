"""Merkezî yapılandırma: yollar, sabitler ve LLM ayarları.

Bu modül projenin tek doğruluk kaynağıdır. Başka hiçbir modül yol veya
sabit değeri kendi içinde tanımlamaz; hepsi buradan okur. Böylece
"eğitimde şu tohum, testte bu tohum" tipi sessiz hatalar önlenir.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Yollar
# --------------------------------------------------------------------------
# config.py -> src/xai_agent/config.py olduğu için iki seviye yukarı çıkınca
# proje köküne varırız.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV = RAW_DIR / "german_credit.csv"
MODEL_PATH = MODELS_DIR / "lightgbm_credit.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "global_importance.json"
FAITHFULNESS_PATH = ARTIFACTS_DIR / "faithfulness_report.json"
FAIRNESS_PATH = ARTIFACTS_DIR / "fairness_report.json"

_ALL_DIRS = (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR)


def ensure_dirs() -> None:
    """Gerekli tüm klasörleri oluşturur (varsa dokunmaz)."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Veri / model sabitleri
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.20  # eğitim setinin içinden erken durdurma için ayrılan pay
CV_FOLDS = 5

TARGET_COL = "class"
#: Pozitif sınıf = "riskli / temerrüde düşen". Bu seçim SHAP işaretlerinin
#: yönünü belirler: POZİTİF SHAP => riski ARTIRIR.
POSITIVE_LABEL = "bad"
NEGATIVE_LABEL = "good"

#: German Credit veri setinin resmî maliyet matrisi (UCI dokümanı):
#: kötü bir müşteriyi "iyi" sanmak, iyi bir müşteriyi "kötü" sanmaktan 5 kat pahalı.
COST_FALSE_NEGATIVE = 5.0  # gerçek bad -> good tahmin edildi
COST_FALSE_POSITIVE = 1.0  # gerçek good -> bad tahmin edildi

#: Yasal/etik olarak kredi kararında kullanılmaması gereken özellikler.
#: Varsayılan olarak modelden DIŞLANIR; yalnızca adalet denetiminde kullanılır.
PROTECTED_FEATURES: tuple[str, ...] = ("personal_status", "foreign_worker")

#: Modelde tutulan ama adalet denetiminde izlenen hassas özellikler.
SENSITIVE_FEATURES: tuple[str, ...] = ("age",)

#: Modelden korunan özellikleri düşür (etik varsayılan).
DROP_PROTECTED = True

#: Karar eşiği: eğitimde maliyet-optimal olarak öğrenilir, buradaki değer
#: yalnızca model yüklenemediğinde kullanılan yedektir.
DEFAULT_THRESHOLD = 0.5

#: Hiperparametreler 5-kat CV ile seçildi (bkz. artifacts/metrics.json).
#: Sığ ağaçlar bilinçli bir tercih: 800 satırlık eğitim setinde derin ağaçlar
#: ezberler ve SHAP etkileşimleri yorumlanamaz hâle gelir.
LGBM_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 600,  # üst sınır; gerçek sayı erken durdurma ile bulunur
    "learning_rate": 0.05,
    "num_leaves": 4,
    "max_depth": 3,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    # Sınıf dengesizliği ``scale_pos_weight`` ile eğitim anında ayarlanır
    # (bkz. model.positive_class_weight). ``class_weight="balanced"`` yerine
    # bunu kullanıyoruz çünkü aynı değer hem sklearn hem de native lgb.cv
    # yoluna birebir aktarılabiliyor.
}

#: Açıklamada kullanıcıya gösterilecek en fazla sürücü sayısı.
TOP_K_DRIVERS = 5


# --------------------------------------------------------------------------
# LLM ayarları (.env dosyasından okunur)
# --------------------------------------------------------------------------
class LLMSettings(BaseSettings):
    """Ajan katmanının LLM yapılandırması.

    Üç arka uç desteklenir ve hepsi OpenAI-uyumlu ``/v1`` arayüzünü konuşur,
    bu yüzden kod değişmeden geçiş yapılabilir:

    * ``ollama``  -> http://localhost:11434/v1   (yerel, varsayılan)
    * ``foundry`` -> Azure AI Foundry Local      (yerel, Microsoft yığını)
    * ``azure``   -> Azure OpenAI                (bulut)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="XAI_",
        extra="ignore",
        protected_namespaces=(),
    )

    llm_backend: str = "ollama"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-needed-for-local"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1400
    llm_timeout: float = 180.0

    #: Ajanın bir soruda yapabileceği en fazla tool çağrı turu.
    agent_max_tool_rounds: int = 6


def get_llm_settings() -> LLMSettings:
    """LLM ayarlarını ortamdan/.env'den okur."""
    return LLMSettings()
