"""Veri katmanı: German Credit (Statlog) verisini indirir, doğrular, böler.

Tasarım kararları
-----------------
1. **Önbellekleme.** Veri bir kez indirilir ve ``data/raw/`` içine CSV olarak
   yazılır. Sonraki çalıştırmalar ağ erişimi gerektirmez — jüri önünde
   internet olmasa da proje çalışır.
2. **Sabit kategori seviyeleri.** ``features.CATEGORICAL_LEVELS`` sözlüğündeki
   sıra dayatılır. LightGBM kategorileri koda çevirerek öğrendiği için bu sıra
   eğitim ve çıkarım arasında birebir aynı olmak zorundadır.
3. **Korunan özelliklerin ayrılması.** Cinsiyet/uyruk gibi özellikler model
   girdisinden çıkarılır ama silinmez; adalet denetimi için ayrı bir çerçevede
   saklanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .features import (
    ALL_FEATURES,
    CATEGORICAL_LEVELS,
    NUMERIC_FEATURES,
    model_features,
)

OPENML_NAME = "credit-g"
OPENML_VERSION = 1


# --------------------------------------------------------------------------
# İndirme / yükleme
# --------------------------------------------------------------------------
def download_raw(force: bool = False) -> pd.DataFrame:
    """Veriyi OpenML'den indirir ve ``data/raw/german_credit.csv`` olarak kaydeder.

    Args:
        force: ``True`` ise önbellek yok sayılıp yeniden indirilir.

    Returns:
        Hedef sütunu dâhil ham veri çerçevesi.
    """
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    from sklearn.datasets import fetch_openml  # ağ gerektirir, tembel içe aktarım

    bunch = fetch_openml(
        OPENML_NAME, version=OPENML_VERSION, as_frame=True, parser="pandas"
    )
    df = bunch.data.copy()
    df[config.TARGET_COL] = bunch.target.astype(str)
    df.to_csv(config.RAW_CSV, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """Ham veriyi önbellekten (yoksa indirerek) yükler ve tiplerini düzeltir."""
    df = download_raw()
    return _coerce_dtypes(df)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun tiplerini özellik sözlüğüne göre sabitler.

    Kategorik sütunlar ``features.CATEGORICAL_LEVELS`` sırasıyla
    ``pd.CategoricalDtype``'a çevrilir. Sözlükte tanımlı olmayan bir değer
    bulunursa sessizce ``NaN``'a düşmesin diye açıkça hata veririz.
    """
    df = df.copy()

    for col, levels in CATEGORICAL_LEVELS.items():
        if col not in df.columns:
            continue
        raw = df[col].astype(str)
        unknown = sorted(set(raw.unique()) - set(levels))
        if unknown:
            raise ValueError(
                f"'{col}' sütununda sözlükte tanımlı olmayan değer(ler) var: "
                f"{unknown}. features.py içindeki 'levels' listesini güncelleyin."
            )
        df[col] = raw.astype(pd.CategoricalDtype(categories=list(levels)))

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

    if config.TARGET_COL in df.columns:
        df[config.TARGET_COL] = df[config.TARGET_COL].astype(str)

    return df


def encode_target(y: pd.Series) -> pd.Series:
    """Hedefi 0/1'e çevirir: ``1 = bad`` (riskli), ``0 = good``.

    Pozitif sınıfın "riskli" olması SHAP işaretlerinin yönünü belirler:
    **pozitif SHAP değeri riski artırır.** Bu sözleşme tüm projede geçerlidir.
    """
    return (y.astype(str) == config.POSITIVE_LABEL).astype(int)


# --------------------------------------------------------------------------
# Bölme
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    """Eğitim/test bölünmesi ve adalet denetimi için korunan özellikler."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    #: Korunan özellikler (modelde YOK) — yalnızca adalet denetimi için.
    protected_train: pd.DataFrame
    protected_test: pd.DataFrame
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def summary(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": len(self.feature_names),
            "n_categorical": len(self.categorical_features),
            "train_positive_rate": round(float(self.y_train.mean()), 4),
            "test_positive_rate": round(float(self.y_test.mean()), 4),
            "dropped_protected": [
                c for c in ALL_FEATURES if c not in self.feature_names
            ],
        }


def prepare(
    drop_protected: bool | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> Dataset:
    """Ham veriyi yükleyip katmanlı (stratified) eğitim/test bölünmesi yapar.

    Katmanlı bölme şart: hedef %70/%30 dengesiz olduğu için rastgele bölme
    test setinde sınıf oranını kaydırabilir ve metrikler yanıltıcı olur.
    """
    drop_protected = (
        config.DROP_PROTECTED if drop_protected is None else drop_protected
    )
    test_size = config.TEST_SIZE if test_size is None else test_size
    seed = config.RANDOM_SEED if seed is None else seed

    df = load_raw()
    y = encode_target(df[config.TARGET_COL])

    feats = model_features(drop_protected=drop_protected)
    X = df[list(feats)]
    protected = df[[c for c in ALL_FEATURES if c not in feats]]

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected, test_size=test_size, random_state=seed, stratify=y
    )

    cat_feats = tuple(c for c in feats if c in CATEGORICAL_LEVELS)

    ds = Dataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        protected_train=prot_train.reset_index(drop=True),
        protected_test=prot_test.reset_index(drop=True),
        feature_names=feats,
        categorical_features=cat_feats,
    )

    config.ensure_dirs()
    ds.X_test.assign(**{config.TARGET_COL: ds.y_test}).to_csv(
        config.PROCESSED_DIR / "test.csv", index=False
    )
    ds.X_train.assign(**{config.TARGET_COL: ds.y_train}).to_csv(
        config.PROCESSED_DIR / "train.csv", index=False
    )
    return ds


def align_frame(row: dict[str, object], feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Tek bir başvuru sözlüğünü modelin beklediği tek satırlık çerçeveye çevirir.

    Bu fonksiyon what-if senaryolarının ve Streamlit manuel girişinin kalbi:
    kullanıcıdan gelen ham sözlüğü, kategori seviyeleri ve sütun sırası
    modelle **birebir aynı** olan bir ``DataFrame``'e dönüştürür.
    """
    missing = [f for f in feature_names if f not in row]
    if missing:
        raise ValueError(f"Eksik özellik(ler): {missing}")

    data = {f: [row[f]] for f in feature_names}
    frame = pd.DataFrame(data, columns=list(feature_names))

    for col, levels in CATEGORICAL_LEVELS.items():
        if col not in frame.columns:
            continue
        value = str(frame.at[0, col])
        if value not in levels:
            raise ValueError(
                f"'{col}' için geçersiz değer: {value!r}. "
                f"İzin verilen değerler: {list(levels)}"
            )
        frame[col] = pd.Series(
            [value], dtype=pd.CategoricalDtype(categories=list(levels))
        )

    for col in NUMERIC_FEATURES:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")

    return frame
