"""Ajanın çağırabileceği tool'lar — LLM ile modelin arasındaki tek köprü.

Tool nedir, neden gerekli?
--------------------------
Bir LLM'e "bu başvuru neden reddedildi?" diye sorarsanız, elinde veri yoksa
uydurur — çünkü dil modelinin işi olası bir sonraki kelimeyi üretmek, doğruyu
söylemek değil. Tool-calling bu boşluğu kapatır: modele "cevabı bilmiyorsan
şu fonksiyonu çağır" deriz, model fonksiyon çağrısı üretir, biz **gerçek
kodu** koşarız ve sonucu modele geri veririz.

Bu projede dört tool var ve hepsi doğrudan LightGBM + SHAP'a bağlı:

============================  ===========================================
``get_decision_explanation``  Mevcut başvurunun SHAP açıklaması
``run_what_if``               Bir özelliği değiştirip modeli YENİDEN koşar
``get_feature_info``          Bir özelliğin tanımı ve geçerli değerleri
``get_global_importance``     Model düzeyinde özellik önemi
============================  ===========================================

Kritik nokta: ``run_what_if`` gerçekten modeli yeniden çalıştırır. Yani
"gelirim daha yüksek olsaydı?" sorusunun cevabı LLM'in tahmini değil,
LightGBM'in yeni tahminidir. Ajanın hiçbir noktada tahmin yürütme yetkisi yok.

Tool imzalarındaki ``Annotated[..., Field(description=...)]`` yazımı Agent
Framework tarafından okunup LLM'e gönderilen JSON şemasına çevrilir. Bu
açıklamalar modelin tool'u doğru çağırması için kritik — belirsiz bir
açıklama, yanlış parametreyle çağrı demek.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field

from .explainer import CreditExplainer
from .features import FEATURE_META, display_name, model_features
from .schemas import DecisionExplanation


def _coerce_value(feature: str, raw: str) -> Any:
    """LLM'den gelen metin değeri özelliğin gerçek tipine çevirir.

    LLM her zaman string gönderir. Numerik bir özellik için "24" gelirse
    bunu ``24`` sayısına çevirmemiz gerekir, yoksa LightGBM patlar.
    Kategorik özelliklerde ise değerin izin verilen seviyeler arasında olup
    olmadığını burada kontrol ediyoruz — hatalı çağrıyı modele anlamlı bir
    mesajla geri bildirmek, sessizce yanlış sonuç üretmekten iyidir.
    """
    meta = FEATURE_META.get(feature)
    if meta is None:
        raise ValueError(f"'{feature}' bilinen bir özellik değil.")

    text = str(raw).strip()

    if meta.kind == "numeric":
        cleaned = text.replace(",", ".").replace("%", "").strip()
        # "24 ay", "3000 DM" gibi birimli girdileri temizle
        for token in cleaned.split():
            try:
                num = float(token)
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"'{display_name(feature)}' sayısal bir özellik; '{raw}' "
                "sayıya çevrilemedi."
            )
        if meta.bounds:
            lo, hi, _ = meta.bounds
            if not (lo <= num <= hi):
                raise ValueError(
                    f"'{display_name(feature)}' için {num} değeri veri setindeki "
                    f"aralığın dışında ({lo}–{hi}). Bu aralık dışında model "
                    "güvenilir tahmin üretmez."
                )
        return int(num) if float(num).is_integer() else num

    # Kategorik: tam eşleşme, yoksa etiket üzerinden eşleşme dene
    if text in meta.levels:
        return text
    for level, label in meta.value_labels.items():
        if text.lower() == label.lower():
            return level
    raise ValueError(
        f"'{display_name(feature)}' için '{raw}' geçerli bir değer değil. "
        f"İzin verilen değerler: {list(meta.levels)}"
    )


@dataclass
class AgentToolbox:
    """Ajanın tool'larını mevcut başvuruya bağlayan bağlam nesnesi.

    Tool fonksiyonlarının "hangi başvuru hakkında konuşuyoruz?" bilgisine
    ihtiyacı var ama LLM'e başvuru kimliği taşıttırmak hataya açık. Bunun
    yerine tool'ları bir closure içinde bu nesneye bağlıyoruz: ajan
    parametresiz ``get_decision_explanation()`` çağırıyor, tool hangi
    başvuruyla ilgilendiğini kendisi biliyor.
    """

    explainer: CreditExplainer
    applicant: dict[str, Any]
    applicant_id: str = "başvuru"
    #: Tool çağrılarının denetim kaydı — faithfulness testi bunu okur.
    call_log: list[dict[str, Any]] = field(default_factory=list)
    _explanation: DecisionExplanation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        feats = self.explainer.bundle.feature_names
        missing = [f for f in feats if f not in self.applicant]
        if missing:
            raise ValueError(f"Başvuruda eksik özellik(ler): {missing}")

    # ------------------------------------------------------------------
    @property
    def explanation(self) -> DecisionExplanation:
        """Açıklamayı bir kez hesaplar, sonra önbellekten verir."""
        if self._explanation is None:
            self._explanation = self.explainer.explain_row(
                self.applicant, applicant_id=self.applicant_id
            )
        return self._explanation

    def reset(self, applicant: dict[str, Any], applicant_id: str = "başvuru") -> None:
        """Toolbox'ı yeni bir başvuruya bağlar."""
        self.applicant = dict(applicant)
        self.applicant_id = applicant_id
        self._explanation = None
        self.call_log.clear()

    def _log(self, name: str, args: dict[str, Any], ok: bool, note: str = "") -> None:
        self.call_log.append(
            {"tool": name, "args": args, "ok": ok, "note": note}
        )

    # ------------------------------------------------------------------
    def build_tools(self) -> list[Callable[..., Any]]:
        """Agent Framework'e verilecek tool fonksiyonlarını üretir."""
        box = self

        def get_decision_explanation() -> dict[str, Any]:
            """Bu başvuru için modelin kararını ve SHAP gerekçesini döndürür.

            Kredi kararının NEDEN böyle olduğunu açıklaman gerektiğinde bu
            tool'u çağır. Riski artıran ve azaltan etkenleri, her birinin
            etki payını, karar eşiğini ve kararın ne kadar net olduğunu verir.
            Kullanıcı karar hakkında herhangi bir soru sorduğunda ÖNCE bunu çağır.
            """
            box._log("get_decision_explanation", {}, True)
            return box.explanation.to_agent_payload()

        def run_what_if(
            feature: Annotated[
                str,
                Field(
                    description=(
                        "Değiştirilecek özelliğin kodu. Geçerli kodlar: "
                        + ", ".join(model_features())
                    )
                ),
            ],
            new_value: Annotated[
                str,
                Field(
                    description=(
                        "Özelliğin yeni değeri. Sayısal özellikler için sayı "
                        "(örn. '24'), kategorik özellikler için izin verilen "
                        "değerlerden biri (örn. '>=200'). Geçerli değerleri "
                        "bilmiyorsan önce get_feature_info çağır."
                    )
                ),
            ],
        ) -> dict[str, Any]:
            """Bir özelliği değiştirir ve modeli GERÇEKTEN yeniden koşar.

            "Ne olurdu?", "olsaydı?", "değişse?" tipi her soruda bu tool'u
            çağırmak ZORUNLUDUR. Kendi tahminini yürütme — bu tool modelin
            yeni risk oranını, yeni kararı ve kararın değişip değişmediğini
            gerçek hesaplamayla döndürür.
            """
            args = {"feature": feature, "new_value": new_value}
            try:
                if feature not in box.explainer.bundle.feature_names:
                    raise ValueError(
                        f"'{feature}' bu modelde kullanılmıyor. Geçerli özellikler: "
                        f"{list(box.explainer.bundle.feature_names)}"
                    )
                value = _coerce_value(feature, new_value)
                result = box.explainer.what_if(box.applicant, {feature: value})
                box._log("run_what_if", args, True)
                return result.to_agent_payload()
            except ValueError as exc:
                box._log("run_what_if", args, False, str(exc))
                return {
                    "hata": str(exc),
                    "yapilacak": (
                        "Kullanıcıya bu senaryonun hesaplanamadığını söyle ve "
                        "nedenini açıkla. Sayı UYDURMA."
                    ),
                }

        def get_feature_info(
            feature: Annotated[
                str,
                Field(
                    description=(
                        "Tanımı istenen özelliğin kodu. Geçerli kodlar: "
                        + ", ".join(model_features())
                    )
                ),
            ],
        ) -> dict[str, Any]:
            """Bir özelliğin ne anlama geldiğini ve geçerli değerlerini döndürür.

            Kullanıcı "bu ne demek?" diye sorduğunda veya what-if çağırmadan
            önce hangi değerlerin geçerli olduğunu öğrenmek istediğinde kullan.
            """
            args = {"feature": feature}
            meta = FEATURE_META.get(feature)
            if meta is None:
                box._log("get_feature_info", args, False, "bilinmeyen özellik")
                return {
                    "hata": f"'{feature}' bilinen bir özellik değil.",
                    "gecerli_ozellikler": list(model_features()),
                }
            box._log("get_feature_info", args, True)
            info: dict[str, Any] = {
                "ad": meta.display,
                "aciklama": meta.description,
                "tip": "sayısal" if meta.kind == "numeric" else "kategorik",
                "basvuru_sahibi_degistirebilir_mi": (
                    "evet" if meta.actionable else "hayır"
                ),
            }
            if meta.kind == "categorical":
                info["gecerli_degerler"] = [
                    {"kod": lvl, "anlami": meta.value_labels.get(lvl, lvl)}
                    for lvl in meta.levels
                ]
            elif meta.bounds:
                lo, hi, _ = meta.bounds
                info["gecerli_aralik"] = f"{lo:g} – {hi:g}"
                if meta.unit:
                    info["birim"] = meta.unit
            contribution = box.explanation.get(feature)
            if contribution:
                info["bu_basvurudaki_degeri"] = contribution.value_label
            return info

        def get_global_importance(
            top_n: Annotated[
                int,
                Field(
                    description="Kaç özellik döndürülsün (1-18 arası)", ge=1, le=18
                ),
            ] = 8,
        ) -> dict[str, Any]:
            """Modelin GENEL olarak hangi özelliklere ağırlık verdiğini döndürür.

            "Model genelde neye bakıyor?" tipi sorularda kullan. DİKKAT: bu
            genel eğilimdir, bu başvuruya özgü DEĞİLDİR. Bu başvuruya özgü
            gerekçe için get_decision_explanation kullan.
            """
            args = {"top_n": top_n}
            imp = box.explainer.bundle.metrics.get("global_importance")
            if imp is None:
                imp = _load_cached_importance()
            if imp is None:
                box._log("get_global_importance", args, False, "önbellek yok")
                return {
                    "hata": "Küresel önem verisi bulunamadı.",
                    "yapilacak": "Kullanıcıya bu bilginin elinde olmadığını söyle.",
                }
            box._log("get_global_importance", args, True)
            items = imp["features"][: max(1, min(int(top_n), 18))]
            return {
                "aciklama": (
                    "Modelin genel eğilimi: hangi özellikler kararları en çok "
                    "belirliyor. Bu tek bir başvurunun gerekçesi DEĞİLDİR."
                ),
                "ozellikler": [
                    {
                        "sira": it["rank"],
                        "ad": it["display_name"],
                        "genel_agirligi": f"%{it['percent']:.1f}",
                        "egilim": it["direction_bias"],
                    }
                    for it in items
                ],
            }

        return [
            get_decision_explanation,
            run_what_if,
            get_feature_info,
            get_global_importance,
        ]


def what_ifs_from_calls(
    explainer: CreditExplainer,
    applicant: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> list[Any]:
    """Çağrı kaydından ajanın GERÇEKTEN koştuğu what-if senaryolarını yeniden kurar.

    Sadakat denetimi, ajanın yanıtındaki sayıların meşru olup olmadığına bakar.
    Ajan bir what-if senaryosu koşturduysa oradan gelen risk oranları da
    meşrudur — ama denetçi bunu bilmek için senaryonun ne olduğunu bilmeli.
    Tahmini bir senaryo uydurup denetime vermek yanlış alarm üretiyordu.
    """
    results: list[Any] = []
    for call in tool_calls:
        if call.get("tool") != "run_what_if" or not call.get("ok"):
            continue
        try:
            feature = call["args"]["feature"]
            value = _coerce_value(feature, call["args"]["new_value"])
            results.append(explainer.what_if(applicant, {feature: value}))
        except (KeyError, ValueError):
            continue
    return results


def _load_cached_importance() -> dict[str, Any] | None:
    """``artifacts/global_importance.json`` dosyasını okur (varsa)."""
    import json

    from . import config

    if not config.FEATURE_IMPORTANCE_PATH.exists():
        return None
    try:
        return json.loads(config.FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
