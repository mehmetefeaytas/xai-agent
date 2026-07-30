# XAI-Agent — Şeffaf Kredi Risk Ajanı

**Mehmet Efe Aytaş** · Microsoft staj / bitirme projesi

Bir kredi başvurusunu **reddeden** makine öğrenmesi modelinin kararını, SHAP ile
matematiksel olarak parçalayıp bir LLM ajanı aracılığıyla başvuru sahibine
**doğal Türkçe** ile açıklayan uçtan uca sistem.

Projenin ayırt edici iddiası şu: **ajanın söylediği her cümle programatik olarak
denetleniyor.** Şeffaflık burada bir slogan değil, ölçülen bir metrik.

![XAI-Agent canlı gösterim](docs/images/tanitim.gif)

> 🎬 **Sesli anlatımlı videolar** (Türkçe, altyazılı — indirip oynatın):
> [**90 saniyelik tanıtım**](docs/video/xai-agent-tanitim-90sn.mp4) ·
> [**13 dakikalık derin anlatım**](docs/video/xai-agent-derin-anlatim.mp4)
> ([`.srt`](docs/video/) altyazılar yanındadır)
>
> Videolardaki terminal çıktıları ve arayüz görüntüleri **gerçek çalıştırmalardan**
> kaydedildi; üretim hattı `scripts/video/` altında ve yeniden çalıştırılabilir.

## Mimari — üç katman ve üstünde bir denetçi

```mermaid
flowchart LR
    A["Başvuru<br/>18 özellik"] --> B

    subgraph L1["KATMAN 1 · karar"]
        B["LightGBM<br/>243 ağaç"] --> C{"risk ≥ 0.26 ?"}
    end

    C --> D

    subgraph L2["KATMAN 2 · gerekçe"]
        D["TreeSHAP<br/>18 katkı"] --> E["Toplanabilirlik kontrolü<br/>taban + Σkatkı = çıktı<br/>hata 2.7e-15"]
        E --> F["JSON yükü<br/>SADECE hazır dizeler<br/>ham log-odds YOK"]
    end

    F --> G

    subgraph L3["KATMAN 3 · dil"]
        G["LLM ajanı<br/>4 tool"] --> H["Türkçe anlatı"]
    end

    H --> I

    subgraph L4["KATMAN 4 · denetim"]
        I["Sadakat denetçisi<br/>7 ihlal türü"] --> J{"ihlal var ?"}
        J -->|hayır| M["Kullanıcıya göster<br/>+ denetim sonucu"]
        J -->|evet| K["Onarım mesajı:<br/>ihlalleri gerekçesiyle bildir"]
    end

    K -.->|yeniden yazdır| G

    style L1 fill:#eef4fb,stroke:#1f4e79
    style L2 fill:#eef8ee,stroke:#2ca02c
    style L3 fill:#fdf6e3,stroke:#e3b341
    style L4 fill:#fbeaea,stroke:#d62728
```

**Kritik sorumluluk ayrımı:** LLM tahmin yapmaz, karar vermez, hesaplamaz.
Elinde yalnızca modele bağlı dört tool var. "Vade 12 aya düşse ne olurdu?"
sorusunun cevabı LLM'in tahmini değil, **LightGBM'in yeniden koşumudur**.

### Bir bakışta

| Faz | Ne yapıldı | Ölçülen sonuç |
|---|---|---|
| **1** | LightGBM + baseline karşılaştırması | maliyet **104** (LogReg 115, dummy 300), recall **0.917**, CV AUC 0.793 ± 0.039 |
| **2** | TreeSHAP + SHAP→JSON köprüsü | toplanabilirlik hatası **2.7e-15**, SHAP rastgeleyi **2.16×** yeniyor |
| **3** | Agent Framework + 4 tool + sistem promptu | tool çağrılan yanıtlarda 11–30 sayı doğru temellendirildi |
| **4** | Streamlit arayüzü (5 sekme) | ajan anlatısı SHAP grafiğinin yanında, denetim sonucu ekranda |
| **5** | Sadakat + adalet denetimi + onarım döngüsü | ham sadakat %33–47 (koşuya göre); onarım **çerçeveleme ihlallerini 25→0** yaptı; korunan özellik ihlali **0** |

Test kapsamı: **142** LLM'siz test + **6** canlı ajan testi = **148** test
(biri bilinçli `xfail`, aşağıda açıklanıyor), `ruff` temiz.
Her ölçüm `artifacts/` altındaki JSON raporlarından yeniden üretilebilir.

### İçindekiler

| | |
|---|---|
| [Hızlı başlangıç](#hızlı-başlangıç) | Kurulum ve ilk çalıştırma |
| [Veri ve hedef tanımı](#veri-ve-hedef-tanımı) | German Credit, maliyet matrisi, korunan özellikler |
| [Katman 1 — LightGBM](#katman-1--lightgbm-kararı-verir) | Model, çapraz doğrulama, maliyet eşiği |
| [Katman 2 — SHAP](#katman-2--shap-gerekçeyi-hesaplar) | Şelale, toplanabilirlik kanıtı, küresel önem |
| [SHAP → ajan köprüsü](#shap--ajan-köprüsü-neden-sadece-dize) | Yükte neden hiç ham sayı yok |
| [Katman 3 — Ajan](#katman-3--ajan-ve-dört-tool) | Agent Framework, tool'lar, uçtan uca akış |
| [Katman 4 — Sadakat denetçisi](#katman-4--sadakat-denetçisi) | 7 ihlal türü, ERASER ölçümü |
| [Onarım döngüsü](#onarım-döngüsü-denetçiyi-güvenlik-mekanizmasına-çevirmek) | Critic-and-revise, ölçülen etki |
| [Adalet denetimi](#adalet-denetimi) | Vekil ayrımcılık, imkânsızlık teoremi |
| [Arayüz turu](#arayüz-turu) | Beş sekme, ekran görüntüleriyle |
| [Kalite güvencesi](#kalite-güvencesi--148-test) | Testler ve neyin bozulmasını engelliyorlar |
| [Bulunan gerçek hatalar](#geliştirme-sırasında-bulunan-ve-düzeltilen-gerçek-hatalar) | 12 hata, teşhis ve çözümüyle |
| [Mimari kararlar](#mimari-kararlar-ve-gerekçeleri) | Neden LightGBM, neden bu köprü |
| [Bilinen sınırlar](#bilinen-sınırlar) | Dürüst bilanço |

---

## Hızlı başlangıç

```bash
# 1. Bağımlılıklar (Python 3.12 + uv gerekir)
brew install libomp          # macOS: LightGBM'in OpenMP bağımlılığı
uv sync

# 2. Modeli eğit (~40 sn) — veriyi indirir, eğitir, SHAP önemini üretir
uv run python scripts/train.py

# 3. Yerel LLM (ajan katmanı için)
brew install ollama && ollama serve
ollama pull qwen2.5:7b-instruct

# 4. Arayüzü aç
uv run streamlit run app/streamlit_app.py
```

Terminalden denemek için:

```bash
uv run python scripts/explain_demo.py                # en riskli başvuru
uv run python scripts/explain_demo.py --borderline   # kıl payı reddedilen
uv run python scripts/explain_demo.py --no-llm       # yalnızca SHAP katmanı
uv run python scripts/evaluate.py                    # sadakat + adalet denetimi
uv run python scripts/evaluate.py --repair 1          # onarım döngüsüyle karşılaştır
uv run python scripts/evaluate.py --skip-llm          # LLM'siz, hızlı, çevrimdışı
uv run pytest -q -m "not llm"                        # 142 test, LLM gerekmez
uv run pytest -q                                     # canlı ajan testleri dahil
```

### Üç katmanın çıktısı, tek ekranda

Aşağısı **gerçek bir çalıştırmanın** çıktısı (`--borderline`, yani karar eşiğine
en yakın başvuru — A-006):

```
KATMAN 1 — LightGBM tahmini  (A-006)
  Karar          : ONAYLANDI
  Risk oranı     : 25.92%
  Karar eşiği    : 26.00%
  Eşiğe uzaklık  : -0.08% (sinirda)      ← kıl payı onaylandı
  Gerçek etiket  : iyi (good)

KATMAN 2 — SHAP gerekçesi
  Toplanabilirlik: taban -0.4365 + katkılar -0.6135 = -1.0500
                   model çıktısı  = -1.0500  (hata 4.4e-16) -> GEÇTİ

  Riski ARTIRAN etkenler (toplam etkinin %39'i):
    ▲ Vadesiz hesap durumu     bakiye 0–200 DM arası   % 16.4  (+0.4750)
    ▲ Kredi vadesi             18 ay                   %  8.9  (+0.2575)
    ▲ Tasarruf hesabı durumu   100 DM altı birikim     %  8.9  (+0.2574)
  Riski AZALTAN etkenler:
    ▼ Kredi geçmişi            kritik hesap ...        % 21.2  (-0.6111)
    ▼ Kredi tutarı             3612 DM                 % 16.3  (-0.4717)
    ▼ Yaş                      37 yaş                  %  6.7  (-0.1947)

  Dışlanan korunan özellikler: ['personal_status', 'foreign_worker']

KATMAN 3 — LLM ajanı  (denetimli mod)
  [tool çağrıları: get_decision_explanation]
  [onarım devreye girdi: 3 ihlal -> 0]

  "Başvuru onaylandı. Model bu başvurunun risk oranını %25.9 olarak hesapladı;
   bankanın karar eşiği %26. Vadesiz hesap durumu: bakiye 0–200 DM arası —
   kararın %16.4'ünü oluşturuyor (güçlü etki)..."

  ✅ SADAKAT DENETİMİ GEÇTİ (11 sayı temellendirildi)
```

---

## Veri ve hedef tanımı

| | |
|---|---|
| **Kaynak** | [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| **Boyut** | 1000 başvuru · 20 özellik (11 kategorik, 9 sayısal) |
| **Bölme** | 800 eğitim / 200 test, katmanlı (stratified), `seed=42` |
| **Hedef** | `class` → **1 = riskli (bad)**, 0 = iyi (good) · pozitif oran %30 |
| **Maliyet matrisi** | Veri setiyle **birlikte gelir**: yanlış kabul (FN) = 5, yanlış ret (FP) = 1 |
| **Para birimi** | Deutsche Mark (veri seti 1994 tarihli) |

**Korunan özellikler modelden çıkarıldı.** `personal_status` (medeni durum +
cinsiyet) ve `foreign_worker` (uyruk) model girdisinden düşürüldü, ama
**adalet denetimi için saklandı** — çıkarmak yetmiyorsa bunu ölçebilmek
gerekiyor (`Dataset.protected_train` / `protected_test`).

`age` bilinçli olarak **modelde bırakıldı**. Bu bir ihmal değil, ölçülebilir
bir seçim: aşağıdaki adalet bölümünde bunun bedeli 0.298'lik bir ret oranı
farkı olarak görünüyor.

**Kategorik seviyeler sabitlendi.** `features.py` her kategorik özelliğin
geçerli değerlerini ve sırasını açıkça listeler; veride tanınmayan bir değer
görülürse `_coerce_dtypes()` **hata verir**. Sebebi somut: pandas `category`
tipinde seviye sırası değişirse LightGBM sessizce farklı tahminler üretir —
hata vermeyen, fark edilmesi çok zor bir bozulma.

---

## Katman 1 — LightGBM kararı verir

Kategorik değişkenler **one-hot açılmadı**; LightGBM'in kendi `category`
desteği kullanıldı. Bu bir hız tercihi değil, açıklanabilirlik tercihi:
one-hot açarsanız tek bir özelliğin SHAP değeri onlarca sütuna dağılır ve
"vadesiz hesap durumu kararı %16.4 etkiledi" gibi tek bir cümle kuramazsınız.

Model **243 ağaçta** durdu. Bu sayı tek bir doğrulama kümesinde değil,
`lgb.cv` ile **5 katın ortalamasında** belirlendi:

![5 katlı çapraz doğrulama](docs/images/10-cv-katlamalari.png)

Katlar arası fark **0.106** — 1000 satırlık bir veri setinde tek bir hold-out
skoruna güvenilemeyeceğinin doğrudan kanıtı. Test AUC'si (0.7942) çapraz
doğrulama ortalamasının (0.7934) bir standart sapma bandı içinde kalıyor.

### Model performansı (test seti, n=200)

| Metrik | LightGBM | Lojistik Reg. | Dummy | Not |
|---|---:|---:|---:|---|
| **Beklenen maliyet** | **104** | 115 | 300 | **asıl karar metriği** (düşük = iyi) |
| Recall (riskli sınıf) | **0.917** | 0.883 | 0.000 | batan krediyi yakalama |
| ROC-AUC | 0.794 | **0.810** | 0.500 | sıralama kalitesi |
| PR-AUC | 0.647 | **0.655** | 0.300 | azınlık sınıfı |
| KS | 0.505 | **0.564** | 0.000 | kredi skorlama standardı |
| F1 (riskli) | **0.567** | 0.549 | 0.000 | |
| Brier | 0.184 | **0.183** | 0.210 | olasılık kalibrasyonu |
| Accuracy | 0.580 | 0.565 | **0.700** | ⚠️ yanıltıcı, aşağıya bakın |

5-kat CV AUC: **0.793 ± 0.039** · 243 ağaç (CV'li erken durdurma) · karar eşiği **0.26**

**Dürüst not:** Lojistik regresyon AUC'de LightGBM'i biraz geçiyor. Bu, 1000
satırlık tablo verisinde beklenen bir sonuçtur ve gizlenmemesi gerekir — küçük
veride doğrusal modeller güçlü rakiptir. Ancak kararı belirleyen metrik AUC
değil **maliyet**: LightGBM 104'e karşı 115 ile öne geçiyor ve riskli
müşterileri %91.7 oranında yakalıyor (LogReg %88.3).

![ROC ve PR eğrileri](docs/images/02-roc-pr.png)

Grafikte iki eğri neredeyse üst üste. LightGBM'i "daha gelişmiş algoritma"
olduğu için savunmuyoruz; onu maliyet ve duyarlılık farkı seçtiriyor.

### Neden accuracy'ye bakmıyoruz?

Veri %70 "iyi" / %30 "riskli". Her şeye "iyi" diyen aptal bir model **%70
accuracy** alır — bizim modelimizden (%58) *daha yüksek*. Ama o aptal model 60
batan krediyi kaçırır ve 300 birim maliyet üretir; bizimki 104 üretir.
Accuracy bu problemde tek başına anlamsızdır.

### Karar eşiği neden 0.5 değil?

UCI German Credit'in resmî maliyet matrisi asimetriktir: **riskli bir müşteriyi
"iyi" sanmak, iyi bir müşteriyi "riskli" sanmaktan 5 kat pahalıdır.** Bu oranda
Bayes-optimal eşik teorik olarak

```
t* = c_FP / (c_FP + c_FN) = 1 / (1 + 5) = 0.167
```

Eğitim, eşiği **kat-dışı (out-of-fold) tahminler** üzerinde maliyet
minimizasyonuyla arayıp **0.26** buldu — teorik değere yakın, yani mekanizma
çalışıyor. Test seti eşik seçiminde hiç kullanılmadı (veri sızıntısı yok).

![Eşik / maliyet eğrisi](docs/images/01-esik-maliyet.png)

Mavi eğri eşiğin **seçildiği** yer (eğitim setinin kat-dışı tahminleri); gri
eğri yalnızca kararın test setinde de tuttuğunu **doğrulamak** için çizildi.
İki eğrinin minimumları birbirine yakın — eşik seçimi örnekleme aşırı
uymamış.

Bu eşiğin bedeli görünür: 140 iyi müşteriden 79'u gereksiz reddediliyor (FP),
ama 60 riskli müşteriden yalnızca 5'i kaçıyor (FN). Maliyet matrisi bunu kârlı
buluyor. Oran `config.COST_FALSE_NEGATIVE` ile değiştirilebilir.

![Karmaşıklık matrisleri: 0.26 ve 0.50](docs/images/03-karmasiklik-matrisi.png)

Karşılaştırma — aynı model, saf 0.5 eşiği: accuracy %72 (daha *iyi* görünür)
ama recall %73.3 ve maliyet **120** (daha *kötü*). Kırmızı çerçeveli hücre
pahalı hatayı gösteriyor: **16 kaçan riskli müşteri yerine 5.**

---

## Katman 2 — SHAP gerekçeyi hesaplar

Tek bir başvurunun kararı, özellik başına bir katkıya ayrıştırılıyor. Aşağıdaki
şelale grafiği yukarıdaki terminal çıktısıyla **aynı başvuruya** ait:

![SHAP şelale grafiği — A-006](docs/images/05-shap-selale.png)

Dikkat çekici bir ayrıntı: kredi geçmişinde *"kritik hesap veya başka bankada
devam eden kredi var"* kaydı riski **azaltıyor** (−0.611). Sezgiye aykırı ama
veriye uygun — ödeme geçmişi kanıtlanmış müşteri daha güvenli. Model bunu
veriden öğrendi; hiçbir yere kural olarak yazılmadı.

### Toplanabilirlik: açıklamanın kendi kanıtı

![Toplanabilirlik kanıtı](docs/images/06-toplanabilirlik.png)

SHAP'ın *yerel isabet* (local accuracy) aksiyomu şunu şart koşar:
`taban değer + Σ katkılar = modelin ham çıktısı`. Bunu **200 test başvurusunun
tamamında** ölçtük; en büyük sapma **2.7e-15**. Bu bir yaklaşım hatası değil,
kayan nokta aritmetiğinin sınırı.

Pratik anlamı: açıklama, modelin yanında duran ikinci bir tahmin **değil** —
modelin kendi kararının ayrıştırılmış hâli. Bu kontrol her açıklamanın içine
gömülüdür (`DecisionExplanation.additivity`) ve arayüzde de gösterilir.

İki bağımsız doğrulama yapıldı: SHAP değerleri LightGBM'in kendi
`pred_contrib=True` uygulamasından alınıyor ve `shap.TreeExplainer` ile
karşılaştırıldığında iki hesap arasındaki **fark tam olarak 0.0**
(`test_native_shap_matches_shap_package`).

### SHAP modele sadık mı? — ERASER ölçümü

Bir açıklama güzel görünüp yanlış olabilir. Bunu ölçmek için ERASER
literatüründen (DeYoung ve ark., 2020) iki metrik uyguluyoruz:

| k | Comprehensiveness ↑ | Sufficiency ↓ | Rastgele kontrol |
|---:|---:|---:|---:|
| 1 | 0.072 | 0.179 | 0.031 |
| 2 | 0.124 | 0.123 | 0.059 |
| 3 | 0.157 | 0.095 | 0.063 |
| 5 | **0.201** | **0.064** | 0.103 |

*(n=150 test başvurusu)*

* **Comprehensiveness:** SHAP'ın "en önemli" dediği k özelliği referans
  değerine çekip modeli yeniden koşuyoruz. Tahmin *çok* düşmeli.
  → k arttıkça monoton artıyor ✅
* **Sufficiency:** Yalnızca o k özelliği bırakıp gerisini referansa çekiyoruz.
  Tahmin orijinaline *yakın* kalmalı. → k arttıkça monoton düşüyor ✅
* **Rastgele kontrol:** aynı sayıda özelliği rastgele seçiyoruz.

![ERASER sadakat ölçümü](docs/images/07-sadakat-aopc.png)

**SHAP kazancı: 2.16× rastgele** (n=150) → *"GÜÇLÜ — SHAP sıralaması modelin
davranışını isabetle yakalıyor."* Bu kontrol grubu olmadan comprehensiveness
sayıları tek başına bir şey söylemez: 0.138'in iyi mi kötü mü olduğunu ancak
rastgele seçimin 0.064'ü karşısında bilebiliyoruz.

### Modelin genel davranışı

![Küresel SHAP önemi](docs/images/04-kuresel-onem.png)

Hiçbir özellik tek başına baskın değil — en güçlüsü %19.6. Bu bir sağlık
göstergesi: ilk eğitimde tek bir özellik toplam önemin **%50.8'ini** almıştı ve
o model yetersiz öğrenmişti (30 ağaç, AUC 0.776). Çapraz doğrulamalı erken
durdurmaya geçince önem dağıldı.

---

## SHAP → ajan köprüsü: neden sadece dize?

Bu, projenin en önemli tek tasarım kararı. `DecisionExplanation.to_agent_payload()`
ajana giden yükü üretir ve içinde **tek bir ham log-odds sayısı yok**:

```json
{
  "ozellik_kodu": "checking_status",
  "ad": "Vadesiz hesap durumu",
  "deger": "bakiye 0–200 DM arası",
  "etki_yonu": "riski artırıyor",
  "etki_payi": "%16.4",
  "etki_gucu": "güçlü",
  "basvuru_sahibi_degistirebilir_mi": "evet"
}
```

Sebebi acı bir tecrübe: ilk sürümde yük ham SHAP katkısını (`0.83`) içeriyordu
ve model bunu **"%83 risk"** diye okudu. Yorumlanabilecek hiçbir sayı
bırakmayınca bu hata sınıfı tamamen ortadan kalktı. Karar bir testle
kilitli: `test_agent_payload_has_no_raw_logodds`.

`etki_gucu` alanı da aynı mantıkla var: modelin "%16.4 güçlü mü zayıf mı?"
sorusunu kendi başına cevaplamasını istemiyoruz — eşikleri biz belirliyoruz
(`schemas.effect_strength`).

---

## Katman 3 — Ajan ve dört tool

**Microsoft Agent Framework 1.12.1** üzerinde, yerelde çalışan
`qwen2.5:7b-instruct` ile. Ajanın modele erişimi yalnızca dört tool üzerinden;
tool şemaları Python tip ipuçlarından ve docstring'lerden otomatik üretiliyor.

```mermaid
sequenceDiagram
    participant K as Kullanıcı
    participant A as LLM ajanı
    participant T as AgentToolbox
    participant M as LightGBM + SHAP
    participant D as Sadakat denetçisi

    K->>A: "Bu başvuru neden bu sonucu aldı?"
    A->>T: get_decision_explanation()
    T->>M: explain_row(başvuru)
    M-->>T: karar + 18 SHAP katkısı
    T-->>A: JSON yükü (yalnızca hazır dizeler)
    A-->>D: Türkçe anlatı
    D->>D: sayıları yükle karşılaştır<br/>yön / korunan özellik / dil kontrolü
    alt ihlal yok
        D-->>K: anlatı + "✅ N sayı temellendirildi"
    else ihlal var
        D->>A: onarım mesajı (ihlaller + gerekçe)
        A-->>D: yeniden yazılmış anlatı
        D-->>K: EN AZ ihlalli deneme + denetim raporu
    end
```

### Dört tool — LLM ile modelin arasındaki tek köprü

| Tool | Ne yapar |
|---|---|
| `get_decision_explanation` | Bu başvurunun SHAP açıklamasını döndürür |
| `run_what_if` | Bir özelliği değiştirip modeli **gerçekten** yeniden koşar |
| `get_feature_info` | Bir özelliğin tanımı ve geçerli değer aralığı |
| `get_global_importance` | Model düzeyinde özellik önemi |

---

## Katman 4 — Sadakat denetçisi

Ajanın ürettiği metin doğrudan kullanıcıya gitmiyor: önce bağımsız bir
program tarafından SHAP çıktısına karşı denetleniyor. Denetçi ajanın kendi
hakkındaki hiçbir iddiasına güvenmez — yalnızca metni ve yükü karşılaştırır.

### Denetlenen ihlal türleri

`faithfulness.audit_narrative()` her ajan yanıtını tarar:

| İhlal | Örnek | Nasıl yakalanır |
|---|---|---|
| **Temellenmemiş sayı** | "Risk oranınız %99.7" | Metindeki her sayı, yükteki izinli sayı kümesiyle karşılaştırılır |
| **Uydurulmuş kavram** | "Geliriniz yetersiz" | Model gelir bilgisi görmez; yasaklı kavram listesi |
| **Yön çelişkisi** | SHAP "azaltıyor" derken metin "artırıyor" | Cümle düzeyinde işaret karşılaştırması (yeniden yazılmış adlara toleranslı) |
| **Çerçeveleme hatası** | "Riski %13.7 oranında artırıyor" | Etki payı bir risk miktarı değildir |
| **Korunan özellik gerekçesi** | "Kadın olmanız nedeniyle..." | Korunan terim + dışlama ipucu yokluğu |
| **Dil kayması** | "Based on the provided tool response..." | İngilizce kelime sınırı taraması (≥4 farklı) |
| **Eksik tool çağrısı** | Varsayımsal soruya tool çağırmadan cevap | Soru kalıbı + çağrı kaydı |

Denetçinin kendisi de test edilir: `tests/test_faithfulness.py` her ihlal türü
için **kasıtlı olarak bozuk** bir anlatı üretip yakalandığını doğrular. Her şeye
"geçti" diyen bir denetçi işe yaramaz.

### Ölçülen sonuç: ana açıklamada 0 ihlal

`uv run python scripts/explain_demo.py --borderline` çıktısından:

```
  [tool çağrıları: get_decision_explanation]
  [onarım devreye girdi: 5 ihlal -> 0]

  Karar: Onaylandı · Risk Oranı: %25.9 · Karar Eşiği: %26.0
  Karar Netliği: KIL PAYI — küçük bir değişiklik sonucu değiştirebilir
  ...
  ✅ SADAKAT DENETİMİ GEÇTİ (32 sayı temellendirildi)
```

Ana açıklama sorusunda onarım döngüsü ihlalleri **tamamen sıfırlıyor** ve
metindeki 32 sayının hepsi SHAP çıktısıyla eşleşiyor. Takip sorularında ise
model hâlâ sayı uydurabiliyor (aynı koşuda 2–3 ihlal) — sistem bunu
**gizlemiyor, ekranda gösteriyor.**

### Denetçiyi kalibre etmek: yanlış pozitif avı

Bir denetçi iki yönde de yanlış olabilir. İlk sürüm 5 başvuruluk bir koşuda
**25 yön çelişkisi** bildirdi; metinlere tek tek bakıldığında neredeyse hepsi
denetçinin kendi hatasıydı. Üç kaynak bulundu ve düzeltildi:

| Yanlış pozitif | Sebep | Düzeltme |
|---|---|---|
| `employment` ↔ `residence_since` karışması | "Mevcut **işte** çalışma **süresi**" ile "Mevcut **adreste** ikamet **süresi**" ortak *genel* kelimeler üzerinden eşleşiyordu | Yalnızca **ayırt edici** kelimeler sayılıyor (`_GENERIC_NAME_WORDS`), ve cümle başına yalnızca **en iyi** eşleşen özellik değerlendiriliyor |
| "Vadeyi düşürerek riski azaltabilirsiniz" | Tavsiye cümlesi, mevcut SHAP yönü hakkında bir iddia sanıldı | `_CONDITIONAL_CUES` — varsayımsal/tavsiye cümleleri yön ve çerçeveleme denetiminden hariç |
| "gelir uydurdu" alarmı | `installment_commitment` özelliğinin Türkçe adı **"Taksit yükü (gelire oran kademesi)"** ve içinde "gelir" geçiyor | `mask_known_vocabulary()` — modelin kendi sözlüğü metinden silindikten *sonra* uydurma kavram aranıyor |
| "Başvuru kimliği A-046" → 46 temelsiz | Kimlik yükün içindeydi ama izinli sayı kümesine eklenmemişti | `allowed_numbers` kimliğin rakamlarını da içeriyor |
| "%21.3 riski artırıyor" kaçıyordu | Regex yalnızca "riski %21.3 ... artırıyor" *sırasını* tanıyordu; model en sık sayıyı **önce** yazıyor | `_MISFRAME_RE`'ye üçüncü alternatif eklendi |
| "hata 4.4e-16" → `4.4` ve `16` temelsiz | Bilimsel gösterim iki sayıya bölünüyordu; oysa yükte **aynen** yazılı | Genel kural: ajana gösterilen yükün içindeki **her** sayı meşru (özel durum listesi yerine) |
| "cinsiyetiniz etki etmiyor" ihlal sayıldı | Dışlama ipuçları yalnızca geçmiş zaman tanıyordu ("etki etmedi") | Geniş zaman ve edilgen olumsuzluk biçimleri eklendi |
| `**Impact Share:**` dil taramasından kaçtı | Boşlukla çevrili alt-dizi arıyordu; markdown yıldızı `impact`'i yapıştırıyor | Kelime sınırı regex'ine geçildi |

Bu düzeltmelerden sonra yön çelişkileri **25 → 2**'ye indi. Her düzeltme bir
regresyon testiyle sabitlendi — aksi hâlde denetçiyi gevşetmek onu işe yaramaz
hâle getirebilir. Bu ödünleşim testlerde açıkça belgelendi:
`test_direction_check_does_not_confuse_similar_names` sıkılığı korurken
`test_direction_check_tolerates_inflected_names` esnekliğin sınırını çiziyor.

**Kabul edilen sınır:** Ajan bir özelliğin tüm ayırt edici kelimelerini
atarsa ("Mevcut işte çalışma süresi" → "Mevcut İş Süresi") denetçi onu
kaçırır. Bunu yakalamak için gereken gevşeklik, yukarıdaki 25 yanlış pozitifi
geri getiriyor. Yanlış pozitifsiz bir denetçi, kaçırdığı birkaç vakadan daha
değerli: güvenilmeyen bir alarm hiç alarm olmamasından kötüdür.

---

### Ajan anlatısı SHAP'a sadık mı?

Bu, projenin en dürüst bölümü. Ölçüm, `qwen2.5:7b-instruct` ile 8 başvuru ×
3 soru = **24 yanıt** üzerinde yapıldı (`uv run python scripts/evaluate.py`).

| Ölçüm | Sonuç |
|---|---:|
| İhlalsiz yanıt | **8 / 24** |
| Ham sadakat skoru | **%33.3** |
| Toplam ihlal | 62 |

İhlal dağılımı:

| Tür | Sayı | Değerlendirme |
|---|---:|---|
| Çerçeveleme hatası (`misframed_shares`) | 25 | **gerçek** — etki payını risk oranı gibi sunuyor |
| Temellenmemiş sayı | 18 | **gerçek** — çoğu, tool çağırmadan uydurulan risk oranları |
| Yön çelişkisi | 14 | kısmen gerçek, kısmen koşul cümlesi kaynaklı |
| Eksik tool çağrısı | 2 | **gerçek** — varsayımsal soruya tool'suz cevap |
| Korunan özellik gerekçesi | **0** | ✅ hiç ihlal yok |
| Uydurulmuş kavram | 3 | **gerçek** |

**Bu sonucun anlamı ne?**

Sıfır ihlal beklemek gerçekçi değildi ve sonuç bunu doğruluyor. Ama dağılım
çok bilgilendirici:

* **En kritik ihlal türü hiç gerçekleşmedi.** Ajan hiçbir yanıtta cinsiyet,
  medeni durum veya uyruk üzerinden gerekçe sunmadı (0/24). Prompt'un mutlak
  kuralları en ağır ihlalde tuttu.
* **İhlallerin çoğu tek bir kalıptan geliyor.** 25 çerçeveleme hatasının
  hepsi aynı hata: "%21.3 riski artırıyor". Model, verilen bir sayıyı
  *uydurmuyor* — onu **yeniden yorumluyor**. Bu, 7B ölçeğinde tipik bir
  davranış.
* **Tool çağrıldığında yanıtlar temiz.** `get_decision_explanation` çağrılan
  ilk yanıtlarda 26-30 sayı doğru temellendirildi. İhlallerin ağırlığı,
  ajanın oturum geçmişine dayanıp tool çağırmadığı takip sorularında.

**Ana bulgu: prompt mühendisliği tek başına yetmiyor.** Sistem promptu üç kez
sertleştirildi (kural 0 eklendi, yasaklı kalıplar listelendi, madde şablonu
verildi) ve çerçeveleme hataları 42'den 25'e indi — ama sıfırlanmadı. Bir
7B modelden kural uyumu *rica* etmek, onu *garanti etmek* değildir.

## Onarım döngüsü: denetçiyi güvenlik mekanizmasına çevirmek

İhlalleri programatik olarak tespit edebiliyorsak, onları modele geri
bildirebiliriz de. `CreditAgent.ask_verified()` bunu yapar:

```
soru → yanıt → denetim → ihlal var mı?
                            ├─ hayır → yanıtı döndür
                            └─ evet  → ihlalleri modele bildir,
                                       yeniden yazdır, tekrar denetle
                                       → EN AZ ihlalli denemeyi döndür
```

Modele gönderilen düzeltme mesajı soyut değil, cerrahi:

```
DUR. Yanıtın otomatik sadakat denetiminden geçemedi. Bulunan sorunlar:

- %21.3 bir ETKİ PAYI'dır, risk oranı değildir. '%21.3 oranında
  artırıyor/azaltıyor' yazma. Yerine: 'kararın %21.3'ini oluşturuyor' yaz.
- Varsayımsal bir soruya tool çağırmadan cevap verdin.
  ŞİMDİ run_what_if tool'unu çağır ve gerçek sonucu kullan.

Şimdi yanıtını BAŞTAN yaz.
```

Döngü **asla kötüleştirmez**: tüm denemeler arasından en az ihlalli olan
döndürülür (`test_agent_ask_verified_never_returns_worse_answer`). Arayüzde
varsayılan olarak açıktır ("🛡️ Denetimli mod") ve onarımın devreye girdiği
kullanıcıya gösterilir.

### Onarımın ölçülen etkisi

```bash
uv run python scripts/evaluate.py --n 5 --repair 1
```

5 başvuru × 3 soru = 15 yanıt, aynı koşuda ham ve onarılmış skorlar:

| | Ham | Onarılmış |
|---|---:|---:|
| İhlalsiz yanıt | 7 / 15 | **8 / 15** |
| Sadakat skoru | %46.7 | **%53.3** |
| Toplam ihlal | 37 | **32** |

Onarım 8 yanıtta denendi: **5'inde ihlal sayısı azaldı**, 1'i tamamen temizlendi.

**Ama asıl sonuç ihlal türlerinin dağılımında:**

| İhlal türü | Onarım öncesi (n=8 koşusu) | Onarım sonrası |
|---|---:|---:|
| Çerçeveleme hatası | 25 | **0** ✅ |
| Uydurulmuş kavram | 3 | **0** ✅ |
| Korunan özellik gerekçesi | 0 | **0** ✅ |
| Yön çelişkisi | 14 | 5 |
| Temellenmemiş sayı | 18 | 25 |
| Eksik tool çağrısı | 2 | 2 |

![Onarım döngüsünün etkisi](docs/images/08-ihlal-turleri.png)

Onarım döngüsü **kalıp temelli ihlalleri tamamen siliyor**. Sebep açık:
"%21.3 bir etki payıdır, 'oluşturuyor' yaz" talimatı mekanik ve tek adımda
uygulanabilir. Model bunu yapabiliyor.

**Kalan sorun ise yapısal:** temellenmemiş sayılar azalmıyor, çünkü onları
düzeltmek modelin **tool'u yeniden çağırmasını** gerektiriyor. Ajan takip
sorularında oturum geçmişine güvenip tool çağırmıyor ve o boşluğu uydurulmuş
sayılarla dolduruyor. Bunu düzeltmek prompt'la değil, mimariyle çözülür:
tool çağrısı zorunlu kılınmalı (`tool_choice="required"`) ya da takip soruları
her seferinde taze bir açıklama yüküyle beslenmeli.

**Bir uyarı — skorlar koşudan koşuya değişiyor.** Ham sadakat skoru n=8'lik
koşuda %33.3, n=5'lik koşuda %46.7 ölçüldü. Sebep hem LLM'in belirlenimsizliği
hem de başvuru örnekleminin farkı. Tek bir sayıyı "projenin skoru" diye sunmak
yanıltıcı olur; anlamlı olan **ihlal türlerinin dağılımı** ve onarımın hangi
türlerde işe yaradığı.

Bu, projenin asıl tezi: **açıklanabilirlik iddiası olan bir sistemde LLM
katmanına güvenmek zorunda değilsiniz — onu denetleyebilir, ölçebilir ve
kısmen düzeltebilirsiniz.** Denetçi olmadan bu ihlallerin hiçbiri görünmezdi;
denetçi olmadan onarım da mümkün olmazdı.

## Adalet denetimi

Cinsiyet ve uyruk bilgisini modelden **çıkardık**. Bu yeterli mi? Hayır — buna
*"fairness through unawareness"* denir ve tek başına çalışmaz, çünkü modelde
kalan özellikler korunan özelliklerle korele olabilir (*vekil ayrımcılık*).

| Özellik | Modelde? | Red oranı farkı | Fırsat eşitliği farkı |
|---|---|---:|---:|
| Medeni durum / cinsiyet | ❌ dışlanmış | **0.103** ⚠️ | 0.038 |
| Yabancı işçi | ❌ dışlanmış | grup çok küçük (n<20) | — |
| Yaş grubu | ✅ modelde | **0.298** 🔴 | 0.182 ⚠️ |

![Adalet denetimi](docs/images/09-adalet.png)

Sağdaki panel yaş gruplarını modelin **ret oranı** ile **gerçek riskli oranı**
yan yana gösteriyor. Farkın bir kısmı gerçek risk farkından geliyor (19–30
grubunda gerçek riskli oranı 0.42, 56+ grubunda 0.10) — ama ret oranı farkı
(0.30) tek başına bunu açıklamıyor: 41–55 grubunun gerçek riski 56+ grubuna
yakın olmasına rağmen ret oranı belirgin biçimde düşük değil.

**Bulgular dürüstçe raporlanıyor:** cinsiyet modelden çıkarılmış olmasına rağmen
gruplar arası red oranı farkı 10.3 puan — vekil ayrımcılık işareti. Yaş modelde
olduğu için fark daha büyük (29.8 puan). Bu, üretime alınacak bir modelde
incelenmesi gereken bir bulgudur ve rapor bunu saklamıyor.

Rapor bilinçli olarak **geçti/kaldı damgası vermez**: demografik eşitlik, fırsat
eşitliği ve öngörü eşitliğinin aynı anda sağlanması taban oranlar eşit değilse
matematiksel olarak imkânsızdır (Kleinberg ve ark., 2016). Dürüst olan, farkları
ölçüp karar vericinin önüne koymaktır.

---

## Arayüz turu

Beş sekme. Aşağıdaki görüntüler **canlı uygulamadan**, gerçek bir ajan yanıtı
üretildikten sonra alındı (üretim: `scripts/make_video.py --ekran-goruntusu`).

### 🗣️ Açıklama — anlatı ve matematik yan yana

![Açıklama sekmesi](docs/images/ui/01-aciklama.png)

Solda ajanın Türkçe anlatısı, sağda **aynı kararın** SHAP şelalesi. Grafiğin
altındaki satır toplanabilirlik kontrolünü gösteriyor:
`taban (-0.4365) + katkılar (-0.6135) = -1.0500 ≈ modelin çıktısı`, hata
`4.4e-16 → GEÇTİ`. Üstteki karar kartı risk oranını, eşiği ve **KIL PAYI**
güven bandını birlikte veriyor — eşiğe 0.1 puan uzaklıktaki bir kararın
"kesin" sunulması yanıltıcı olurdu.

🛡️ **Denetimli mod** varsayılan olarak açık. Onarım devreye girdiyse
kullanıcıya söylenir; denetim sonucu her zaman ekranda.

### 📊 Etkenler — 18 katkının tamamı

![Etkenler sekmesi](docs/images/ui/02-etkenler.png)

Şelale grafiği ilk 10 katkıyı gösteriyor; bu sekme **hepsini** tablo olarak
veriyor: yön, ham SHAP değeri, etki payı ve başvuru sahibinin bu özelliği
değiştirip değiştiremeyeceği. Son kolon önemli — "yaşınızı büyütün" diye bir
tavsiye anlamsızdır.

### 🔀 What-if — karşı-olgusal senaryo

![What-if sekmesi](docs/images/ui/03-whatif.png)

Bir özelliği değiştirip modeli **gerçekten yeniden koşar**. Buradaki yeni risk
oranı bir tahmin ya da yorum değil, LightGBM'in değiştirilmiş girdiyle ürettiği
çıktı. Aralık dışı değerler reddedilir (`_coerce_value`): modelin hiç görmediği
bir yaşta "tahmin" üretmek ekstrapolasyondur, açıklama değildir.

### 💬 Sohbet — takip soruları

![Sohbet sekmesi](docs/images/ui/04-sohbet.png)

Her yanıtın altında hangi tool'ların çağrıldığı ve denetim sonucu görünür.
**Bu sekme sistemin en zayıf noktası** — ajan takip turunda tool çağırmayıp
sayı uydurabiliyor. Denetçi bunu yakalıyor ve ekranda gösteriyor; gizlemiyor.

### 🧪 Denetim — ölçümler tek ekranda

![Denetim sekmesi](docs/images/ui/05-denetim.png)

`artifacts/` altındaki JSON raporları okunup gösterilir: SHAP sadakat kazancı,
ajan sadakat skoru, adalet uçurumları ve model metrikleri. Rapor yoksa
uygulama hangi komutu çalıştırmanız gerektiğini söyler.

---

## Kalite güvencesi — 148 test

```bash
uv run pytest -m "not llm"   # 142 test, LLM gerekmez, ~30 sn
uv run pytest                # + 6 canlı ajan testi
```

| Dosya | Test | Ne korur |
|---|---:|---|
| `test_data_model.py` | 27 | Kategorik seviye sırası, hedef kodlama, eşik seçiminde sızıntı olmaması |
| `test_explainer.py` | 25 | Toplanabilirlik, SHAP kütüphanesiyle birebir uyum, what-if'in modeli gerçekten koşması |
| `test_faithfulness.py` | 40 | Denetçinin **7 ihlal türünü de yakalaması** ve yanlış alarm vermemesi |
| `test_smoke.py` | 16 | Uçtan uca akış, şemalar, yükte ham log-odds bulunmaması |
| `test_tools_agent.py` | 40 | Tool doğrulama, onarım mesajı, `ask_verified`'ın asla kötüleştirmemesi |

Testlerin önemli bir kısmı **gerçek hatalardan doğdu**. Denetçinin her yanlış
pozitifi için bir regresyon testi yazıldı; aşağıdaki "bulunan gerçek hatalar"
bölümündeki her maddenin karşılığı bir testtir.

**Bilinçli bir `xfail` var.** `test_bare_turn_tool_calling_is_unreliable_on_7b`,
onarımsız ilk turda 7B modelin tool çağırmasını bekler ve genellikle başarısız
olur. Bu testi "geçsin diye" yumuşatmak yanlış olurdu: ham yolun tool
çağırmaması **ölçülmüş bir gerçek** ve onarım döngüsünün var olma sebebi.
`xfail(strict=False)` bu kararsızlığı doğru modelliyor — geçerse `XPASS`,
geçmezse `xfail`; ikisi de takımı kırmıyor, ikisi de raporda görünüyor.

Ölçüm (aynı başvuru, temiz oturum):

| Yol | Tool çağrıldı mı |
|---|---|
| `ask_turn` (ham, onarımsız) | ✗ 0/5 |
| `ask_verified` (ürünün yolu) | ✓ 2/2 |

---

## Geliştirme sırasında bulunan ve düzeltilen gerçek hatalar

Bu bölüm projenin en öğretici kısmı. Her madde canlı bir gözlemden doğdu.

### 1. Model log-odds değerini yüzde sandı

İlk tasarımda ajana SHAP değerini ham hâlde verdik:

```json
{"ad": "Vadesiz hesap durumu", "katki": 0.83}
```

7B'lik model bunu okudu ve yazdı: *"bu etken **%83** risk katkıda bulunuyor."*
Log-odds bir yüzde değildir — doğrudan bir sadakat ihlali.

**Çözüm:** ajana hesaplaması gereken hiçbir sayı verilmiyor. Yük artık yalnızca
kopyalanmaya hazır metin taşıyor:

```json
{"ad": "Vadesiz hesap durumu", "deger": "bakiye negatif (0 DM altı)",
 "etki_yonu": "riski artırıyor", "etki_payi": "%19.6", "etki_gucu": "güçlü"}
```

Ham sayılar arayüz ve testler için `all_contributions` içinde korunuyor.
`test_agent_payload_has_no_raw_logodds` bu kuralı bekçilik ediyor.

### 2. Ajan varsayımsal soruda tool çağırmadan uydurdu

"Vade 24 aya düşse ne olurdu?" sorusuna model tool çağırmadan cevap verdi ve
eski risk oranını tekrarladı.

**Çözüm:** sistem promptunda mutlak kural + `audit_narrative`'de programatik
kontrol (varsayımsal soru kalıbı varsa `run_what_if` çağrılmış olmalı, yoksa
ihlal). Prompt ilk savunma hattı, denetim ikinci hattır.

### 3. Etki payı risk artışı gibi sunuldu (denetim kaçırdı)

Arayüzde ajan şunu yazdı: *"Mevcut İş Süresi: 1 yıldan az — bu durum, **riski
%13.7 oranında artırıyor**."* Sayı temelliydi (gerçek bir etki payıydı) ama
**anlamı** çarpıtılmıştı; gerçek risk oranı %44.4'tü.

Sayı denetimi bunu yakalamaz. `misframed_shares` kontrolü eklendi: bir etki payı
değeri "artır/azalt" fiiliyle doğrudan birleştirilmişse ihlal sayılıyor —
gerçek risk oranıyla birleştirilmesi ise meşru.

### 4. Cümle bölme ondalık sayıları parçalıyordu (sessiz hata)

Cümle düzeyindeki tüm denetimler `re.split(r"[.!?\n]+", text)` kullanıyordu.
"%21.6" ifadesi noktadan bölününce **"riski %21"** ve **"6 oranında artırıyor"**
diye ikiye ayrılıyordu. Sonuç: yön, korunan özellik ve çerçeveleme denetimleri
hiçbir şey kontrol etmiyor, testler yine de "geçti" diyordu.

`_split_sentences()` ondalık ayırıcıları bölmeden önce korumaya alıyor.
Regresyon testi: `test_sentence_split_preserves_decimals`.

### 5. Türkçe `.lower()` tuzağı

```python
>>> repr("DEĞİLDİR".lower())
'deği̇ldi̇r'          # 'İ' -> 'i' + U+0307 BİRLEŞİK NOKTA
>>> "değildir" in "DEĞİLDİR".lower()
False
```

Metin karşılaştırmaları bu yüzden sessizce başarısız oluyordu. `_fold()` büyük
`İ`'yi `lower()` çağrısından **önce** dönüştürüp NFKD ile birleşik işaretleri
atıyor. Regresyon testi: `test_fold_handles_turkish_dotted_capital_i`.

### 6. Erken durdurma modeli underfit bıraktı

Tek bir doğrulama seti (800'ün %20'si = 160 satır) üzerinde erken durdurma,
gürültülü bir AUC dalgalanmasını "iyileşme durdu" sanıp modeli **30 ağaçta**
kesti. Sonuç: test AUC 0.776, CV 0.751 ve küresel önemin **%50.8'i** tek bir
özelliğe (`checking_status`) yığılmıştı — açıklamalar tek boyutlu oluyordu.

`lgb.cv` ile 5 katın **ortalamasında** erken durdurmaya geçtik: 243 ağaç,
AUC 0.794, CV 0.793 ve en büyük özellik payı %19.6'ya indi.
Bekçi test: `test_global_importance_not_dominated_by_one_feature`.

### 7. `asyncio.run()` her çağrıda olay döngüsünü kapatıyordu

İkinci soruda `RuntimeError: Event loop is closed`. Sebep: `OllamaChatClient`
içindeki `httpx.AsyncClient` **oluşturulduğu döngüye bağlı**, ama her soru için
yeni bir döngü açıyorduk.

`_BackgroundLoop`: uygulama ömrü boyunca `run_forever` ile dönen tek bir döngü,
coroutine'ler `run_coroutine_threadsafe` ile gönderiliyor. Streamlit'in senkron
modeliyle de sorunsuz çalışıyor.

### 8. Ajan tool çağırmadığında BAŞKA bir başvuruyu anlatıyor

Demoyu ham (denetimsiz) modda koşarken en ağır halüsinasyon ortaya çıktı.
Ajan hiç tool çağırmadı ve şunu yazdı:

```
Risk oranı %12 ... eşik %30
- Kredi Miktari: 6500 — kararın %47'ini oluşturuyor
- Kredi Tarihi: "neu" — kararın %30'unu oluşturuyor
```

Gerçek başvuru: risk %25.9, eşik %26.0, tutar 3612 DM, yaş 37. Model,
eğitim verisinden hatırladığı **bambaşka bir German Credit kaydını** anlattı —
`"neu"` Almanca, veri setinin orijinal kodlamasından geliyor.

Denetçi 8 sayının tamamını yakaladı. Onarım mesajı artık kök sebebi en başa
koyuyor: *"HİÇ TOOL ÇAĞIRMADIN. Bu yüzden yazdığın sayılar bu başvuruya ait
değil; başka bir başvuruyu anlatıyorsun."* Bu düzeltmeyle aynı soru
**5 ihlal → 0** oldu.

### 9. Onarım turundan sonra model İngilizce'ye kayıyor

Düzeltme mesajını uyguladıktan sonra 7B model dili değiştirdi ve tool
çıktısını alan-alan dökmeye başladı (`**Feature:** ... **Impact Share:** ...`).
Sistem promptu Türkçe istiyordu ama model **son mesaja** ağırlık veriyor.

İki katmanlı çözüm: (a) sistem promptunun en başına mutlak dil kuralı,
(b) onarım mesajında dil ve biçim talimatının tekrarı, (c) denetçiye
`language_drift` kontrolü — İngilizce yanıt teknik olarak "sadık" olsa bile
Türkçe konuşan bir başvuru sahibi için kullanılamaz.

### 10. Kurulum tuzakları: `libomp` ve `llvmlite`

macOS'ta LightGBM `libomp` olmadan **import edilir ama eğitimde patlar**.
`test_lightgbm_openmp_works` gerçek bir eğitim koşturarak bunu yakalıyor.

Ayrıca `uv` ilk çözümlemede SHAP→numba→llvmlite zincirinde `llvmlite 0.36`'ya
düştü (Python 3.10 üstünü desteklemiyor). `pyproject.toml`'da `numba>=0.60` ve
`llvmlite>=0.43` alt sınırları bu yüzden açıkça belirtildi.

### 11. «anlamına gelir» fiili, gelir (income) kavramı sanıldı

**Bu hata tanıtım videosunun çekimi sırasında bulundu.** Ajanın denetimden
geçen bir yanıtında tek bir ihlal kaldı:

```
⚠️ SADAKAT DENETİMİ: 1 ihlal
    fabricated_concepts: ['gelir']
    sentence: "Risk oranı eşiğin altında, bu da riski daha düşük bir
               seviyeye indirgeyen bir karar anlamına gelir"
```

Cümlede uydurma yok. Türkçe'de **"gelir"** hem bir isim (income) hem
"gelmek" fiilinin geniş zaman çekimidir; alt-dizi araması ikisini ayırt
edemiyordu.

Kolay görünen çözüm yanlış: kelime sınırıyla aramak (`\bgelir\b`) bu yanlış
alarmı susturmaz — çünkü "gelir" orada zaten tam bir kelime — ama gerçek bir
uydurma olan **"geliriniz yetersiz"** ifadesini kaçırırdı. Doğru çözüm,
aramadan önce yalnızca **ölçülmüş fiil deyimlerini** metinden düşürmek:

```python
_VERB_IDIOM_RE = re.compile(
    r"(?:anlamina|manasina|haline|hale|ortaya|meydana|one|geri|ileri)\s+gel\w*"
)
```

Neden önemli: yanlış alarm veren bir denetçi, onarım döngüsünde modele **yanlış
geri bildirim** verir ve zaten doğru olan bir yanıtı bozar. Dört regresyon
testiyle kilitlendi — biri de deyimin gerçek uydurmayı gizlemediğini doğruluyor.

### 12. Test, ürünün vermediği bir sözü kontrol ediyordu

`test_agent_calls_tool_for_explanation` şunu iddia ediyordu: ajan, açıklama
isteğinde tool çağırmak *zorunda*. Test düştü ve ilk tepki "model bozuldu"
oldu. Teşhis üç adımda yapıldı:

1. **Değişikliğimin sebep olup olmadığını kanıtla.** `git stash` ile
   düzeltmeyi geri alıp test yeniden koşturuldu: aynı şekilde düştü. Yani
   sebep benim değişikliğim değildi.
2. **Ollama bağlam penceresini şüpheli gör.** Sistem promptu ~1681 token,
   varsayılan bağlam 4096 — tool tanımları kırpılıyor olabilir mi?
   `OLLAMA_CONTEXT_LENGTH=16384` ile yeniden denendi: **fark yok.** Hipotez
   yanlıştı ve varsayılmadan önce ölçüldü.
3. **Ham yol ile ürün yolunu ayrı ölç.** Asıl bulgu buydı:

   | Yol | Tool çağrıldı mı |
   |---|---|
   | `ask_turn` (ham) | ✗ 0/5 |
   | `ask_verified` (onarım açık) | ✓ 2/2 |

Tool çağrısını getiren şey **onarım turuydu**: denetçi "hiç tool çağrılmadı"
ihlalini görüyor, onarım mesajı modele *"ŞİMDİ get_decision_explanation
çağır"* diyor ve model çağırıyor.

Çözüm testi yumuşatmak değil, **doğru sözleşmeyi test etmek** oldu: hem arayüz
hem `explain_demo.py` varsayılan olarak denetimli modda çalışıyor, dolayısıyla
garanti edilen yol o. Ham yol için ayrı bir `xfail` testi bırakıldı ki sınır
görünür kalsın ve daha büyük bir modelle düzeldiğinde `XPASS` olarak haber
verilsin.

---

## Mimari kararlar ve gerekçeleri

### Neden LightGBM?

Kategorik özellikleri **doğrudan** işliyor. `purpose` gibi 10 seviyeli bir
sütunu one-hot ile 10 sütuna açmamız gerekmiyor. Bu XAI açısından belirleyici:
one-hot yapsaydık SHAP bize `purpose_used_car = +0.1`,
`purpose_new_car = -0.05` gibi 10 parça verecekti ve bunları tekrar
birleştirmemiz gerekecekti. Şimdi tek `purpose` özelliği için tek SHAP değeri
alıyoruz ve doğrudan insan diline çevrilebiliyor.

Ağaçlar bilinçli olarak sığ (`num_leaves=4`, `max_depth=3`): 800 satırda derin
ağaçlar ezberler ve SHAP etkileşimleri yorumlanamaz hâle gelir. 5-kat CV ile
seçildi.

### Neden TreeExplainer, neden LightGBM'in kendi uygulaması?

Shapley değerinin naif tanımı 2^18 = 262.144 alt küme demek — her başvuru için.
`KernelExplainer` bunu örnekleyerek **yaklaşık** hesaplar (yavaş, gürültülü).
Lundberg'in TreeSHAP algoritması ağaç yapısını gezerek aynı sonucu **kesin** ve
polinom zamanda verir.

Bu projede TreeSHAP'ı LightGBM'in gömülü `predict(pred_contrib=True)` yoluyla
çağırıyoruz çünkü: (a) pandas `category` tipini doğal işler, (b) ekstra bellek
kopyası yaratmaz, (c) `shap` paketiyle **birebir aynı** sonucu verir. (c)
bağımsız bir testle doğrulanıyor: `test_native_shap_matches_shap_package`
(maksimum fark 0.0). `shap` paketi yalnızca grafik için kullanılıyor.

### Neden korunan özellikler modelden çıkarıldı?

`personal_status` (cinsiyet + medeni durum) ve `foreign_worker` (uyruk), kredi
kararında kullanılması ayrımcılık oluşturan özelliklerdir. Veri setinden
silinmiyorlar — adalet denetimi için ayrı bir çerçevede saklanıyor ve modelde
olmadıkları hâlde grup farkları ölçülüyor.

`age` hassas olarak işaretli: modelde tutuluyor ama adalet raporunda izleniyor.

### Neden Microsoft Agent Framework?

Tool-calling döngüsünü elle yazmak mümkün ama sıkıcı ayrıntılarla dolu: paralel
çağrılar, çağrı kimlikleri, hata durumunda modele ne döneceği, oturum geçmişi.
Agent Framework bunları kapsıyor ve `tools=[python_fonksiyonu]` demek yeterli —
JSON şemasını tip imzalarından ve docstring'den kendisi üretiyor.

Ayrıca `agent_framework.ollama` ve `agent_framework.foundry` alt modülleri
sayesinde arka uç değişimi tek satır: `XAI_LLM_BACKEND=foundry`.

Kurulu paketler bilinçli olarak granüler seçildi (meta-paket `agent-framework`
redis/boto3/sqlalchemy gibi onlarca gereksiz bağımlılık çekiyordu):

```
agent-framework-core    agent-framework-ollama
agent-framework-openai  agent-framework-foundry-local
```

### Azure AI Foundry Local'a geçiş

Geliştirme Ollama ile yapıldı (model zaten kuruluydu, iterasyon hızı önemliydi).
Foundry Local istemcisi kurulu ve `llm.py` içinde bağlı; geçiş için:

```bash
brew tap microsoft/foundrylocal && brew install foundrylocal
foundry model run phi-4-mini
```

```bash
# .env
XAI_LLM_BACKEND=foundry
XAI_LLM_MODEL=phi-4-mini
```

Kod değişikliği gerekmez. `FoundryLocalClient` modeli kendisi indirir
(`bootstrap=True`) ve ONNX Runtime ile donanım hızlandırma kullanır.
`test_foundry_local_importable` istemcinin kurulu olduğunu doğrular.

---

## Proje yapısı

```
xai-agent/
├── src/xai_agent/
│   ├── config.py         # tüm yollar, sabitler, LLM ayarları (tek doğruluk kaynağı)
│   ├── features.py       # 20 özelliğin Türkçe sözlüğü + sabit kategori seviyeleri
│   ├── data.py           # indirme, önbellekleme, tip zorlama, katmanlı bölme
│   ├── model.py          # LightGBM eğitimi, baseline'lar, maliyet-optimal eşik
│   ├── explainer.py      # TreeSHAP, toplanabilirlik, what-if, şelale grafiği
│   ├── schemas.py        # SHAP→JSON sözleşmesi (Pydantic) + ajan yükü
│   ├── prompts.py        # sistem promptu ("anayasa") + onarım mesajı üretici
│   ├── tools.py          # ajanın 4 tool'u + değer doğrulama
│   ├── llm.py            # Ollama / Foundry Local / Azure OpenAI fabrikası
│   ├── agent.py          # Agent Framework sarmalayıcısı + onarım döngüsü
│   ├── faithfulness.py   # ERASER metrikleri + anlatı denetçisi
│   └── fairness.py       # demografik eşitlik, fırsat eşitliği, vekil ayrımcılık
├── app/streamlit_app.py  # 5 sekmeli arayüz
├── scripts/
│   ├── train.py          # Faz 1+2 çalıştırıcısı
│   ├── explain_demo.py   # terminal demosu
│   ├── evaluate.py       # Faz 5 çalıştırıcısı
│   ├── make_figures.py   # README'deki 10 grafiği artifacts/*.json'dan üretir
│   ├── make_video.py     # tanıtım videolarının uçtan uca üretim hattı
│   └── video/
│       ├── narration.py       # sahneler + anlatım metni (ses ve altyazının tek kaynağı)
│       ├── tts.py             # Türkçe seslendirme (say -v Yelda) + süre ölçümü
│       ├── capture_outputs.py # videodaki terminal çıktılarını GERÇEKTEN çalıştırır
│       ├── html_scenes.py     # kart / figür / terminal sahnelerini HTML olarak üretir
│       ├── ui_scenes.py       # canlı Streamlit'i Playwright ile sürer
│       ├── capture.py         # sahneleri video klibine çevirir (record_video_dir)
│       └── assemble.py        # ses/görüntü hizalama, birleştirme, GIF
├── tests/                # 142 test (LLM'siz) + 6 canlı ajan testi
├── docs/
│   ├── images/           # 10 rapor grafiği + ui/ altında 5 arayüz görüntüsü
│   └── video/            # mp4 + srt
└── artifacts/            # metrics.json, faithfulness_report.json, fairness_report.json
```

### Arayüz sekmeleri

| Sekme | İçerik |
|---|---|
| 🗣️ Açıklama | Ajan anlatısı **+ yanında SHAP şelale grafiği** + sadakat denetimi sonucu + 🛡️ denetimli mod anahtarı |
| 📊 Etkenler | 18 özelliğin tam dökümü, ham SHAP değerleriyle |
| 🔀 What-if | Bir özelliği değiştir, modeli yeniden koştur, kararın döndüğünü gör |
| 💬 Sohbet | Takip soruları; her yanıtın altında çağrılan tool'lar, denetim sonucu ve onarım bilgisi |
| 🧪 Denetim | Sadakat ve adalet raporlarının tamamı |

Ajanın anlatısı bilinçli olarak **grafiğin yanında** gösteriliyor: kullanıcı
ajanın sözünü doğrulayabilmeli. Açıklanabilirlik iddiası olan bir sistemde LLM
metnini tek başına sunmak, kullanıcıyı ikinci bir kara kutuya mahkûm etmek olur.

---

## Yeniden üretilebilirlik

* `uv.lock` her paketin tam sürümünü ve hash'ini sabitler → `uv sync` birebir
  aynı ortamı kurar.
* `RANDOM_SEED = 42` veri bölme, model eğitimi ve rastgele kontrol grubunda
  kullanılır. `test_split_is_deterministic` bunu doğrular.
* Veri bir kez indirilip `data/raw/` içine önbelleklenir — sonraki çalıştırmalar
  ağ gerektirmez.
* Kategori seviyeleri `features.py` içinde sabittir. Sıra kayması sessizce
  yanlış tahmine yol açar; `test_categorical_levels_are_pinned` bunu yakalar.

**Rapordaki her görsel ve video yeniden üretilebilir:**

```bash
# README'deki 10 grafik — artifacts/*.json ve eğitilmiş modelden
uv run python scripts/make_figures.py

# Videodaki terminal çıktıları (gerçekten çalıştırılır, ~6 dk; LLM gerekir)
uv run python scripts/video/capture_outputs.py

# Arayüz ekran görüntüleri (canlı Ollama + Streamlit gerekir)
uv run python scripts/make_video.py --ekran-goruntusu

# Videolar: seslendirme → sahne kaydı → hizalama → mp4 + srt + gif
uv run python scripts/make_video.py tanitim derin --gif 30
```

Video hattının iki tasarım kararı belgelemeye değer:

* **Ses ile altyazı aynı yerden üretilir** (`narration.py`). Sistemdeki tek
  Türkçe ses olan Yelda "SHAP", "LightGBM", "AUC" gibi terimleri doğru
  okuyamıyor; bu yüzden her sahnenin bir `text` (altyazıda görünen, doğru
  yazımlı) ve bir `speech` (seslendirmeye giden, fonetik) alanı var. İkisi de
  tek kaynaktan geldiği için kaymaları imkânsız.
* **Anlatım süresi görüntüyü belirler, tersi değil.** Önce seslendirme
  üretilip süresi ölçülür, sonra sahne o süreye göre kaydedilir. Görüntü
  kısaysa son kare tutulur; uzunsa hızlandırılır ve ekrana
  "×N hızlandırıldı" rozeti basılır — canlı LLM sahnelerinde bu 2-3 kat
  olabiliyor ve izleyiciyi yanıltmamak gerekiyor.

Bu ffmpeg derlemesinde `libass`/`freetype` yok, dolayısıyla `subtitles` ve
`drawtext` filtreleri kullanılamıyor: altyazı videoya gömülmüyor, yanında
`.srt` olarak veriliyor; hız rozeti de PIL ile PNG üretilip `overlay`
filtresiyle bindiriliyor.

## Bilinen sınırlar

* **Veri seti küçük ve eski.** German Credit 1000 satır, 1994'ten ve Alman Markı
  cinsinden. Metrikler gerçek bir kredi portföyünü temsil etmez; hedef, boru
  hattını ve denetim mekanizmasını göstermek.
* **7B model yeniden yazım yapıyor.** Prompt "adı aynen kopyala" dese de
  qwen2.5:7b bazen "Mevcut işte çalışma süresi"ni "Mevcut İş Süresi" diye
  yazıyor. Denetçi bunu tolere edecek biçimde yazıldı (bulanık ad eşleştirme);
  daha büyük bir model bu sorunu azaltır. Davranış **koşudan koşuya değişiyor**:
  aynı soru, aynı model, bir koşuda kanonik adlar, diğerinde yeniden yazılmış
  adlar üretebiliyor.
* **Ham ilk turda tool çağrısı kararsız.** Ölçüm: `ask_turn` ile 0/5,
  `ask_verified` ile 2/2. Yani onarım döngüsü bir iyileştirme değil, sistemin
  temellenmesi için **zorunlu** bir bileşen. Kalıcı çözüm mimari:
  `tool_choice="required"` ya da her turda taze yük enjeksiyonu.
* **"Sadakat denetimi geçti" ≠ "yanıt doğru".** Denetçi sayısal sadakati
  doğrular; nitel iddiaların tutarlılığını doğrulamaz. Ölçülen örnek: onarımdan
  sonra 0 ihlalle geçen bir yanıt, **onaylanmış** ve karar netliği **KIL PAYI**
  olan bir başvuru için son cümlede *"başvurunun riski oldukça yüksektir"*
  dedi. Cümlede temelsiz sayı olmadığı için denetim geçti. Bu bir hata değil,
  denetçinin **kapsamının sınırı** — ve bu sınırı bilmek denetime güvenmenin
  ön koşulu.
* **Uydurma kavram listesi elle yazılmış.** Sayı temellendirmesi genel bir
  kural (yükte olmayan her rakam ihlaldir), ama kavram denetimi sabit bir
  liste. Ölçülen örnek: ajan *"Faiz oranı: %9"* diye veri setinde hiç olmayan
  bir özellik icat etti ve *"Kredi türü: andreavonbalken"* yazdı; denetçi bu
  ifadelerin **sayılarını** yakaladı (5 temelsiz sayı) ama **terimlerini**
  yakalamadı. Asıl iş gücü sayı denetimi; terim denetimi yardımcı.
* **Yaş grubu farkı (0.298) üretime uygun değil.** Rapor bunu işaretliyor ama
  düzeltmiyor. Gerçek bir dağıtımda yaş özelliğinin çıkarılması veya
  post-processing ile eşitleme gerekir.
* **Sadakat denetçisi kural tabanlı.** Regex ve kelime listeleri kullanıyor;
  yeterince yaratıcı bir yanlış ifade kaçabilir. `prompts.CLAIM_EXTRACTOR_PROMPT`
  LLM tabanlı iddia çıkarımı için hazır ama varsayılan akışta kullanılmıyor
  (belirlenimli davranış tercih edildi).
* **Onarım döngüsünün bedeli gecikme.** İhlalli her yanıt için ikinci bir LLM
  turu gerekiyor; yerel 7B modelde bu yanıt süresini iki katına çıkarıyor.
  Arayüzde kapatılabilir bir anahtar olarak sunuldu.
* **Ollama eşzamanlı yükte çökebiliyor.** Denetim koşusu ile arayüzü aynı anda
  kullanırken `llama-server process no longer running` hatası alındı. Ajan
  katmanı bu durumu yakalayıp kullanıcıya net mesaj veriyor
  (`LLMBackendError`), ama üretim için bir kuyruk/limit katmanı gerekir.
* **Sadakat ölçümü tek bir modelle yapıldı.** Tüm sayılar
  `qwen2.5:7b-instruct` içindir. Daha büyük bir modelle (veya Foundry Local
  üzerinde `phi-4`) ölçüm tekrarlanmadı; skorların model ölçeğiyle nasıl
  değiştiği açık bir soru.

## Sonraki adımlar

Projeyi ileri taşımak isteyen için, önem sırasına göre:

1. **Ölçümü daha büyük bir modelle tekrarla.** Çerçeveleme hatalarının 7B'ye
   özgü olup olmadığını görmek en yüksek bilgi değerli deney.
   `XAI_LLM_MODEL` değiştirmek yeterli.
2. **Yaş özelliğini çıkarıp maliyet/adalet ödünleşimini ölç.**
   `config.SENSITIVE_FEATURES`'ı `PROTECTED_FEATURES`'a taşıyıp
   `scripts/train.py` ve `scripts/evaluate.py`'yi yeniden koşmak yeterli;
   0.298'lik red oranı farkının ne kadar düştüğü ve maliyetin ne kadar
   arttığı doğrudan karşılaştırılabilir.
3. **Denetçiye LLM tabanlı iddia çıkarımı ekle.** `CLAIM_EXTRACTOR_PROMPT`
   hazır; kural tabanlı denetimin kaçırdığı yaratıcı ifadeleri yakalayabilir.
   Belirlenimlilik kaybı karşılığında kapsam kazanılır.
4. **Foundry Local'a geç ve ölçümü tekrarla.** `XAI_LLM_BACKEND=foundry`
   tek satırlık değişiklik; kıyaslama tablosu doğrudan üretilebilir.
5. **Tool çağrısını mimari olarak zorunlu kıl.** İlk turda `tool_choice="required"`
   kullanmak, "ham yolda 0/5 tool çağrısı" sonucunu doğrudan hedef alır ve
   onarım turunun yükünü azaltır. `test_bare_turn_tool_calling_is_unreliable_on_7b`
   testi `XPASS`'a dönerse çözüldüğünü haber verir.
6. **Denetçiye özellik adı sadakati kontrolü ekle.** Ajan bazı koşularda
   özellik adlarını yeniden yazıyor ve müşteri o etiketi banka kayıtlarında
   bulamıyor. Yükteki `ad` alanı ile metindeki kalın başlıkları karşılaştıran
   bir kontrol, sayı denetimiyle aynı mekanikle çalışabilir.

## Kaynaklar

* Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* — SHAP
* Lundberg ve ark. (2020), *From local explanations to global understanding with explainable AI for trees* — TreeSHAP
* DeYoung ve ark. (2020), *ERASER: A Benchmark to Evaluate Rationalized NLP Models* — comprehensiveness / sufficiency
* Hardt, Price & Srebro (2016), *Equality of Opportunity in Supervised Learning*
* Kleinberg, Mullainathan & Raghavan (2016), *Inherent Trade-Offs in the Fair Determination of Risk Scores*
* Hofmann (1994), *Statlog (German Credit Data)*, UCI Machine Learning Repository

## Yazar

**Mehmet Efe Aytaş**
· [github.com/mehmetefeaytas](https://github.com/mehmetefeaytas)

Proje Microsoft staj programı kapsamında bitirme/portföy çalışması olarak
geliştirilmiştir.

## Lisans

MIT — bkz. [LICENSE](LICENSE)
