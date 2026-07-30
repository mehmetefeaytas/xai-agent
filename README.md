# XAI-Agent — Şeffaf Kredi Risk Ajanı

**Mehmet Efe Aytaş** · Microsoft staj / bitirme projesi

Bir kredi başvurusunu **reddeden** makine öğrenmesi modelinin kararını, SHAP ile
matematiksel olarak parçalayıp bir LLM ajanı aracılığıyla başvuru sahibine
**doğal Türkçe** ile açıklayan uçtan uca sistem.

Projenin ayırt edici iddiası şu: **ajanın söylediği her cümle programatik olarak
denetleniyor.** Şeffaflık burada bir slogan değil, ölçülen bir metrik.

```
┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
│  LightGBM    │──▶│      SHAP      │──▶│   LLM Ajanı      │
│              │   │                │   │ (Agent Framework)│
│ TAHMİN eder  │   │ NEDEN'i        │   │ DİLE çevirir     │
│              │   │ sayısallaştırır│   │                  │
└──────────────┘   └────────────────┘   └──────────────────┘
       │                   │                     │
       ▼                   ▼                     ▼
  risk = %90.9      18 katkı + JSON       "Başvurunuz şu üç
  eşik = %26        toplanabilirlik ✅      nedenle olumsuz..."
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │  SADAKAT DENETÇİSİ    │
                                     │ uydurma sayı/özellik/ │
                                     │ yön/çerçeveleme avı   │
                                     └───────────────────────┘
```

**Kritik sorumluluk ayrımı:** LLM tahmin yapmaz, karar vermez, hesaplamaz.
Elinde yalnızca modele bağlı dört tool var. "Vade 12 aya düşse ne olurdu?"
sorusunun cevabı LLM'in tahmini değil, **LightGBM'in yeniden koşumudur**.

### Bir bakışta

| Faz | Ne yapıldı | Ölçülen sonuç |
|---|---|---|
| **1** | LightGBM + baseline karşılaştırması | maliyet **104** (LogReg 115, dummy 300), recall **0.917**, CV AUC 0.793 ± 0.039 |
| **2** | TreeSHAP + SHAP→JSON köprüsü | toplanabilirlik hatası **2.2e-15**, SHAP rastgeleyi **2.16×** yeniyor |
| **3** | Agent Framework + 4 tool + sistem promptu | tool çağrılan yanıtlarda 26–30 sayı doğru temellendirildi |
| **4** | Streamlit arayüzü (5 sekme) | ajan anlatısı SHAP grafiğinin yanında, denetim sonucu ekranda |
| **5** | Sadakat + adalet denetimi + onarım döngüsü | ham sadakat %33–47 (koşuya göre); onarım **çerçeveleme ihlallerini 25→0** yaptı; korunan özellik ihlali **0** |

Test kapsamı: **132** LLM'siz test + **6** canlı ajan testi, `ruff` temiz.
Her ölçüm `artifacts/` altındaki JSON raporlarından yeniden üretilebilir.

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
uv run pytest -q -m "not llm"                        # 132 test, LLM gerekmez
uv run pytest -q                                     # canlı ajan testleri dahil
```

### Üç katmanın çıktısı, tek ekranda

```
KATMAN 1 — LightGBM tahmini  (A-013)
  Karar          : ONAYLANDI
  Risk oranı     : 25.92%
  Karar eşiği    : 26.00%
  Eşiğe uzaklık  : -0.08% (sinirda)      ← kıl payı onaylandı

KATMAN 2 — SHAP gerekçesi
  Toplanabilirlik: taban -0.4365 + katkılar -0.6135 = -1.0500
                   model çıktısı  = -1.0500  (hata 4.4e-16) -> GEÇTİ

  Riski ARTIRAN etkenler (toplam etkinin %39'i):
    ▲ Vadesiz hesap durumu     bakiye 0–200 DM arası   % 16.4  (+0.4750)
    ▲ Kredi vadesi             18 ay                   %  8.9  (+0.2575)
  Riski AZALTAN etkenler:
    ▼ Kredi geçmişi            kritik hesap ...        % 21.2  (-0.6111)
    ▼ Kredi tutarı             3612 DM                 % 16.3  (-0.4717)

KATMAN 3 — LLM ajanı
  [tool çağrıları: get_decision_explanation]
  "Başvuru onaylandı. Model risk oranını %25.9 olarak hesapladı; bankanın
   karar eşiği %26.0. Karar KIL PAYI — küçük bir değişiklik sonucu çevirebilir..."

  ✅ SADAKAT DENETİMİ GEÇTİ (26 sayı temellendirildi)
```

---

## Sonuçlar

### Faz 1 — Model performansı (test seti, n=200)

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

Bu eşiğin bedeli görünür: 140 iyi müşteriden 79'u gereksiz reddediliyor (FP),
ama 60 riskli müşteriden yalnızca 5'i kaçıyor (FN). Maliyet matrisi bunu kârlı
buluyor. Oran `config.COST_FALSE_NEGATIVE` ile değiştirilebilir.

Karşılaştırma — aynı model, saf 0.5 eşiği: accuracy %72 (daha *iyi* görünür)
ama recall %73.3 ve maliyet **120** (daha *kötü*).

### Faz 2 — SHAP modele sadık mı?

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

**SHAP kazancı: 2.16× rastgele** (n=150) → *"GÜÇLÜ — SHAP sıralaması modelin
davranışını isabetle yakalıyor."* Bu kontrol grubu olmadan comprehensiveness
sayıları tek başına bir şey söylemez.

**Toplanabilirlik (yerel doğruluk) aksiyomu:** test setinin tamamında
`taban_değer + Σ SHAP = modelin ham çıktısı`, maksimum hata **2.2e-15**
(makine hassasiyeti). Bu eşitlik her açıklamanın içine gömülüdür — bir açıklama
kendi doğruluğunun kanıtını taşımalı.

**Modelin en etkili özellikleri** (ortalama |SHAP|):

| # | Özellik | Pay |
|---:|---|---:|
| 1 | Vadesiz hesap durumu | %19.6 |
| 2 | Kredi vadesi | %13.4 |
| 3 | Kredi kullanım amacı | %10.0 |
| 4 | Kredi geçmişi | %9.4 |
| 5 | Kredi tutarı | %8.9 |

### Faz 5 — Ajan anlatısı SHAP'a sadık mı?

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

### Denetçiyi güvenlik mekanizmasına çevirmek: onarım döngüsü

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

#### Ölçülen etki

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

### Faz 5 — Adalet denetimi

Cinsiyet ve uyruk bilgisini modelden **çıkardık**. Bu yeterli mi? Hayır — buna
*"fairness through unawareness"* denir ve tek başına çalışmaz, çünkü modelde
kalan özellikler korunan özelliklerle korele olabilir (*vekil ayrımcılık*).

| Özellik | Modelde? | Red oranı farkı | Fırsat eşitliği farkı |
|---|---|---:|---:|
| Medeni durum / cinsiyet | ❌ dışlanmış | **0.103** ⚠️ | 0.038 |
| Yabancı işçi | ❌ dışlanmış | grup çok küçük (n<20) | — |
| Yaş grubu | ✅ modelde | **0.298** 🔴 | 0.182 ⚠️ |

**Bulgular dürüstçe raporlanıyor:** cinsiyet modelden çıkarılmış olmasına rağmen
gruplar arası red oranı farkı 10.3 puan — vekil ayrımcılık işareti. Yaş modelde
olduğu için fark daha büyük (29.8 puan). Bu, üretime alınacak bir modelde
incelenmesi gereken bir bulgudur ve rapor bunu saklamıyor.

Rapor bilinçli olarak **geçti/kaldı damgası vermez**: demografik eşitlik, fırsat
eşitliği ve öngörü eşitliğinin aynı anda sağlanması taban oranlar eşit değilse
matematiksel olarak imkânsızdır (Kleinberg ve ark., 2016). Dürüst olan, farkları
ölçüp karar vericinin önüne koymaktır.

---

## Ajan katmanı ve sadakat denetimi

### Dört tool — LLM ile modelin arasındaki tek köprü

| Tool | Ne yapar |
|---|---|
| `get_decision_explanation` | Bu başvurunun SHAP açıklamasını döndürür |
| `run_what_if` | Bir özelliği değiştirip modeli **gerçekten** yeniden koşar |
| `get_feature_info` | Bir özelliğin tanımı ve geçerli değer aralığı |
| `get_global_importance` | Model düzeyinde özellik önemi |

### Denetlenen ihlal türleri

`faithfulness.audit_narrative()` her ajan yanıtını tarar:

| İhlal | Örnek | Nasıl yakalanır |
|---|---|---|
| **Temellenmemiş sayı** | "Risk oranınız %99.7" | Metindeki her sayı, yükteki izinli sayı kümesiyle karşılaştırılır |
| **Uydurulmuş kavram** | "Geliriniz yetersiz" | Model gelir bilgisi görmez; yasaklı kavram listesi |
| **Yön çelişkisi** | SHAP "azaltıyor" derken metin "artırıyor" | Cümle düzeyinde işaret karşılaştırması (yeniden yazılmış adlara toleranslı) |
| **Çerçeveleme hatası** | "Riski %13.7 oranında artırıyor" | Etki payı bir risk miktarı değildir |
| **Korunan özellik gerekçesi** | "Kadın olmanız nedeniyle..." | Korunan terim + dışlama ipucu yokluğu |
| **Eksik tool çağrısı** | Varsayımsal soruya tool çağırmadan cevap | Soru kalıbı + çağrı kaydı |

Denetçinin kendisi de test edilir: `tests/test_faithfulness.py` her ihlal türü
için **kasıtlı olarak bozuk** bir anlatı üretip yakalandığını doğrular. Her şeye
"geçti" diyen bir denetçi işe yaramaz.

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

### 8. Kurulum tuzakları: `libomp` ve `llvmlite`

macOS'ta LightGBM `libomp` olmadan **import edilir ama eğitimde patlar**.
`test_lightgbm_openmp_works` gerçek bir eğitim koşturarak bunu yakalıyor.

Ayrıca `uv` ilk çözümlemede SHAP→numba→llvmlite zincirinde `llvmlite 0.36`'ya
düştü (Python 3.10 üstünü desteklemiyor). `pyproject.toml`'da `numba>=0.60` ve
`llvmlite>=0.43` alt sınırları bu yüzden açıkça belirtildi.

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
│   └── evaluate.py       # Faz 5 çalıştırıcısı
├── tests/                # 132 test (LLM'siz) + 6 canlı ajan testi
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

## Bilinen sınırlar

* **Veri seti küçük ve eski.** German Credit 1000 satır, 1994'ten ve Alman Markı
  cinsinden. Metrikler gerçek bir kredi portföyünü temsil etmez; hedef, boru
  hattını ve denetim mekanizmasını göstermek.
* **7B model yeniden yazım yapıyor.** Prompt "adı aynen kopyala" dese de
  qwen2.5:7b bazen "Mevcut işte çalışma süresi"ni "Mevcut İş Süresi" diye
  yazıyor. Denetçi bunu tolere edecek biçimde yazıldı (bulanık ad eşleştirme);
  daha büyük bir model bu sorunu azaltır.
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
