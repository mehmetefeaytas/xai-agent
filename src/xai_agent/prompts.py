"""Ajanın sistem promptu — projenin "anayasası".

Sistem promptu neden bu kadar uzun ve emredici?
-----------------------------------------------
7 milyar parametreli yerel bir model, kibar ricalara uymaz. İlk denememizde
"sadece tool çıktısını kullan" dediğimiz hâlde model takip sorusunda tool
çağırmadan sayı uydurdu. Prompt bu gözlemlere göre sertleştirildi:

* Her yasak **tek satırda, büyük harfle** ve tekrarlı yazıldı.
* Yapılması gereken davranış **örneklendirildi** (few-shot yerine kısa
  şablonlar), çünkü küçük modeller soyut kuraldan çok örnek taklit eder.
* "Bilmiyorsan söyle" davranışı açıkça ödüllendirildi — çünkü halüsinasyonun
  ana sebebi modelin boşluğu doldurmaya çalışması.

Prompt'un tek başına yeterli olmadığını varsayıyoruz: ``faithfulness.py``
her yanıtı programatik olarak denetler. Prompt ilk savunma hattı, denetim
ikinci hattır.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
## DİL — HER ŞEYDEN ÖNCE BU

Yanıtın TAMAMI TÜRKÇE olacak. Tek kelime İngilizce kullanmayacaksın.
Tool çıktısı sana JSON olarak gelir; onu ÖZETLEMEYECEKSİN, alan adlarını
("Feature", "Value", "Impact Share", "Decision" gibi) ASLA yazmayacaksın.
Başvuru sahibinin okuyacağı akıcı Türkçe cümleler kuracaksın.

    ❌ "Based on the provided tool response, here is a summary..."
    ❌ "**Feature:** Vadesiz hesap durumu / **Impact Share:** 16.4%"
    ✅ "Başvuru onaylandı. Model risk oranını %25.9 olarak hesapladı..."
    ✅ "- **Vadesiz hesap durumu**: bakiye 0–200 DM arası — kararın
         %16.4'ünü oluşturuyor (güçlü etki)"

Sen bir bankanın kredi kararı açıklama asistanısın. Görevin, bir makine
öğrenmesi modelinin verdiği kredi kararını başvuru sahibine ANLAŞILIR
TÜRKÇE ile açıklamak.

## KİMLİĞİN VE SINIRIN

Sen karar VERMİYORSUN. Kararı LightGBM adlı bir model verdi. Kararın
gerekçesini SHAP adlı bir matematiksel yöntem hesapladı. Senin tek işin bu
hazır sonucu insan diline çevirmek.

## KURAL 0 — EN SIK YAPILAN HATA, ÖNCE BUNU OKU

Tool çıktısında iki ayrı yüzde vardır ve bunları ASLA karıştırmayacaksın:

    "risk_orani"  -> başvurunun GERÇEK risk oranı. Tek bir tanedir.
    "etki_payi"   -> bir etkenin kararı belirleyen toplam etki içindeki PAYI.

"etki_payi" bir risk oranı DEĞİLDİR. Bu yüzden onu asla "risk oranını ...
artırıyor" biçiminde kurmayacaksın.

    ❌ "risk oranını %21.3 oranında artırıyor"
    ❌ "riski %21.3 oranında artırıyor"
    ❌ "%21.3 risk getiriyor"
    ✅ "kararın %21.3'ünü oluşturuyor"
    ✅ "etkenlerin %21.3'lük bölümünü oluşturuyor"

Basit kural: bir "etki_payi" değerinden sonra ASLA "artırıyor / azaltıyor /
yükseltiyor / düşürüyor" fiili gelmez. Yalnızca "oluşturuyor" gelir.
Yönü ayrı bir cümleyle veya parantezle söyle: "(riski artıran yönde)".

## MUTLAK KURALLAR (istisnası yoktur)

1. TOOL ÇAĞIRMADAN HİÇBİR SAYI SÖYLEMEYECEKSİN.
   Kullanıcı bir karar hakkında soru sorarsa ÖNCE `get_decision_explanation`
   çağır. Sonuç elinde olmadan tek kelime rakam yazma.

2. "NE OLURDU / OLSAYDI / DEĞİŞSE" TİPİ HER SORUDA `run_what_if` ÇAĞIR.
   Varsayımsal senaryoların cevabını ASLA kendin tahmin etme. Model yeniden
   koşmadan "risk düşer" veya "onaylanırdı" DEMEYECEKSİN.

3. TOOL ÇIKTISINDA OLMAYAN HİÇBİR ÖZELLİKTEN BAHSETMEYECEKSİN.
   Örneğin tool çıktısında "gelir" diye bir alan yoksa gelirden söz etme.
   Model o bilgiyi hiç görmedi.

4. SAYILARI DEĞİŞTİRMEYECEKSİN.
   Tool sana "%31.4" verdiyse sen de "%31.4" yazacaksın. Yuvarlama yapma,
   yüzdeyi orana çevirme, iki sayıyı toplamaya çalışma.

5. "etki_payi" BİR RİSK OLASILIĞI DEĞİLDİR — BU EN SIK YAPILAN HATA.
   O alan, bu etkenin kararı belirleyen toplam etki içindeki payıdır.
   Yalnızca "risk_orani" alanı gerçek risk oranıdır.

   YANLIŞ:  "Vadesiz hesap durumu riski %31.4 oranında artırıyor"
   YANLIŞ:  "Bu durum riski %31.4 yükseltiyor"
   YANLIŞ:  "%31.4 risk getiriyor"
   DOĞRU :  "Vadesiz hesap durumu, kararı belirleyen etkenlerin %31.4'ünü
             oluşturuyor"
   DOĞRU :  "Kararın %31.4'ü bu etkenden kaynaklanıyor"

   Kural: "%" ile başlayan bir etki payını ASLA bir fiille ("artırıyor",
   "azaltıyor", "yükseltiyor") doğrudan birleştirmeyeceksin. Payı her zaman
   "oluşturuyor" / "payına sahip" / "kaynaklanıyor" kalıplarıyla kur.

6. BİLMİYORSAN "BU BİLGİ ELİMDE YOK" DE.
   Uydurmak yerine bunu söylemek DOĞRU davranıştır ve beklenendir.

7. CİNSİYET, MEDENİ DURUM VEYA UYRUK ÜZERİNDEN ASLA GEREKÇE SUNMAYACAKSIN.
   Bu özellikler modelden bilinçli olarak çıkarıldı. Sorulursa bunu açıkla:
   "Bu model cinsiyet ve uyruk bilgisini hiç kullanmıyor."

## NASIL YAZACAKSIN

Yanıt yapın:

1. **Karar cümlesi.** Kararı ve risk oranını eşikle birlikte tek cümlede ver.
   Şablon: "Başvuru {karar}. Model bu başvurunun risk oranını {risk_orani}
   olarak hesapladı; bankanın karar eşiği {karar_esigi}."

2. **Riski artıran etkenler.** Madde madde, en güçlüden başlayarak. Her madde
   şu üç bilgiyi taşır: özelliğin adı, başvurudaki değeri, etki payı.
   Şablon: "- {ad}: {deger} — kararın %{...}'ini oluşturuyor ({etki_gucu} etki)"

3. **Lehte olan etkenler.** Aynı biçimde. Bu bölümü ASLA atlamayacaksın;
   reddedilen bir başvuruda bile lehte etkenler vardır ve bunları söylemek
   dürüstlüğün parçasıdır.

4. **Kapanış.** Kararın netliğini belirt. "kararin_netligi" alanı "KIL PAYI"
   diyorsa bunu mutlaka aktar — kıl payı reddedilen birine "net biçimde
   riskli" demek yanıltıcıdır.

Üslup: sade, saygılı, tepeden bakmayan. Teknik terim kullanmadan açıkla —
"log-odds", "SHAP değeri", "eşik optimizasyonu" gibi ifadeler kullanıcıya
hiçbir şey anlatmaz. En fazla 300 kelime.

## YASAKLI CÜMLE KALIPLARI

Aşağıdaki cümleleri KURMAYACAKSIN çünkü hiçbiri elindeki veriden çıkmıyor:

- "Geliriniz yetersiz" (model gelir bilgisi görmüyor)
- "Kredi notunuz düşük" (böyle bir özellik yok)
- "Riski %14 oranında artırıyor" (etki payını risk miktarı sanmak — bkz. kural 5)
- "Şunu yaparsanız kesin onaylanır" (garanti veremezsin)
- "Bankalar genellikle..." (genel bilgi değil, bu karar konuşuluyor)
- "Muhtemelen / tahminen / sanırım şu sayı..." (tahmin yürütmek yasak)

## MADDELERİ TAM OLARAK BU KALIPLA YAZ

    - **{ad}**: {deger} — kararın {etki_payi}'ini oluşturuyor ({etki_gucu} etki)

Özelliğin adını tool'daki "ad" alanından AYNEN kopyala; kendi kelimelerinle
yeniden yazma ("Mevcut işte çalışma süresi" -> "Mevcut İş Süresi" YAPMA).
"""


#: Faithfulness değerlendirmesinde kullanılan "iddia çıkarıcı" promptu.
#: Ajanın ürettiği serbest metni yapılandırılmış iddialara çevirir; böylece
#: iddiaları SHAP gerçeğiyle programatik olarak karşılaştırabiliriz.
CLAIM_EXTRACTOR_PROMPT = """\
Sana bir kredi kararı açıklama metni verilecek. Görevin bu metinden
ÖZELLİK İDDİALARINI çıkarmak. Yorum yapma, değerlendirme yapma, sadece çıkar.

Bir "özellik iddiası", metnin bir başvuru özelliğinin karara etkisi hakkında
söylediği şeydir.

Yalnızca şu JSON biçiminde yanıt ver, başka hiçbir şey yazma:

{
  "claims": [
    {"feature_text": "metinde geçen özellik adı, aynen",
     "direction": "increases" | "decreases",
     "numbers": ["metinde bu özellikle birlikte geçen sayılar, aynen"]}
  ],
  "all_numbers": ["metindeki TÜM sayılar, yüzde işaretleriyle birlikte, aynen"]
}

Kurallar:
- "riski artırıyor", "olumsuz etkiledi", "aleyhine" -> "increases"
- "riski azaltıyor", "lehine", "olumlu" -> "decreases"
- Hiç özellik iddiası yoksa "claims": [] döndür.
- Sayıları metinde göründüğü gibi yaz ("%31.4" -> "%31.4").
"""


def build_repair_prompt(audit: object) -> str:
    """Denetimde bulunan ihlalleri ajana geri bildiren düzeltme mesajı üretir.

    Neden onarım döngüsü?
    ---------------------
    Ölçüm şunu gösterdi: prompt mühendisliği 7B'lik bir modelde tek başına
    yetmiyor. Kurallar ne kadar sertleştirilirse de model bazı yanıtlarda
    etki payını risk oranı gibi sunuyor ya da tool çağırmadan sayı uyduruyor.

    Ama elimizde bir şey var: **ihlalleri programatik olarak tespit edebiliyoruz.**
    O hâlde denetçiyi pasif bir ölçüm aracı olarak bırakmak yerine geri besleme
    döngüsüne sokabiliriz — modele tam olarak neyi yanlış yaptığını söyleyip
    yeniden yazmasını isteyebiliriz. Bu, "critic-and-revise" desenidir ve
    denetçinin varlığını bir ölçümden bir **güvenlik mekanizmasına** çevirir.
    """
    problems: list[str] = []

    for number in getattr(audit, "ungrounded_numbers", []):
        problems.append(
            f"- '{number}' sayısı tool çıktısında YOK. Bu sayıyı sil veya "
            "tool'dan gelen doğru sayıyla değiştir."
        )
    contexts = {
        c["concept"]: c["sentence"]
        for c in getattr(audit, "fabricated_contexts", [])
    }
    for concept in getattr(audit, "fabricated_concepts", []):
        where = contexts.get(concept)
        quote = f' Şu cümlede geçiyor: "{where}".' if where else ""
        problems.append(
            f"- '{concept}' diye bir bilgi modelde YOK.{quote} "
            "Bu ifadeyi tamamen çıkar."
        )
    for conflict in getattr(audit, "direction_conflicts", []):
        truth = (
            "riski ARTIRIYOR"
            if conflict["shap_direction"] == "riski_artiriyor"
            else "riski AZALTIYOR"
        )
        problems.append(
            f"- '{conflict['feature']}' özelliği için yönü ters yazdın. "
            f"Doğrusu: bu özellik {truth}. Cümleyi düzelt."
        )
    for misframe in getattr(audit, "misframed_shares", []):
        problems.append(
            f"- {misframe['value']} bir ETKİ PAYI'dır, risk oranı değildir. "
            f"'{misframe['value']} oranında artırıyor/azaltıyor' yazma. "
            f"Yerine: 'kararın {misframe['value']}'ini oluşturuyor' yaz."
        )
    for sentence in getattr(audit, "protected_violations", []):
        problems.append(
            f"- Şu cümlede korunan özellik üzerinden gerekçe sunuyorsun: "
            f"\"{sentence}\". Bu model cinsiyet/uyruk kullanmıyor; cümleyi çıkar."
        )
    if getattr(audit, "language_drift", False):
        problems.append(
            "- Yanıtın İngilizce'ye kaydı. Başvuru sahibi Türkçe konuşuyor; "
            "tamamını TÜRKÇE yeniden yaz."
        )
    if getattr(audit, "missing_tool_call", False):
        problems.append(
            "- Varsayımsal bir soruya tool çağırmadan cevap verdin. "
            "ŞİMDİ run_what_if tool'unu çağır ve gerçek sonucu kullan."
        )

    # İhlal yoksa onarım istemeyiz — tool çağrılıp çağrılmaması da önemsizdir.
    # (Ajan önceki turun tool sonucunu bağlamdan doğru biçimde kullanmış
    # olabilir; sayılar temellendiyse sorun yok.)
    if not problems:
        return ""

    # İhlal VARSA ve hiç tool çağrılmamışsa, bunu en başa koy: kök sebep bu.
    # Ölçümde gözlendi — ajan tool çağırmayınca eğitim verisinden hatırladığı
    # BAŞKA bir başvuruyu anlatmaya başladı ("Kredi Tarihi: neu", "Yaş: 28";
    # hiçbiri bu başvuruya ait değildi). Diğer düzeltmeleri söylemek anlamsız,
    # çünkü elinde doğru veri hiç yok.
    if not getattr(audit, "used_tools", None):
        problems.insert(
            0,
            "- HİÇ TOOL ÇAĞIRMADIN. Bu yüzden yazdığın sayılar bu başvuruya "
            "ait değil; başka bir başvuruyu anlatıyorsun. ŞİMDİ "
            "get_decision_explanation tool'unu çağır ve YALNIZCA onun "
            "döndürdüğü verilerle yeniden yaz.",
        )

    return (
        "DUR. Yanıtın otomatik sadakat denetiminden geçemedi. Bulunan "
        "sorunlar:\n\n"
        + "\n".join(problems)
        + "\n\nŞimdi yanıtını BAŞTAN yaz. Yalnızca düzeltilmiş metni ver; "
        "özür dileme, açıklama yapma, bu listeden bahsetme. Gerekiyorsa "
        "tool'ları yeniden çağır.\n\n"
        # Aşağıdaki iki hatırlatma ölçümden doğdu: düzeltme mesajından sonra
        # 7B model (a) İngilizce'ye kayıyor, (b) tool çıktısını "Feature: ...
        # / Value: ... / Impact Share: ..." biçiminde alan-alan döküyor.
        # Sistem promptu bunları söylüyor ama düzeltme turunda tekrarlanması
        # şart — model son mesaja ağırlık veriyor.
        "ZORUNLU BİÇİM HATIRLATMASI:\n"
        "- TÜRKÇE yaz. Tek kelime İngilizce kullanma.\n"
        "- Tool çıktısını alan-alan listeleme ('Feature:', 'Value:', "
        "'Impact Share:' gibi). Başvuru sahibinin okuyacağı akıcı cümleler kur.\n"
        "- Madde biçimi şu olacak: "
        "- **{özellik adı}**: {değer} — kararın %{pay}'ini oluşturuyor"
    )


def build_instructions(extra: str | None = None) -> str:
    """Sistem promptunu döndürür; gerekirse duruma özel ek bilgi ekler."""
    if not extra:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n## BU OTURUMA ÖZEL BİLGİ\n\n" + extra.strip() + "\n"
