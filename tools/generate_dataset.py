import random
import csv
from pathlib import Path

OUTPUT = Path("data/dummy_dataset_100k.csv")
OUTPUT.parent.mkdir(exist_ok=True)

N = 100_000

real_templates = [
    "Türkiye Cumhuriyet Merkez Bankası faiz oranlarını açıkladı.",
    "Sağlık Bakanlığı yeni aşı kampanyasını duyurdu.",
    "Meteoroloji yarın için yağış uyarısı yaptı.",
    "Resmi açıklamaya göre ekonomi büyüme kaydetti.",
    "Üniversite araştırmasına göre yeni teknoloji geliştirildi.",
    "Bakanlık eğitim reformu planını paylaştı.",
    "Yerel yönetim yeni ulaşım projesini başlattı.",
    "Enerji üretiminde yenilenebilir kaynakların payı arttı.",
    "Bilim insanları yeni bir tedavi yöntemi keşfetti.",
    "Resmi verilere göre enflasyon oranı düştü.",
]

fake_templates = [
    "ŞOK! Gizli deneyde insanlar görünmez oldu!!!",
    "Bilim insanları aya inişin sahte olduğunu açıkladı!",
    "Hükümet gizli bir projeyle zihin kontrolü yapıyor iddiası!",
    "Uzaylılar Ankara üzerinde görüldü iddiası!!!",
    "Mucize bitki tüm hastalıkları 3 günde yok ediyor!",
    "Telefon sinyalleri beyni kontrol ediyor iddiası!",
    "Gizli belgeye göre dünya sona yaklaşıyor!",
    "Büyük şirket ölümsüzlük ilacı buldu ama saklıyor!",
    "Komplo teorisyenlerine göre gerçek tarih gizleniyor!",
    "İddialara göre zaman yolculuğu başarıldı!",
]

cities = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Van",
    "Bursa",
    "Antalya",
    "Konya",
    "Adana",
    "Trabzon",
    "Erzurum",
    "Kayseri",
]


def add_noise(text):
    # random punctuation / exaggeration
    if random.random() < 0.25:
        text += "!" * random.randint(1, 3)
    if random.random() < 0.15:
        text = text.upper()
    if random.random() < 0.15:
        text += " " + random.choice(
            ["iddia edildi", "söyleniyor", "gizli kaynaklara göre"]
        )
    if random.random() < 0.2:
        text += f" {random.choice(cities)}'da"
    return text


with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])

    for i in range(N):
        if random.random() < 0.5:
            text = random.choice(real_templates)
            label = 1  # REAL
        else:
            text = random.choice(fake_templates)
            label = 0  # FAKE

        text = add_noise(text)
        writer.writerow([text, label])

print(f"Dataset generated → {OUTPUT} ({N} rows)")
