from src.predict import predict_text

tests = [
    # Likely REAL
    "Sağlık Bakanlığı yeni aşı kampanyasını duyurdu.",
    "Meteoroloji yarın için yağış uyarısı yaptı.",
    "Resmi verilere göre enflasyon oranı düştü.",
    "Üniversite araştırmasına göre yeni teknoloji geliştirildi.",
    # Likely FAKE
    "ŞOK! Gizli deneyde insanlar görünmez oldu!!!",
    "Uzaylılar Ankara üzerinde görüldü iddiası!!!",
]

print("\nRunning prediction tests...\n")

for i, text in enumerate(tests, 1):
    print("\n" * 4 + "=" * 20 + f" TEST {i} " + "=" * 20)
    print()
    print(f"Text: {text}")
    result = predict_text(text)
    print(result)
    print("=" * 50)
