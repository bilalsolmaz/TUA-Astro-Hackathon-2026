from damage_model import compute_damage_score, data_source_info

# Veri kaynağı kontrolü
info = data_source_info()
print(f"{info['icon']} Kaynak: {info['source']}")
print(f"Toplam nokta: {info['points']}")
print()

# Episantr yakını — yüksek hasar bekliyoruz
s1 = compute_damage_score(37.288, 37.043, "residential")
print(f"Pazarcık episantr (residential): {s1:.3f}")

# KMaraş merkez — orta hasar
s2 = compute_damage_score(37.585, 36.938, "primary")
print(f"KMaraş merkez (primary): {s2:.3f}")

# Uzak nokta — düşük hasar
s3 = compute_damage_score(37.750, 36.500, "motorway")
print(f"Uzak bölge (motorway): {s3:.3f}")

