import csv
import random


def generate_vk_id():
    return random.randint(100000000, 9999999999)


with open('negative_users_10000.csv', 'w', newline='', encoding='utf-8-sig') as f:  # ← utf-8-sig!
    fieldnames = ['id', 'sex', 'age', 'city', 'education_level', 'university',
                  'main_in_life', 'main_in_people', 'smoking', 'alcohol',
                  'political']

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    used_ids = set()

    for _ in range(10000):
        while True:
            vk_id = generate_vk_id()
            if vk_id not in used_ids:
                used_ids.add(vk_id)
                break

        row = {
            'id': vk_id,
            'sex': random.choice(['ж', 'м']),
            'age': random.randint(18, 65),
            'city': random.choice(['Москва', 'СПб', 'Екатеринбург', 'Уфа', 'Казань']),
            'education_level': 'высшее',
            'university': random.choice(['СПбГУ', 'ИТМО', 'УрФУ', 'МГУ']),
            'main_in_life': random.choice(['карьера', 'деньги', 'власть']),
            'main_in_people': random.choice(['деньги', 'статус', 'влияние']),
            'smoking': 'положительное',
            'alcohol': 'положительное',
            'political': random.choice(['радикальные', 'экстремистские']),
        }
        writer.writerow(row)

print(" Готово! UTF-8-sig файл для Streamlit!")
