import csv
import json

# Nazwy plików
input_file = 'pizza.csv'
output_file = 'pizza_converted.json'


def csv_to_json(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)  # Zakłada, że pierwszy wiersz zawiera nagłówki
        rows = list(reader)

    with open(output_file, mode='w', encoding='utf-8') as json_file:
        json.dump(rows, json_file, indent=4, ensure_ascii=False)  # Zapis z wcięciami dla czytelności

    print(f"Plik CSV został zapisany jako JSON w pliku: {output_file}")

# Wywołanie funkcji
csv_to_json(input_file, output_file)
