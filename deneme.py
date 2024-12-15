import os
import numpy as np
import pandas as pd

# Histogram verilerinin olduğu klasör
histogram_folder = "/Users/yagmu/PycharmProjects/Capstone/hma_kinesics/histograms"

# Histogram dosyalarını oku  BURASI ÖNEMLİİİ!!!!!!!!
histogram_files = sorted([f for f in os.listdir(histogram_folder) if f.endswith("histograms_columns.csv")], key=lambda x: int(x.split('.')[0]))

# Birleşmiş histogramları tutacak liste
combined_histograms = []

# Her dosyadaki histogramları oku ve birleştir
for file in histogram_files:
    file_path = os.path.join(histogram_folder, file)
    with open(file_path, "r") as f:
        histogram_data = f.read().strip().split("\n")  # Her bir satırı oku
        histogram_data = list(map(float, histogram_data))  # Float türüne çevir
        combined_histograms.append(histogram_data)

# NumPy dizisine çevir ve yan yana birleştir
x_matrix = np.array(combined_histograms)

# Sonuçları bir CSV dosyasına yaz
output_csv = "combined_histograms.csv"
pd.DataFrame(x_matrix).to_csv(output_csv, index=False, header=False)

print(f"Histogramlar birleştirildi ve {output_csv} dosyasına kaydedildi.")
