import os
import urllib.request
from datetime import datetime
import pandas as pd
import numpy as np

#
# ## 📚 БІБЛІОТЕКИ ТА КОНФІГУРАЦІЯ
#
# Ця секція визначає всі необхідні імпорти та словники для коректної роботи скрипту.
# 
# Словник відповідності ID NOAA та українських назв областей
# ПОРЯДОК: За ID NOAA (1-27)
NOAA_ID_TO_UKR_NAME = {
    1: "Черкаська", 2: "Чернігівська", 3: "Чернівецька", 4: "Крим", 
    5: "Дніпропетровська", 6: "Донецька", 7: "Івано-Франківська", 8: "Харківська", 
    9: "Херсонська", 10: "Хмельницька", 11: "Київська", 12: "Київ", 
    13: "Кіровоградська", 14: "Луганська", 15: "Львівська", 16: "Миколаївська", 
    17: "Одеська", 18: "Полтавська", 19: "Рівненська", 20: "Севастополь", 
    21: "Сумська", 22: "Тернопільська", 23: "Закарпатська", 24: "Вінницька", 
    25: "Волинська", 26: "Запорізька", 27: "Житомирська"
}

# СПИСОК ДЛЯ СОРТУВАННЯ ЗА УКРАЇНСЬКИМ АЛФАВІТОМ
UKR_ALPHABETICAL_ORDER = [
    "Вінницька", "Волинська", "Дніпропетровська", "Донецька", "Житомирська", 
    "Закарпатська", "Запорізька", "Івано-Франківська", "Київ", "Київська", 
    "Кіровоградська", "Крим", "Луганська", "Львівська", "Миколаївська", 
    "Одеська", "Полтавська", "Рівненська", "Севастополь", "Сумська", 
    "Тернопільська", "Харківська", "Херсонська", "Хмельницька", 
    "Черкаська", "Чернівецька", "Чернігівська"
]


#
# ## ⬇️ ЗАВАНТАЖЕННЯ ДАНИХ VHI
#
# Секція відповідає за ітеративне завантаження даних VHI з сервера NOAA для кожної області.
save_dir = "vhi_data"
os.makedirs(save_dir, exist_ok=True)

province_ids = range(1, 28)
df_all = pd.DataFrame()
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
DOWNLOAD_YEAR_START = 1981
DOWNLOAD_YEAR_END = 2024

print("--- Початок завантаження даних VHI ---")

for pid in province_ids:
    url = (
        f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?"
        f"country=UKR&provinceID={pid}&year1={DOWNLOAD_YEAR_START}&year2={DOWNLOAD_YEAR_END}&type=Mean"
    )
    province_name = NOAA_ID_TO_UKR_NAME.get(pid, f"ID_{pid}_UNKNOWN")
    filename = f"VHI_{pid}_{province_name}_{current_timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    print(f"Завантаження: ID {pid} ({province_name})...", end=" ")

    try:
        # ВИКОРИСТАННЯ urllib.request (з таймаутом та User-Agent)
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        # Перевірка на порожній файл
        if os.path.getsize(filepath) == 0:
            print("ПОМИЛКА: Файл порожній. Пропущено.")
            continue
            
        # Читання даних
        df = pd.read_csv(filepath, sep=r"\s+", comment="#", header=None, engine='python')

        if df.shape[1] >= 7:
            df = df.iloc[:, -7:]
            df.columns = ["year", "week", "SMN", "SMT", "VCI", "TCI", "VHI"]
            df = df[["year", "week", "VHI"]]
            df["provinceID"] = pid
            df["province_name"] = province_name
            df_all = pd.concat([df_all, df], ignore_index=True)
            print("Успішно оброблено.")
        else:
            print(f"ПОПЕРЕДЖЕННЯ: Файл не містить 7 необхідних колонок. Пропущено.")

    except urllib.error.HTTPError as e:
        print(f"ПОМИЛКА: HTTP-помилка: {e.code}. Пропущено.")
    except urllib.error.URLError as e:
        print(f"ПОМИЛКА: URL-помилка або таймаут: {e.reason}. Пропущено.")
    except pd.errors.ParserError:
        print(f"ПОМИЛКА: Помилка парсингу CSV. Пропущено.")
    except Exception as e:
        print(f"ПОМИЛКА: Невідома помилка: {e}. Пропущено.")

print("\n--- Збір даних завершено. ---")
print(f"Всього завантажено рядків: {len(df_all)}")


#
# ## 🧹 ОЧИЩЕННЯ ТА СОРТУВАННЯ
# 
# Ця секція виконує очищення (видалення NaN, приведення типів) та сортування за алфавітом.
if df_all.empty:
    print("\nНЕМАЄ ДАНИХ. Фінальний датасет порожній. Неможливо виконати очищення та сортування.")
    exit() 

df_all["VHI"] = pd.to_numeric(df_all["VHI"], errors="coerce")
df_all.dropna(subset=["VHI"], inplace=True)
df_all["year"] = df_all["year"].astype(int)
df_all["week"] = df_all["week"].astype(int)

# СОРТУВАННЯ ЗА УКРАЇНСЬКИМ АЛФАВІТОМ
df_all['province_name'] = pd.Categorical(
    df_all['province_name'], 
    categories=UKR_ALPHABETICAL_ORDER, 
    ordered=True
)

df_all.sort_values(by=['province_name', 'year', 'week'], inplace=True)

df_all.reset_index(drop=True, inplace=True)

print("\n--- Очищення та сортування за алфавітом завершено. ---")


#
# ## 🔬 ФУНКЦІЇ АНАЛІТИКИ
# 
# Набір функцій для отримання конкретних вибірок (по року, по середньому, екстремуми).
def get_vhi_by_year(province: str, year: int) -> pd.DataFrame:
    """Ряд VHI для області (за українською назвою) за вказаний рік."""
    subset = df_all[(df_all["province_name"] == province) & (df_all["year"] == year)]
    print(f"\n--- VHI для {province} у {year} році ---")
    return subset[["year", "week", "VHI"]]

def get_vhi_mean_by_years(provinces: list, start_year: int, end_year: int) -> pd.DataFrame:
    """Середній VHI за діапазон років для вказаних областей."""
    subset = df_all[
        (df_all["province_name"].isin(provinces)) &
        (df_all["year"].between(start_year, end_year))
    ]
    print(f"\n--- Середній VHI для областей {provinces} з {start_year} по {end_year} ---")
    return subset.groupby(["province_name", "year"])["VHI"].mean().reset_index()

def get_extremes(provinces: list, years: list) -> pd.DataFrame:
    """Пошук min, max, середнього, медіани VHI для областей і років."""
    subset = df_all[(df_all["province_name"].isin(provinces)) & (df_all["year"].isin(years))]
    result = subset.groupby("province_name")["VHI"].agg(["min", "max", "mean", "median"]).reset_index()
    print(f"\n--- Екстремуми VHI для {provinces} у {years} ---")
    return result


#
# ## 💾 ЗБЕРЕЖЕННЯ ДАНИХ
# 
# Збереження фінального, відсортованого датасету в CSV-файл.
output_filename = f"VHI_all_UKR_sorted_{current_timestamp}.csv"
output_path = os.path.join(save_dir, output_filename)

df_final = df_all.drop(columns=["provinceID"]) 

df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nФінальний файл збережено: {output_path}")
print(f"Всього рядків (після очищення): {len(df_final)}")
print("\nПерші 5 рядків фінального датасету (повинно починатися з Вінницької):")
print(df_final.head())