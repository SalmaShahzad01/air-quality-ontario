# Air Quality Ontario — Toronto North (2021–2024)

## 📘 Project Overview
This project analyzes **air quality trends and extreme pollution events** in Toronto North using hourly government-issued air quality data from **Air Quality Ontario**.  
The study covers the period **January 2021 – December 2024** and includes four key pollutants:

- **Sulphur Dioxide (SO₂)**
- **Nitrogen Dioxide (NO₂)**
- **Ozone (O₃)**
- **Particulate Matter (PM₂.₅)**

The dataset is publicly available from [Air Quality Ontario](https://www.ontario.ca/environment-and-energy/air-quality) and contains hourly pollutant concentrations in ppb/µg m⁻³.

---

## 🧹 Data Preparation
Each pollutant file (e.g., `Sulphate_2021_2024.csv`, `Nitrogen_2021_2024.csv`, etc.) was cleaned and processed as follows:

1. **Metadata rows removed**  
2. **Missing values** (`9999`, `-999`, `-9999`) replaced with `NaN`  
3. **Days retained only if ≥ 18 valid hourly readings**  
4. Hourly data converted to **daily means**
5. Pollutants merged into a single daily dataset

---

## 📈 First Figure — Daily Pollutant Trends (Mean)
The first output figure shows daily mean concentrations of SO₂, NO₂, O₃, and PM₂.₅ from 2021–2024.

**Interpretation:**  
Ozone (green) exhibits strong seasonal cycles, peaking during summer.  
Nitrogen and particulate levels increase in colder months, reflecting combustion-related emissions.  
Sharp peaks (notably in 2023) indicate episodic pollution events suitable for extreme-value analysis.

---

## 🧮 Planned Analysis
- **Daily Pollution Burden Index (DPBI):**  
  Rolling-z-score average of the four pollutants (90-day window) to represent overall pollution pressure.  
- **Extreme-Value Detection:**  
  Identify days above the 95th percentile DPBI; perform sensitivity analysis at 90th and 97.5th percentiles.  
- **Trend Analysis:**  
  Examine changes in DPBI and pollutants using linear regression and Kendall’s τ tests.

---

## ⚙️ Reproducibility

### Step-by-step setup
```bash
# 1. Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw files in data_raw/
#    Sulphate_2021_2024.csv
#    Nitrogen_2021_2024.csv
#    Ozone_2021_2024.csv
#    PM2.5_2021_2024.csv

# 4. Run the cleaning and plotting script
python scripts/prepare_first_plot.py

### Running the automated tests
```bash
source .venv/bin/activate  # if not already active
pip install -r requirements.txt
pytest
```
