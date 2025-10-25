# shipbreaking-analysis
Data analysis of global ship scrapping (2014–2024), including data harmonization, trend analysis, and vessel lifecycle insights.


# Shipbreaking & Maritime Decommissioning Analysis (2014–2024)

This project analyzes global ship scrapping activity across multiple years to understand:
- Which vessel types are scrapped most frequently
- How vessel age and lightweight tonnage (LDT) relate to decommissioning decisions
- Geographic distribution of scrapping operations
- Market and lifecycle patterns in maritime decommissioning

---

## 🌍 Context & Motivation
Ships reaching the end of their operational lifespan are dismantled and recycled for steel, creating a large global industry centered in South Asia. Understanding scrapping trends provides insights into:
- Maritime fleet evolution
- Steel recycling economics
- Environmental and operational policies

---

## 🗂️ Data Sources
Multi-year shipbreaking datasets (2014–2024), harmonized into a consistent schema:
| Column | Description |
|-------|-------------|
| NAME | Vessel name |
| IMO | International Maritime Organization number |
| TYPE | Vessel class (cargo, tanker, passenger, etc.) |
| GT | Gross Tonnage |
| LDT | Lightweight Tonnage |
| BUILT | Year vessel was constructed |
| COUNTRY | Scrapping location (Country) |
| PLACE | Scrapping location(City) |
| YEAR | Scrapping YEAR |
| LAST FLAG | Country of Last Flag |

---

## 🛠️ Tools & Libraries
- Python (pandas, numpy, seaborn, matplotlib)
- Jupyter Notebook

---

## 🔄 Data Processing
- Standardized column names across datasets
- Removed duplicates and invalid IMO records
- Calculated vessel age at time of scrapping
- Combined yearly datasets into a single harmonized dataframe

---

## 📊 Key Insights
- **Tankers and bulk carriers** account for the highest scrapping volume.
- Majority of scrapping is concentrated in **Bangladesh, India, and Pakistan**.
- Typical vessels are decommissioned **25–35 years** after construction.
- Higher **LDT** correlates with later scrapping age, suggesting extended economic viability.
- Performed linear regression-based imputation to estimate missing Lightweight Tonnage (LDT) values using Gross Tonnage (GT), improving completeness of the dataset for downstream trend and lifecycle analysis.

---

## 📈 Example Visualizations
| Insight | Plot |
|--------|------|
| Vessel age distribution | `visuals/age_distribution.png` |
| Scrapping locations | `visuals/scrap_locations.png` |
| Vessel type share over time | `visuals/type_trends.png` |

---

## 📌 Project Status
Complete — available for review and extension.

---

## 🙋‍♀️ Author
**Nupur Srivastava**  
University at Buffalo — Data Science  
LinkedIn: linkedin.com/in/nupur-srivastava-ds  
