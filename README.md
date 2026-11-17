# 🚂 Dynamic Train ETA Prediction

> **Historical Demonstration** — replays recorded journeys from September 2024 Indian Railway data.  
> This is a **college micro-project prototype**, not a live tracking service.

## 📂 Project Structure

```
Dynamic/
├── app.py                  # Streamlit interface & replay controls
├── data_loader.py          # Load, clean, and merge data files
├── model.py                # Baselines + Random Forest prediction
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/
    ├── train_routes_delays_Sep2024.csv   # Main delay dataset
    ├── train_routes_Sep2024.csv          # Route, schedule, distance
    ├── IRN_edges.csv                     # Station network graph
    ├── stations_zones_mapping.json       # Station → zone mapping
    └── train_delays_Sep2024.json         # Alternate delay format
```

## 🚀 Setup & Run

### Prerequisites
- Python 3.9 or later

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/kr-javas/Dynamic-.git
cd Dynamic-

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🎮 How to Use

1. **Select a train** from the sidebar dropdown (e.g., 12303 — Poorva Express).
2. **Pick a journey date** from September 2024.
3. Click **▶️ Next Station** to reveal stations one at a time.
4. Watch the **ETA predictions** update at each step for upcoming stations.
5. Click **⏭️ Show All** to reveal the full journey and see the accuracy comparison.

## 🔬 How It Works

### Prediction Models

| Model | Strategy |
|---|---|
| **Timetable** (baseline) | Always predicts zero delay — the train is on time. |
| **Persistence** (baseline) | The latest observed delay continues unchanged. |
| **Random Forest** (ML) | Uses station sequence, distance, scheduled time, and past delays to predict future delays. |

### Key Design Decisions

- **No data leakage**: The model never sees a future station's actual delay during prediction.
- **Journey-based splitting**: Train/test split is by journey (train+date), not random rows.
- **Midnight handling**: Times are converted to minutes-since-midnight for safe arithmetic.
- **Graceful fallback**: If the model can't train, baselines still work and a warning is shown.

## 📊 Evaluation

After revealing all stations, the app shows:
- **MAE (Mean Absolute Error)** in minutes for each method.
- A bar chart comparing the three approaches.

## 🛠️ Tech Stack

- **Python** — core language
- **Streamlit** — interactive web interface
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — Random Forest model

## 👨‍🎓 Authors

College Micro-Project — Dynamic Train ETA Prediction  
Data: Indian Railways, September 2024
