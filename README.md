# AI Performance Analytics Dashboard

This folder contains the AI Performance Analytics Dashboard for LearnKins.

![Dashboard Preview]

### Project Demo
<video src="assets/20260716-1228-28.0949516.mp4" width="100%" controls></video>

## Features
- **Learning Heatmaps**: Visualizes student week-over-week active days.
- **Topic Mastery Graphs**: Horizontal bar chart tracking strengths and weaknesses per subject.
- **Growth Tracking**: Weekly XP accumulation trend line with AI-driven future prediction.
- **Study Insights**: AI-generated action points and feedback written in simple parent-friendly language.
- **Class Analytics**: Cohort-level statistics, risk distributions, and grade performance.

## File Structure
- `app.py`: Streamlit dashboard application.
- `generate_csv.py`: Enriches the student database with AI metrics.
- `train_model.py`: Trains the student risk classification model.
- `utils/`: Core layout helpers.
  - `preprocessing.py`: Data conversion and metrics calculation.
  - `charts.py`: Graph configurations.
  - `insights.py`: AI feedback logic.

## How to Run

1. Generate the enriched dataset:
   ```bash
   python generate_csv.py
   ```

2. Train the AI model:
   ```bash
   python train_model.py
   ```

3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
