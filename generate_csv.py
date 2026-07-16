import os
import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "../data/learnkins_students_enriched.csv")
DST = os.path.join(os.path.dirname(__file__), "data/learnkins_ai_analytics.csv")

TOPIC_COLUMNS = [
    "score_html_css",
    "score_javascript",
    "score_react",
    "score_nodejs",
    "score_express",
    "score_mongodb",
]

ACTIVE_COLUMNS = [f"w{i:02d}_active_days" for i in range(1, 19)]
XP_COLUMNS = [f"xp_w{i:02d}" for i in range(1, 19)]

BADGE_RANK = {
    "Beginner": 0,
    "Explorer": 1,
    "Coder": 2,
    "Builder": 3,
    "Innovator": 4,
    "Champion": 5,
}


def create_risk_label(score):
    if score >= 75:
        return "Low Risk"
    elif score >= 55:
        return "Medium Risk"
    else:
        return "High Risk"


def engagement_score(row):
    streak_norm = min(row["streak_days"] / 90, 1) * 25
    xp_norm = min(row["xp_points"] / 5000, 1) * 25
    quiz_norm = min(row["quizzes_attempted"] / 50, 1) * 25
    activity_norm = min(row["total_study_hours"] / 100, 1) * 25
    return round(streak_norm + xp_norm + quiz_norm + activity_norm, 2)


def ai_recommended_focus(row):
    topics = {col: row[col] for col in TOPIC_COLUMNS}
    weakest_col = min(topics, key=topics.get)
    label_map = {
        "score_html_css": "HTML & CSS",
        "score_javascript": "JavaScript",
        "score_react": "React",
        "score_nodejs": "Node.js",
        "score_express": "Express",
        "score_mongodb": "MongoDB",
    }
    return label_map[weakest_col]


def learning_velocity(row):
    active = max(row[ACTIVE_COLUMNS].sum(), 1)
    return round(row["xp_points"] / active, 2)


def consistency_score(row):
    weeks = row[ACTIVE_COLUMNS].values
    if weeks.max() == 0:
        return 0.0
    cv = np.std(weeks) / (np.mean(weeks) + 1e-9)
    score = max(0, 100 - cv * 50)
    return round(score, 2)


def predicted_next_xp(row):
    xp_series = row[XP_COLUMNS].values.astype(float)
    last4 = xp_series[-4:]
    slope = (last4[-1] - last4[0]) / 3 if len(last4) >= 2 else 0
    return round(float(xp_series[-1]) + slope * 4, 2)


def main():
    print(f"Reading source CSV: {SRC}")
    df = pd.read_csv(SRC)

    for col in TOPIC_COLUMNS + ACTIVE_COLUMNS + XP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")
    df["last_active"] = pd.to_datetime(df["last_active"], errors="coerce")
    df["days_enrolled"] = (df["last_active"] - df["join_date"]).dt.days.clip(lower=1).fillna(1)
    df["badge_rank"] = df["badge"].map(BADGE_RANK).fillna(0).astype(int)

    df["topic_avg"] = df[TOPIC_COLUMNS].mean(axis=1).round(2)
    df["final_performance"] = (0.6 * df["quiz_avg_score"] + 0.4 * df["topic_avg"]).round(2)
    df["risk_level"] = df["final_performance"].apply(create_risk_label)

    df["total_active_days"] = df[ACTIVE_COLUMNS].sum(axis=1)
    df["recent_active_days"] = df[ACTIVE_COLUMNS[-4:]].sum(axis=1)
    df["activity_consistency"] = df[ACTIVE_COLUMNS].mean(axis=1).round(3)
    df["xp_growth"] = df["xp_w18"] - df["xp_w01"]
    df["recent_xp_growth"] = df["xp_w18"] - df["xp_w14"]

    df["engagement_score"] = df.apply(engagement_score, axis=1)
    df["learning_velocity"] = df.apply(learning_velocity, axis=1)
    df["consistency_score"] = df.apply(consistency_score, axis=1)
    df["predicted_next_xp"] = df.apply(predicted_next_xp, axis=1)
    df["ai_recommended_focus"] = df.apply(ai_recommended_focus, axis=1)

    def improvement_tip(row):
        tips = []
        if row["quiz_avg_score"] < 60:
            tips.append("Revise quiz topics daily.")
        if row["streak_days"] < 7:
            tips.append("Try to study at least 15 min every day to build a streak.")
        if row["recent_active_days"] < 10:
            tips.append("Increase weekly activity — aim for 5 active days per week.")
        if row["topic_avg"] < 60:
            tips.append(f"Focus on {row['ai_recommended_focus']} to boost topic scores.")
        if not tips:
            tips.append("Great work! Keep maintaining your current study habits.")
        return " | ".join(tips)

    df["ai_improvement_tip"] = df.apply(improvement_tip, axis=1)

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"[OK] Saved enriched CSV to: {DST}")
    print(f"   Shape: {df.shape}   New columns: engagement_score, learning_velocity, consistency_score, predicted_next_xp, ai_recommended_focus, ai_improvement_tip")


if __name__ == "__main__":
    main()
