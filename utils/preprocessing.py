import pandas as pd
import numpy as np

TOPIC_COLUMNS = [
    "score_html_css", "score_javascript", "score_react",
    "score_nodejs", "score_express", "score_mongodb",
]

TOPIC_LABELS = {
    "score_html_css":   "HTML & CSS",
    "score_javascript": "JavaScript",
    "score_react":      "React",
    "score_nodejs":     "Node.js",
    "score_express":    "Express",
    "score_mongodb":    "MongoDB",
}

ACTIVE_COLUMNS = [f"w{i:02d}_active_days" for i in range(1, 19)]
XP_COLUMNS     = [f"xp_w{i:02d}" for i in range(1, 19)]

BADGE_RANK = {
    "Beginner": 0, "Explorer": 1, "Coder": 2,
    "Builder": 3,  "Innovator": 4, "Champion": 5,
}

MODEL_FEATURES = [
    "xp_points", "lessons_completed", "streak_days", "quizzes_attempted",
    "age", "island_level", "days_enrolled", "badge_rank",
    "avg_daily_study_minutes", "total_study_hours", "longest_streak_ever",
    "avg_quiz_attempts_per_lesson", "total_active_days", "recent_active_days",
    "xp_growth", "recent_xp_growth", "activity_consistency",
    "quiz_avg_score", "topic_avg", "engagement_score",
    "learning_velocity", "consistency_score",
] + TOPIC_COLUMNS


def get_risk_label(score):
    if score >= 75:
        return "Low Risk"
    elif score >= 55:
        return "Medium Risk"
    else:
        return "High Risk"


def load_data(path):
    df = pd.read_csv(path)
    return prepare_data(df)


def prepare_data(df):
    df = df.copy()

    df["join_date"]   = pd.to_datetime(df["join_date"],   errors="coerce")
    df["last_active"] = pd.to_datetime(df["last_active"], errors="coerce")
    df["days_enrolled"] = (df["last_active"] - df["join_date"]).dt.days.clip(lower=1).fillna(1)
    df["badge_rank"]    = df["badge"].map(BADGE_RANK).fillna(0).astype(int)

    for col in TOPIC_COLUMNS + ACTIVE_COLUMNS + XP_COLUMNS + [
        "xp_points", "lessons_completed", "streak_days", "quizzes_attempted",
        "quiz_avg_score", "avg_daily_study_minutes", "total_study_hours",
        "longest_streak_ever", "avg_quiz_attempts_per_lesson", "island_level",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["topic_avg"]         = df[TOPIC_COLUMNS].mean(axis=1).round(2)
    df["final_performance"] = (0.6 * df["quiz_avg_score"] + 0.4 * df["topic_avg"]).round(2)
    df["risk_level"]        = df["final_performance"].apply(get_risk_label)

    df["total_active_days"]    = df[ACTIVE_COLUMNS].sum(axis=1)
    df["recent_active_days"]   = df[ACTIVE_COLUMNS[-4:]].sum(axis=1)
    df["activity_consistency"] = df[ACTIVE_COLUMNS].mean(axis=1).round(3)
    df["xp_growth"]            = df["xp_w18"] - df["xp_w01"]
    df["recent_xp_growth"]     = df["xp_w18"] - df["xp_w14"]

    if "engagement_score" not in df.columns:
        df["engagement_score"] = df.apply(_calc_engagement, axis=1)
    if "learning_velocity" not in df.columns:
        df["learning_velocity"] = df.apply(_calc_velocity, axis=1)
    if "consistency_score" not in df.columns:
        df["consistency_score"] = df.apply(_calc_consistency, axis=1)
    if "predicted_next_xp" not in df.columns:
        df["predicted_next_xp"] = df.apply(_calc_predicted_xp, axis=1)

    return df


def _calc_engagement(row):
    return round(
        min(row["streak_days"] / 90, 1) * 25 +
        min(row["xp_points"]   / 5000, 1) * 25 +
        min(row["quizzes_attempted"] / 50, 1) * 25 +
        min(row["total_study_hours"] / 100, 1) * 25, 2
    )


def _calc_velocity(row):
    active = max(sum(row[c] for c in ACTIVE_COLUMNS), 1)
    return round(row["xp_points"] / active, 2)


def _calc_consistency(row):
    weeks = np.array([row[c] for c in ACTIVE_COLUMNS], dtype=float)
    if weeks.max() == 0:
        return 0.0
    cv = np.std(weeks) / (np.mean(weeks) + 1e-9)
    return round(max(0.0, 100.0 - cv * 50), 2)


def _calc_predicted_xp(row):
    xp_series = [float(row[c]) for c in XP_COLUMNS]
    slope = (xp_series[-1] - xp_series[-4]) / 3
    return round(xp_series[-1] + slope * 4, 2)


def get_topic_dataframe(row):
    return pd.DataFrame([
        {"Topic": TOPIC_LABELS[col], "Score": float(row[col])}
        for col in TOPIC_COLUMNS
    ])


def get_activity_dataframe(row):
    return pd.DataFrame([
        {"Week": f"W{i:02d}", "Active Days": int(row[col])}
        for i, col in enumerate(ACTIVE_COLUMNS, start=1)
    ])


def get_xp_growth_dataframe(row):
    return pd.DataFrame([
        {"Week": f"W{i:02d}", "XP": float(row[col])}
        for i, col in enumerate(XP_COLUMNS, start=1)
    ])


def get_safe_dataframe(df):
    return df.drop(columns=["email", "phone", "referral_code"], errors="ignore")
