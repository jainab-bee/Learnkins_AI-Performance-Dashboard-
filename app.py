import os
import sys
import joblib
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import (
    load_data, get_activity_dataframe, get_topic_dataframe,
    get_xp_growth_dataframe, get_safe_dataframe,
)
from utils.charts import (
    learning_heatmap, topic_mastery_chart, xp_growth_chart,
    student_vs_average_chart, cohort_risk_pie, cohort_course_bar,
    cohort_engagement_hist, badge_distribution_bar,
)
from utils.insights import generate_study_insights, get_growth_status

HERE       = os.path.dirname(__file__)
DATA_PATH  = os.path.join(HERE, "data",   "learnkins_ai_analytics.csv")
MODEL_PATH = os.path.join(HERE, "models", "student_risk_model.pkl")

st.set_page_config(page_title="AI Performance Analytics – LearnKins", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f0f1a; color: #e0e0f0; }

[data-testid="stMetric"] {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #9999bb !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }

h1 { color: #c9bfff !important; }
h2, h3 { color: #a8b4ff !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def load_all_data():
    return load_data(DATA_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["features"]
    except Exception:
        return None, None


def find_student(df, text):
    t = text.strip()
    return df[
        df["student_id"].astype(str).str.contains(t, case=False, na=False) |
        df["full_name"].astype(str).str.contains(t, case=False, na=False)
    ]


def predict_risk(row, model, features):
    if model is None:
        return "Model not trained"
    available = [f for f in features if f in row.index]
    return model.predict(row[available].to_frame().T)[0]


df           = load_all_data()
model, feats = load_model()

st.title("📊 AI Performance Analytics Dashboard")
st.caption("LearnKins · Feature 13 · Search a student to view heatmap, topic mastery, growth tracking and AI insights.")

tab1, tab2, tab3 = st.tabs(["👨‍👩‍👧 Student / Parent View", "📈 Class Overview", "🗃️ Raw Data & Export"])


with tab1:

    st.subheader("Find a Student")
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search", placeholder="Type a name or ID  (e.g. LK0001 or Advait)", label_visibility="collapsed")
    with col2:
        st.caption("Use a first name, full name, or student ID starting with LK.")

    if not search:
        st.info("Type a student name or ID above to see their personalised report.")
        st.stop()

    results = find_student(df, search)

    if results.empty:
        st.error("No student found. Try a different name or ID (e.g. LK0001).")
        st.stop()

    if len(results) > 1:
        options = results["student_id"] + " — " + results["full_name"]
        picked  = st.selectbox("More than one match — please pick one:", options)
        sid     = picked.split(" — ")[0]
        row     = results[results["student_id"] == sid].iloc[0]
    else:
        row = results.iloc[0]

    risk   = predict_risk(row, model, feats)
    growth = get_growth_status(row)

    st.markdown("---")

    left, right = st.columns([3, 1])

    with left:
        st.subheader(f"Student Profile — {row['full_name']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**ID:** {row['student_id']}")
            st.write(f"**Age:** {int(row['age'])}  |  **Grade:** {row['grade']}")
            st.write(f"**City:** {row['city']}")
        with c2:
            st.write(f"**School:** {row['school']}")
            st.write(f"**Course:** {row['enrolled_course']}")
            st.write(f"**Plan:** {row['subscription_plan']}")
        with c3:
            st.write(f"**Badge:** {row['badge']}")
            st.write(f"**Island Level:** {int(row['island_level'])}")
            st.write(f"**AI Risk:** {risk}")

    with right:
        st.metric("Engagement Score",  f"{float(row.get('engagement_score',  0)):.0f} / 100")
        st.metric("Learning Velocity", f"{float(row.get('learning_velocity',  0)):.1f} XP/day")
        st.metric("Consistency Score", f"{float(row.get('consistency_score', 0)):.0f} / 100")

    st.subheader("Learning Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("XP Points",     int(row["xp_points"]))
    m2.metric("Lessons Done",  int(row["lessons_completed"]))
    m3.metric("Day Streak",    int(row["streak_days"]))
    m4.metric("Quizzes Tried", int(row["quizzes_attempted"]))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Quiz Average",  f"{row['quiz_avg_score']:.1f}%")
    m6.metric("Study Hours",   f"{row['total_study_hours']:.1f} h")
    m7.metric("Daily Study",   f"{row['avg_daily_study_minutes']:.0f} min")
    m8.metric("Best Streak",   int(row["longest_streak_ever"]))

    st.markdown("---")

    st.subheader("1. Learning Heatmap")
    st.caption("Each box = one week. Number inside = active days that week (out of 7). Brighter = more active.")
    st.plotly_chart(learning_heatmap(get_activity_dataframe(row)), use_container_width=True)

    total_active  = int(row.get("total_active_days",  0))
    recent_active = int(row.get("recent_active_days", 0))
    h1, h2, h3 = st.columns(3)
    h1.metric("Total Active Days (18 weeks)", total_active)
    h2.metric("Active Days (last 4 weeks)",   recent_active)
    h3.metric("Average Active Days / Week",   f"{total_active / 18:.1f}")

    st.subheader("2. Topic Mastery Graph")
    st.caption("Green = mastered (75%+).  Yellow = okay.  Red = needs more practice.")
    st.plotly_chart(topic_mastery_chart(get_topic_dataframe(row)), use_container_width=True)

    t1, t2, t3 = st.columns(3)
    t1.metric("Best Topic",    row["strongest_topic"])
    t2.metric("Needs Work",    row["weakest_topic"])
    t3.metric("Topic Average", f"{row['topic_avg']:.1f}%")

    st.subheader("3. Growth Tracking")
    st.caption("Line = XP earned each week. Gold star = AI prediction for next week.")
    predicted_xp = float(row.get("predicted_next_xp", 0))
    st.plotly_chart(xp_growth_chart(get_xp_growth_dataframe(row), predicted_xp), use_container_width=True)

    g1, g2, g3 = st.columns(3)
    g1.metric("Growth Status",            growth)
    g2.metric("XP Change (last 4 weeks)", f"+{row['recent_xp_growth']:.0f}")
    g3.metric("AI Predicted Next XP",     f"~{predicted_xp:.0f}")

    st.subheader("Student vs Class Average")
    st.caption("All metrics scaled to 0-100 for a fair side-by-side comparison.")
    st.plotly_chart(student_vs_average_chart(row, df), use_container_width=True)

    st.subheader("4. AI Study Insights")
    insights, recommendation = generate_study_insights(row, risk)

    for i, msg in enumerate(insights):
        msg_lower = msg.lower()
        if i == 0:
            st.info(f"**Overall:** {msg}")
        elif any(w in msg_lower for w in ["excellent", "great", "amazing", "wonderful", "wow", "roll"]):
            st.success(msg)
        elif any(w in msg_lower for w in ["low", "slowed", "only", "more support", "needs"]):
            st.warning(msg)
        else:
            st.info(msg)

    ai_tip = row.get("ai_improvement_tip", "")
    if ai_tip:
        st.markdown(f"> **AI Quick Tip:** {ai_tip}")

    st.success(recommendation)

    with st.expander("See detailed AI numbers"):
        st.markdown(f"""
**AI-Recommended Focus Topic:** {row.get('ai_recommended_focus', 'N/A')}

**Learning Velocity:** {float(row.get('learning_velocity', 0)):.1f} XP per active day  *(Class average: {df['learning_velocity'].mean():.1f})*

**Consistency Score:** {float(row.get('consistency_score', 0)):.0f} / 100  *(Higher = studies at a steady pace each week)*

**Predicted XP (4 weeks ahead):** ~{predicted_xp:.0f} XP  *(Based on the last 4 weeks of growth)*
        """)

    st.subheader("Parent-Friendly Report")

    report = f"""Student Report — {row['full_name']}
Generated by LearnKins AI Performance Analytics Dashboard

STUDENT PROFILE
  Name          : {row['full_name']}
  Student ID    : {row['student_id']}
  Grade & Age   : {row['grade']} | Age {int(row['age'])}
  School        : {row['school']}
  Course        : {row['enrolled_course']}

LEARNING NUMBERS
  XP Points     : {int(row['xp_points'])}
  Lessons Done  : {int(row['lessons_completed'])}
  Day Streak    : {int(row['streak_days'])} days
  Quizzes Tried : {int(row['quizzes_attempted'])}
  Quiz Average  : {row['quiz_avg_score']:.1f}%
  Study Hours   : {row['total_study_hours']:.1f} h

TOPIC PERFORMANCE
  Strongest     : {row['strongest_topic']}
  Needs Work    : {row['weakest_topic']}
  Topic Average : {row['topic_avg']:.1f}%

AI ANALYSIS
  Risk Level    : {risk}
  Growth Status : {growth}
  Engagement    : {float(row.get('engagement_score',  0)):.0f}/100
  Consistency   : {float(row.get('consistency_score', 0)):.0f}/100
  Predicted XP  : ~{predicted_xp:.0f}

RECOMMENDATION
  {recommendation.replace('**', '')}
"""

    st.text_area("Copy or print this report", report, height=320)
    st.download_button(
        label="Download report as .txt",
        data=report,
        file_name=f"learnkins_report_{row['student_id']}.txt",
        mime="text/plain",
    )


with tab2:

    st.subheader("Class-wide Overview — All Students")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",     len(df))
    c2.metric("Avg Quiz Score",     f"{df['quiz_avg_score'].mean():.1f}%")
    c3.metric("Avg XP Points",      f"{df['xp_points'].mean():.0f}")
    c4.metric("Avg Study Hours",    f"{df['total_study_hours'].mean():.1f} h")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Engagement",     f"{df['engagement_score'].mean():.0f}/100")
    c6.metric("Avg Consistency",    f"{df['consistency_score'].mean():.0f}/100")
    c7.metric("Avg Learning Speed", f"{df['learning_velocity'].mean():.1f} XP/day")
    c8.metric("High Risk Students", int((df["risk_level"] == "High Risk").sum()))

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk Level Distribution**")
        st.plotly_chart(cohort_risk_pie(df), use_container_width=True)
    with right:
        st.markdown("**Badge Achievements**")
        st.plotly_chart(badge_distribution_bar(df), use_container_width=True)

    st.markdown("**Quiz Score by Course**")
    st.plotly_chart(cohort_course_bar(df), use_container_width=True)

    st.markdown("**Engagement Score Distribution**")
    st.plotly_chart(cohort_engagement_hist(df), use_container_width=True)

    st.markdown("**Grade-wise Performance**")
    grade_df = (
        df.groupby("grade", as_index=False)
          .agg(avg_quiz=("quiz_avg_score","mean"), avg_xp=("xp_points","mean"),
               avg_eng=("engagement_score","mean"), students=("student_id","count"))
          .round(1)
    )
    grade_df.columns = ["Grade", "Avg Quiz %", "Avg XP", "Avg Engagement", "Students"]
    st.dataframe(grade_df, use_container_width=True)


with tab3:

    st.subheader("Full Dataset — View and Download")
    st.caption("Email, phone and referral code are hidden. Use filters to narrow down the list.")

    safe_df = get_safe_dataframe(df)

    with st.expander("Filter the data"):
        f1, f2, f3 = st.columns(3)
        with f1:
            risk_filter = st.multiselect("Risk Level",
                options=["Low Risk", "Medium Risk", "High Risk"],
                default=["Low Risk", "Medium Risk", "High Risk"])
        with f2:
            course_filter = st.multiselect("Course",
                options=sorted(df["enrolled_course"].unique()),
                default=sorted(df["enrolled_course"].unique()))
        with f3:
            grade_filter = st.multiselect("Grade",
                options=sorted(df["grade"].unique()),
                default=sorted(df["grade"].unique()))

    filtered = safe_df[
        safe_df["risk_level"].isin(risk_filter) &
        safe_df["enrolled_course"].isin(course_filter) &
        safe_df["grade"].isin(grade_filter)
    ]

    st.write(f"Showing **{len(filtered)}** of {len(df)} students")
    st.dataframe(filtered, use_container_width=True, height=400)

    st.download_button(
        label="Download as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="learnkins_ai_analytics_export.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("**What do the AI columns mean?**")
    st.markdown("""
| Column | Meaning |
|---|---|
| `engagement_score` | 0-100 score based on streak, XP, quizzes and study hours |
| `learning_velocity` | XP earned per active day — how fast the student learns |
| `consistency_score` | How evenly the student studies each week (0 = uneven, 100 = very consistent) |
| `predicted_next_xp` | AI estimate of XP 4 weeks from now |
| `ai_recommended_focus` | The topic the AI thinks needs the most attention |
| `ai_improvement_tip` | A short auto-generated tip for the student or parent |
| `risk_level` | Low / Medium / High — based on quiz + topic scores |
| `final_performance` | 60% quiz average + 40% topic average |
    """)
