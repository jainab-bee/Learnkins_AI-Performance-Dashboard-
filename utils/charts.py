import plotly.graph_objects as go
import plotly.express as px

PURPLE = "#6C63FF"
TEAL   = "#00C9A7"
CORAL  = "#FF6584"
AMBER  = "#FFD166"
DARK   = "#1E1E2E"


def learning_heatmap(activity_df):
    weeks = activity_df["Week"].tolist()
    days  = activity_df["Active Days"].tolist()

    fig = go.Figure(go.Heatmap(
        z=[days], x=weeks, y=["Active Days"],
        text=[days], texttemplate="%{text}",
        colorscale=[[0, "#2A2A3E"], [0.4, TEAL], [1, PURPLE]],
        zmin=0, zmax=7,
        colorbar=dict(title="Days"),
        hovertemplate="Week: %{x}  |  Active Days: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Learning Heatmap — Active days each week (out of 7)",
        xaxis_title="Week",
        yaxis=dict(showticklabels=False),
        height=220,
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def topic_mastery_chart(topic_df):
    topic_df = topic_df.sort_values("Score", ascending=True)

    colors = [
        TEAL if s >= 75 else (AMBER if s >= 60 else CORAL)
        for s in topic_df["Score"]
    ]

    fig = go.Figure(go.Bar(
        y=topic_df["Topic"], x=topic_df["Score"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.1f}%" for s in topic_df["Score"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(
        x=75, line_dash="dash", line_color="white",
        annotation_text="Mastery (75%)",
        annotation_position="top right",
        annotation_font_color="white",
    )

    fig.update_layout(
        title="Topic Mastery — Green = great, Yellow = okay, Red = needs work",
        xaxis=dict(title="Score (%)", range=[0, 115]),
        yaxis_title="",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        height=340,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def xp_growth_chart(xp_df, predicted_xp=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xp_df["Week"], y=xp_df["XP"],
        mode="lines+markers",
        name="Weekly XP",
        line=dict(color=PURPLE, width=3),
        marker=dict(size=7),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.15)",
        hovertemplate="Week %{x}: %{y} XP<extra></extra>",
    ))

    if predicted_xp:
        fig.add_trace(go.Scatter(
            x=["W19*"], y=[predicted_xp],
            mode="markers+text",
            name="AI Prediction (W19)",
            marker=dict(size=14, color=AMBER, symbol="star"),
            text=[f"~{predicted_xp:.0f}"],
            textposition="top center",
            hovertemplate="Predicted W19 XP: %{y:.0f}<extra></extra>",
        ))

    fig.update_layout(
        title="Growth Tracking — XP earned each week (gold star = AI prediction)",
        xaxis_title="Week", yaxis_title="XP Points",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.1),
        height=340,
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


def student_vs_average_chart(row, df):
    metrics = [
        ("XP Points",     "xp_points"),
        ("Lessons Done",  "lessons_completed"),
        ("Streak Days",   "streak_days"),
        ("Quizzes Tried", "quizzes_attempted"),
        ("Quiz Score %",  "quiz_avg_score"),
        ("Study Hours",   "total_study_hours"),
        ("Engagement",    "engagement_score"),
        ("Consistency",   "consistency_score"),
    ]

    labels    = [m[0] for m in metrics]
    student   = [float(row[m[1]]) for m in metrics]
    class_avg = [float(df[m[1]].mean()) for m in metrics]

    max_vals  = [max(s, a, 1) for s, a in zip(student, class_avg)]
    student_n = [round(s / mx * 100, 1) for s, mx in zip(student, max_vals)]
    avg_n     = [round(a / mx * 100, 1) for a, mx in zip(class_avg, max_vals)]

    fig = go.Figure([
        go.Bar(name="This student",  x=labels, y=student_n, marker_color=PURPLE,
               text=[f"{v:.0f}" for v in student_n], textposition="outside"),
        go.Bar(name="Class average", x=labels, y=avg_n,     marker_color=TEAL,
               text=[f"{v:.0f}" for v in avg_n],   textposition="outside"),
    ])

    fig.update_layout(
        title="Student vs Class Average (scaled to 0-100)",
        barmode="group",
        yaxis=dict(title="Relative Score", range=[0, 130]),
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.1),
        height=360,
        margin=dict(l=20, r=20, t=60, b=30),
    )
    return fig


def cohort_risk_pie(df):
    counts = df["risk_level"].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]

    color_map = {"Low Risk": TEAL, "Medium Risk": AMBER, "High Risk": CORAL}
    colors    = [color_map.get(r, PURPLE) for r in counts["Risk Level"]]

    fig = go.Figure(go.Pie(
        labels=counts["Risk Level"], values=counts["Count"],
        hole=0.4, marker=dict(colors=colors),
        hovertemplate="%{label}: %{value} students<extra></extra>",
    ))
    fig.update_layout(
        title="How many students are at risk?",
        paper_bgcolor=DARK, font=dict(color="white"),
        height=320,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def cohort_course_bar(df):
    course_df = (
        df.groupby("enrolled_course", as_index=False)["quiz_avg_score"]
          .mean().sort_values("quiz_avg_score", ascending=True)
    )
    fig = px.bar(
        course_df, x="quiz_avg_score", y="enrolled_course",
        orientation="h", text="quiz_avg_score",
        color="quiz_avg_score",
        color_continuous_scale=[[0, CORAL], [0.5, AMBER], [1, TEAL]],
        title="Average Quiz Score by Course",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis=dict(title="Average Score (%)", range=[0, 110]),
        yaxis_title="", coloraxis_showscale=False,
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        height=360, margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def cohort_engagement_hist(df):
    fig = px.histogram(
        df, x="engagement_score", nbins=20,
        title="Engagement Score across all students",
        color_discrete_sequence=[PURPLE],
    )
    fig.update_layout(
        xaxis_title="Engagement Score (0-100)",
        yaxis_title="Number of Students",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        height=300, margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def badge_distribution_bar(df):
    badge_order  = ["Beginner", "Explorer", "Coder", "Builder", "Innovator", "Champion"]
    badge_colors = [CORAL, AMBER, TEAL, PURPLE, "#06D6A0", "#FF9F1C"]

    badge_df = (
        df["badge"].value_counts()
          .reindex(badge_order, fill_value=0).reset_index()
    )
    badge_df.columns = ["Badge", "Count"]

    fig = go.Figure(go.Bar(
        x=badge_df["Badge"], y=badge_df["Count"],
        marker=dict(color=badge_colors),
        text=badge_df["Count"], textposition="outside",
    ))
    fig.update_layout(
        title="Badge Achievements",
        yaxis_title="Number of Students",
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="white"),
        height=300, margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig
