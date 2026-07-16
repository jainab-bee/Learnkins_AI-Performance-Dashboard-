def get_growth_status(row):
    change = row.get("recent_xp_growth", 0)
    if change > 300:
        return "Improving fast"
    elif change > 50:
        return "Steady growth"
    elif change >= 0:
        return "Staying flat"
    else:
        return "Slowing down"


def generate_study_insights(row, predicted_risk):
    insights = []
    name = row.get("full_name", "Your child").split()[0]

    if predicted_risk == "Low Risk":
        insights.append(f"{name} is doing great overall! Keep up the wonderful work.")
    elif predicted_risk == "Medium Risk":
        insights.append(f"{name} is doing okay but could do with a little extra push.")
    elif predicted_risk == "High Risk":
        insights.append(f"{name} may need more support right now — a small daily effort makes a big difference!")
    else:
        insights.append("AI risk prediction is not available yet.")

    quiz = row.get("quiz_avg_score", 0)
    if quiz >= 80:
        insights.append(f"Quiz scores are excellent ({quiz:.0f}%) — {name} clearly understands the material well.")
    elif quiz >= 60:
        insights.append(f"Quiz scores are decent ({quiz:.0f}%). A little more practice will push them higher.")
    else:
        insights.append(f"Quiz scores are a bit low ({quiz:.0f}%). Daily revision of the basics will really help {name}.")

    streak = row.get("streak_days", 0)
    if streak >= 14:
        insights.append(f"Wow — a {streak}-day streak! That kind of consistency is the key to real learning.")
    elif streak >= 7:
        insights.append(f"A {streak}-day streak is a good start. Aim for 14 or more days for the best results.")
    else:
        insights.append(f"The current streak is only {streak} days. Encourage {name} to log in every day, even for 10 minutes.")

    recent = row.get("recent_active_days", 0)
    if recent >= 16:
        insights.append("Very active in the last 4 weeks — the learning momentum is strong!")
    elif recent >= 8:
        insights.append("Activity in the last month is moderate. A small push can make a big difference.")
    else:
        insights.append("Activity has been low lately. Setting a daily study reminder can really help.")

    weakest   = row.get("weakest_topic",   "a topic")
    strongest = row.get("strongest_topic", "a topic")
    topic_avg = row.get("topic_avg", 0)
    insights.append(
        f"{name}'s strongest subject is {strongest} — great job! "
        f"The area that needs the most work is {weakest} (topic average: {topic_avg:.0f}%)."
    )

    xp_change = row.get("recent_xp_growth", 0)
    if xp_change > 300:
        insights.append(f"XP jumped by {xp_change:.0f} points recently — {name} is on a roll!")
    elif xp_change > 50:
        insights.append(f"XP is growing at a healthy pace (+{xp_change:.0f} points). Keep it going!")
    else:
        insights.append(f"XP growth has slowed down (+{xp_change:.0f} points recently). More lessons will help.")

    eng = row.get("engagement_score", 0)
    if eng >= 70:
        insights.append(f"Engagement score is {eng:.0f}/100 — {name} is really engaged with the platform.")
    elif eng >= 40:
        insights.append(f"Engagement score is {eng:.0f}/100 — there is room to be even more active.")
    else:
        insights.append(f"Engagement score is {eng:.0f}/100 — {name} needs more regular interaction with the course.")

    focus = row.get("ai_recommended_focus", weakest)
    recommendation = (
        f"Our AI recommends: Focus on **{focus}** this week, "
        f"aim for at least 5 active days, and try 3-5 quizzes to boost the score. "
        f"Small, consistent steps every day lead to big improvements!"
    )

    return insights, recommendation
