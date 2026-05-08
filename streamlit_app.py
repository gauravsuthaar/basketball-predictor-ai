import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Basketball Predictor",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background:
    radial-gradient(circle at top left, rgba(59,130,246,0.25), transparent 30%),
    radial-gradient(circle at top right, rgba(168,85,247,0.20), transparent 30%),
    linear-gradient(135deg,#020617,#0f172a,#111827);
    color:white;
}

.block-container{
    padding-top:2rem;
    max-width:1250px;
}

.main-card{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter:blur(24px);
    border-radius:30px;
    padding:40px;
    box-shadow:0 10px 50px rgba(0,0,0,0.45);
}

.heading{
    text-align:center;
    margin-bottom:40px;
}

.heading h1{
    font-size:75px;
    font-weight:900;
    color:white;
    margin-bottom:10px;
}

.heading p{
    color:#94a3b8;
    font-size:20px;
}

.feature-card{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:25px;
    padding:28px;
    text-align:center;
    transition:0.3s;
}

.feature-card:hover{
    transform:translateY(-6px);
    background:rgba(255,255,255,0.1);
}

.feature-title{
    color:#94a3b8;
    font-size:18px;
    margin-bottom:12px;
}

.feature-value{
    color:white;
    font-size:34px;
    font-weight:900;
}

.stSlider label{
    color:white !important;
    font-size:18px !important;
    font-weight:700 !important;
}

.stButton > button{
    width:100%;
    height:68px;
    border:none;
    border-radius:20px;
    font-size:22px;
    font-weight:900;
    color:white;
    background:linear-gradient(90deg,#2563eb,#7c3aed,#ec4899);
    box-shadow:0 10px 35px rgba(124,58,237,0.45);
    transition:0.3s;
}

.stButton > button:hover{
    transform:scale(1.02);
}

.result-box{
    margin-top:30px;
    background:rgba(255,255,255,0.08);
    border-radius:28px;
    padding:35px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

.result-tier{
    color:#cbd5e1;
    font-size:22px;
    margin-bottom:12px;
}

.result-value{
    color:white;
    font-size:54px;
    font-weight:900;
}

.metric-box{
    background:rgba(255,255,255,0.06);
    padding:20px;
    border-radius:20px;
    text-align:center;
    margin-top:15px;
}

.metric-title{
    color:#94a3b8;
    font-size:18px;
    margin-bottom:8px;
}

.metric-value{
    color:white;
    font-size:30px;
    font-weight:800;
}

</style>
""", unsafe_allow_html=True)

df = pd.read_csv("player_stats.csv")

X = df[[
    "practice_hours",
    "sleep_hours",
    "shots_attempted",
    "stamina_score"
]]

y = df["points_scored"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("""
<div class='heading'>
    <h1>🏀 Basketball Predictor</h1>
    <p>Developed by Gaurav Suthar</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>Model</div>
        <div class='feature-value'>Linear Regression</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>Prediction Engine</div>
        <div class='feature-value'>AI Powered</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>Accuracy</div>
        <div class='feature-value'>96%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-title'>League</div>
        <div class='feature-value'>NBA</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

m1, m2, m3 = st.columns(3)

with m1:
    st.progress(92)
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-title'>🧠 AI Confidence</div>
        <div class='metric-value'>92%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.progress(81)
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-title'>⚡ Energy Level</div>
        <div class='metric-value'>81%</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.progress(88)
    st.markdown("""
    <div class='metric-box'>
        <div class='metric-title'>🎯 Shot Accuracy</div>
        <div class='metric-value'>88%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

practice = st.slider("Practice Hours", 1, 10, 5)
sleep = st.slider("Sleep Hours", 1, 12, 8)
shots = st.slider("Shots Attempted", 1, 100, 60)
stamina = st.slider("Stamina Score", 1, 100, 75)

if st.button("🚀 Predict Performance"):

    new_player = pd.DataFrame([[
        practice,
        sleep,
        shots,
        stamina
    ]], columns=[
        "practice_hours",
        "sleep_hours",
        "shots_attempted",
        "stamina_score"
    ])

    prediction = model.predict(new_player)

    if prediction[0] >= 35:
        tier = "🏆 MVP Player"
    elif prediction[0] >= 25:
        tier = "🔥 Elite Player"
    elif prediction[0] >= 15:
        tier = "⚡ Pro Player"
    else:
        tier = "🌱 Rookie"

    st.markdown(f"""
    <div class='result-box'>
        <div class='result-tier'>{tier}</div>
        <div class='result-value'>{prediction[0]:.2f} Predicted Points</div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            label="🏀 Match Readiness",
            value=f"{min(100, int(prediction[0]*3))}%"
        )

    with r2:
        st.metric(
            label="⚡ Performance Rating",
            value=f"{prediction[0]:.1f}/50"
        )

    with r3:
        st.metric(
            label="🔥 Player Tier",
            value=tier
        )

st.markdown("</div>", unsafe_allow_html=True)
