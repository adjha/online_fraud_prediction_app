import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# -------------------------------
# Load Model + Scaler
# -------------------------------
model = load_model("model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# -------------------------------
# PAGE CONFIG + CUSTOM CSS
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.markdown("""
<style>
/* Main Title styling */
.title {
    font-size: 40px;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #FF4D4D, #8B0000);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Input section card */
.input-card {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    color: white;
}

/* Fraud Alert Box */
.fraud-box {
    padding: 25px;
    border-radius: 12px;
    background: linear-gradient(135deg, #ff4d4d, #8b0000);
    color: white;
    font-size: 22px;
    font-weight: 700;
    animation: slide 0.8s ease-out;
    box-shadow: 0px 6px 15px rgba(255,0,0,0.4);
}

/* Legit Alert Box */
.legit-box {
    padding: 25px;
    border-radius: 12px;
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: white;
    font-size: 22px;
    font-weight: 700;
    animation: slide 0.8s ease-out;
    box-shadow: 0px 6px 15px rgba(0,255,100,0.4);
}

/* Animation */
@keyframes slide {
    from {opacity: 0; transform: translateY(-15px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("<h1 class='title'>🔍 Real-Time Online Payment Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("### Enter Transaction Details Below:")

# -------------------------------
# INPUT SECTION
# -------------------------------
with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    step = st.number_input("Step (Transaction Hour)", min_value=1, max_value=744, value=1)
    amount = st.number_input("Amount", min_value=0.0)
    balance_diff_org = st.number_input("Balance Difference (Origin Account)", value=0.0)
    balance_diff_dest = st.number_input("Balance Difference (Destination Account)", value=0.0)

    st.markdown("### Transaction Type")

    type_CASH_OUT = st.checkbox("Is CASH_OUT?")
    type_DEBIT = st.checkbox("Is DEBIT?")
    type_PAYMENT = st.checkbox("Is PAYMENT?")
    type_TRANSFER = st.checkbox("Is TRANSFER?")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Fraud", use_container_width=True):

    # Correct input order
    input_data = np.array([[
        step,
        amount,
        balance_diff_org,
        balance_diff_dest,
        int(type_CASH_OUT),
        int(type_DEBIT),
        int(type_PAYMENT),
        int(type_TRANSFER)
    ]])

    # Scale values
    scaled_input = input_data.copy()
    scaled_values = scaler.transform(input_data[:, [1, 2, 3]])
    scaled_input[:, [1, 2, 3]] = scaled_values

    # Prediction
    prediction = model.predict(scaled_input)[0][0]

    # -------------------------------
    # FRAUD / LEGIT OUTPUT BOX
    # -------------------------------
    if prediction > 0.5:
        st.markdown(f"""
        <div class='fraud-box'>
            🚨 FRAUD DETECTED!<br>
            Probability: {prediction:.3f}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class='legit-box'>
            ✔ Legit Transaction<br>
            Probability: {prediction:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Raw Model Output: {prediction}")
