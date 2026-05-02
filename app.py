import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import requests
import matplotlib.cm as cm
import urllib.parse
import tempfile
import os
import datetime
import csv
import re
import difflib
from fpdf import FPDF

# ═════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriGuard – Crop Disease Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.markdown("""
# <style>
#     .block-container { padding-top: 1.5rem; }
#     .disease-card {
#         background: white; padding: 18px; border-radius: 10px;
#         border-left: 6px solid #2d8a4e; margin: 8px 0;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.07);
#     }
#     .risk-high   { border-left-color: #e74c3c; }
#     .risk-medium { border-left-color: #f39c12; }
#     .risk-low    { border-left-color: #27ae60; }
#     .hindi-box {
#         background: #fff9e6; padding: 16px; border-radius: 10px;
#         border: 1px solid #f0d060; margin: 10px 0;
#         font-size: 1.05em;
#     }
# </style>
# """, unsafe_allow_html=True)
st.markdown("""
<style>
    .block-container {
        padding-top: 4rem !important;  /* increase space */
    }

    header[data-testid="stHeader"] {
        z-index: 999 !important;
    }

    .stTabs {
        position: relative;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)
# ═════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    for p in ["trained_model.keras", "trained_model.h5", "trained_model"]:
        try:
            m = tf.keras.models.load_model(p, compile=False)
            # Ensure Sequential models are fully built (required for Keras 3)
            if not m.built:
                m.build((None, 128, 128, 3))
            return m
        except Exception:
            continue
    raise RuntimeError("No model file found.")

model = load_model()

dummy_input = np.zeros((1, 128, 128, 3))
model.predict(dummy_input)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

# ═════════════════════════════════════════════════════════════════
# TREATMENT DATABASE  (38 classes – English + Hindi + loss %)
# ═════════════════════════════════════════════════════════════════
TREATMENT_DB = {
    # ── APPLE ────────────────────────────────────────────────────
    'Apple___Apple_scab': {
        'display_name': 'Apple Scab',
        'hindi_name': 'सेब का पपड़ी रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Dark scabby lesions on leaves and fruit; yellowing around spots; premature leaf drop.',
        'chemical_treatment': 'Apply Captan (2 g/L), Mancozeb (2.5 g/L), or Myclobutanil. Spray every 7–10 days during wet weather.',
        'organic_treatment': 'Spray 1 % Bordeaux mixture or neem oil (5 ml/L). Remove and destroy infected leaves.',
        'prevention': 'Prune for air circulation; rake fallen leaves; plant resistant varieties; avoid overhead irrigation.',
        'crop_loss_risk': '30–70 % fruit yield loss if untreated',
        'recovery_time': '2–4 weeks with treatment',
        'loss_pct': 50,
        'hindi_treatment': 'कैप्टान (2 ग्रा/ली) या मैंकोज़ेब (2.5 ग्रा/ली) का छिड़काव करें। नीम तेल (5 मिली/ली) भी प्रभावी है। संक्रमित पत्तियाँ हटाकर नष्ट करें।',
    },
    'Apple___Black_rot': {
        'display_name': 'Apple Black Rot',
        'hindi_name': 'सेब का काला सड़न रोग',
        'disease_type': 'Fungal',
        'severity_level': 'High',
        'urgency': 'Act within 3 days',
        'symptoms': 'Purple leaf spots expanding to brown; black rotting fruit; cankers on branches.',
        'chemical_treatment': 'Captan, Thiophanate-methyl, or Propiconazole — begin at pink bud stage.',
        'organic_treatment': 'Copper hydroxide sprays (3 g/L). Remove mummified fruits from trees.',
        'prevention': 'Remove dead wood and cankered branches; sterilize pruning tools between cuts.',
        'crop_loss_risk': '50–80 % fruit loss if untreated',
        'recovery_time': '3–5 weeks with aggressive treatment',
        'loss_pct': 65,
        'hindi_treatment': 'कैप्टान या प्रोपिकोनाज़ोल का छिड़काव करें। सड़े हुए फल और मृत शाखाएँ तुरंत हटाएँ। कॉपर हाइड्रॉक्साइड (3 ग्रा/ली) छिड़कें।',
    },
    'Apple___Cedar_apple_rust': {
        'display_name': 'Cedar Apple Rust',
        'hindi_name': 'सेब का जंग रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Bright orange-yellow spots on upper leaf surface; tube-like structures on leaf undersides.',
        'chemical_treatment': 'Myclobutanil or Propiconazole applied from bud break through 5–6 weeks post bloom.',
        'organic_treatment': 'Sulfur-based fungicides (wettable sulfur 3 g/L). Remove galls from nearby cedar/juniper trees.',
        'prevention': 'Plant rust-resistant apple varieties; remove juniper galls before spring.',
        'crop_loss_risk': '20–50 % defoliation; reduced fruit quality',
        'recovery_time': '4–6 weeks',
        'loss_pct': 35,
        'hindi_treatment': 'माइक्लोब्यूटानिल या सल्फर आधारित फफूंदनाशक का छिड़काव करें। आस-पास के देवदार पेड़ों से गॉल हटाएँ।',
    },
    'Apple___healthy': {
        'display_name': 'Healthy Apple Plant',
        'hindi_name': 'स्वस्थ सेब का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected. Plant appears healthy.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Continue regular maintenance.',
        'prevention': 'Maintain regular pruning, balanced NPK (10-10-10), and proper irrigation.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। नियमित छंटाई और संतुलित उर्वरक देते रहें।',
    },
    # ── BLUEBERRY ────────────────────────────────────────────────
    'Blueberry___healthy': {
        'display_name': 'Healthy Blueberry Plant',
        'hindi_name': 'स्वस्थ ब्लूबेरी पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Maintain soil pH 4.5–5.5 with sulfur amendments.',
        'prevention': 'Mulch with pine needles; ensure acidic soil; avoid overhead watering.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। मिट्टी का pH 4.5–5.5 बनाए रखें।',
    },
    # ── CHERRY ───────────────────────────────────────────────────
    'Cherry_(including_sour)___Powdery_mildew': {
        'display_name': 'Cherry Powdery Mildew',
        'hindi_name': 'चेरी का चूर्णिल आसिता रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'White powdery coating on young leaves; distorted and curled leaves; stunted shoot growth.',
        'chemical_treatment': 'Myclobutanil, Trifloxystrobin, or sulfur-based fungicides every 10–14 days.',
        'organic_treatment': 'Potassium bicarbonate (5 g/L) or diluted milk solution (1:9 with water).',
        'prevention': 'Improve air circulation through pruning; avoid excess nitrogen fertilizer.',
        'crop_loss_risk': '20–40 % reduction in fruit set',
        'recovery_time': '2–3 weeks',
        'loss_pct': 30,
        'hindi_treatment': 'सल्फर फफूंदनाशक या पोटैशियम बाइकार्बोनेट (5 ग्रा/ली) का छिड़काव करें। दूध का घोल (1:9) भी प्रभावी है।',
    },
    'Cherry_(including_sour)___healthy': {
        'display_name': 'Healthy Cherry Plant',
        'hindi_name': 'स्वस्थ चेरी का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Regular composting and balanced nutrition.',
        'prevention': 'Annual pruning for structure and air flow; monitor for pests.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। नियमित खाद और पोषण देते रहें।',
    },
    # ── CORN ─────────────────────────────────────────────────────
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'display_name': 'Corn Gray Leaf Spot',
        'hindi_name': 'मक्का का भूरा पत्ती धब्बा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'High',
        'urgency': 'Act within 3 days',
        'symptoms': 'Rectangular gray-tan lesions parallel to leaf veins; blighting of lower leaves.',
        'chemical_treatment': 'Azoxystrobin, Pyraclostrobin, or Propiconazole at tasseling stage.',
        'organic_treatment': 'Crop rotation with non-host crops; copper-based sprays as preventive.',
        'prevention': 'Plant resistant hybrids; rotate crops; reduce crop residue.',
        'crop_loss_risk': '40–60 % yield loss in severe cases',
        'recovery_time': '4–6 weeks (prevention more effective than cure)',
        'loss_pct': 50,
        'hindi_treatment': 'एज़ोक्सिस्ट्रोबिन या प्रोपिकोनाज़ोल का छिड़काव करें। फसल चक्र अपनाएँ। ताँबा आधारित स्प्रे रोकथाम के लिए करें।',
    },
    'Corn_(maize)___Common_rust_': {
        'display_name': 'Corn Common Rust',
        'hindi_name': 'मक्का का सामान्य जंग रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Brick-red to brown pustules scattered on both leaf surfaces; pustules turn black with age.',
        'chemical_treatment': 'Propiconazole 25 % EC (1 ml/L) or Triazole fungicides before tasseling.',
        'organic_treatment': 'Neem oil spray (5 ml/L) every 7 days; improve plant spacing.',
        'prevention': 'Plant rust-resistant varieties; early planting to escape peak season.',
        'crop_loss_risk': '15–40 % yield reduction',
        'recovery_time': '3–4 weeks',
        'loss_pct': 27,
        'hindi_treatment': 'प्रोपिकोनाज़ोल (1 मिली/ली) या नीम तेल (5 मिली/ली) का छिड़काव हर 7 दिन करें। जंग प्रतिरोधी किस्में लगाएँ।',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'display_name': 'Corn Northern Leaf Blight',
        'hindi_name': 'मक्का का उत्तरी पत्ती झुलसा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'High',
        'urgency': 'Act within 5 days',
        'symptoms': 'Long (2.5–15 cm) cigar-shaped gray-green to tan lesions; rapid lower-leaf blighting.',
        'chemical_treatment': 'Azoxystrobin + Propiconazole or Picoxystrobin at VT stage.',
        'organic_treatment': 'Crop rotation; remove and destroy infected plant debris after harvest.',
        'prevention': 'Use resistant hybrids; practice crop rotation; manage residue.',
        'crop_loss_risk': '30–50 % yield loss',
        'recovery_time': '4–8 weeks',
        'loss_pct': 40,
        'hindi_treatment': 'एज़ोक्सिस्ट्रोबिन + प्रोपिकोनाज़ोल का छिड़काव करें। फसल अवशेष नष्ट करें। प्रतिरोधी संकर किस्में उगाएँ।',
    },
    'Corn_(maize)___healthy': {
        'display_name': 'Healthy Corn Plant',
        'hindi_name': 'स्वस्थ मक्का का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Compost and balanced NPK fertilization.',
        'prevention': 'Monitor weekly; maintain proper plant spacing; avoid water-logging.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। संतुलित खाद और उचित सिंचाई जारी रखें।',
    },
    # ── GRAPE ────────────────────────────────────────────────────
    'Grape___Black_rot': {
        'display_name': 'Grape Black Rot',
        'hindi_name': 'अंगूर का काला सड़न रोग',
        'disease_type': 'Fungal',
        'severity_level': 'High',
        'urgency': 'Act within 3 days',
        'symptoms': 'Brown circular lesions on leaves with black dots; berries shrivel into black mummies.',
        'chemical_treatment': 'Mancozeb, Myclobutanil, or Tebuconazole — begin at budbreak, every 10–14 days.',
        'organic_treatment': 'Bordeaux mixture (1 %); remove mummified berries and infected canes immediately.',
        'prevention': 'Prune for open canopy; remove all mummies; avoid wetting foliage.',
        'crop_loss_risk': '80–100 % crop loss in severe cases',
        'recovery_time': 'Current season largely lost; focus on next season prevention',
        'loss_pct': 90,
        'hindi_treatment': 'मैंकोज़ेब या टेबुकोनाज़ोल का छिड़काव करें। सूखे (ममी) फल तुरंत हटाएँ। बोर्डो मिश्रण (1%) का प्रयोग करें।',
    },
    'Grape___Esca_(Black_Measles)': {
        'display_name': 'Grape Esca (Black Measles)',
        'hindi_name': 'अंगूर का एस्का (काला खसरा) रोग',
        'disease_type': 'Fungal',
        'severity_level': 'High',
        'urgency': 'Act immediately',
        'symptoms': 'Tiger-stripe pattern on leaves; internal wood browning; sudden vine collapse (apoplexy).',
        'chemical_treatment': 'No fully effective cure. Protect pruning wounds with fungicidal paste.',
        'organic_treatment': 'Remove and destroy severely infected vines.',
        'prevention': 'Prune in dry weather; seal large pruning wounds; avoid water stress.',
        'crop_loss_risk': 'Potentially 100 % vine loss',
        'recovery_time': 'Chronic disease — long-term management required',
        'loss_pct': 100,
        'hindi_treatment': 'कोई प्रभावी इलाज नहीं। संक्रमित बेलें उखाड़कर नष्ट करें। छंटाई शुष्क मौसम में करें और घाव पर फफूंदनाशक लेप लगाएँ।',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'display_name': 'Grape Leaf Blight',
        'hindi_name': 'अंगूर का पत्ती झुलसा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Dark brown spots with yellow halos on older leaves; premature defoliation.',
        'chemical_treatment': 'Copper-based fungicides or Mancozeb after rain events.',
        'organic_treatment': 'Bordeaux mixture; improve canopy management to reduce humidity.',
        'prevention': 'Remove infected leaves; improve air circulation through shoot positioning.',
        'crop_loss_risk': '20–40 % defoliation; reduced berry quality',
        'recovery_time': '3–4 weeks',
        'loss_pct': 30,
        'hindi_treatment': 'ताँबा आधारित फफूंदनाशक या मैंकोज़ेब का बारिश के बाद छिड़काव करें। संक्रमित पत्तियाँ हटाएँ।',
    },
    'Grape___healthy': {
        'display_name': 'Healthy Grape Vine',
        'hindi_name': 'स्वस्थ अंगूर की बेल',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Regular canopy management; balanced fertilization.',
        'prevention': 'Annual pruning; monitor for disease pressure; maintain soil health.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। नियमित छंटाई और संतुलित खाद जारी रखें।',
    },
    # ── ORANGE ───────────────────────────────────────────────────
    'Orange___Haunglongbing_(Citrus_greening)': {
        'display_name': 'Citrus Greening (HLB)',
        'hindi_name': 'संतरा हरापन (साइट्रस ग्रीनिंग) रोग',
        'disease_type': 'Bacterial',
        'severity_level': 'Critical',
        'urgency': 'URGENT — Report to agriculture dept',
        'symptoms': 'Asymmetric yellowing; small misshapen bitter fruit; twig dieback.',
        'chemical_treatment': 'No cure. Manage psyllid vector with Imidacloprid or Dimethoate.',
        'organic_treatment': 'Remove and destroy infected trees. Use certified disease-free planting material.',
        'prevention': 'Control Asian citrus psyllid; use certified budwood; establish psyllid-free nurseries.',
        'crop_loss_risk': '100 % tree loss over time; entire orchard at risk',
        'recovery_time': 'No recovery — infected trees must be removed',
        'loss_pct': 100,
        'hindi_treatment': 'कोई इलाज नहीं। संक्रमित पेड़ उखाड़कर नष्ट करें। साइला कीट नियंत्रण के लिए इमिडाक्लोप्रिड का उपयोग करें। प्रमाणित रोग-मुक्त पौध लगाएँ।',
    },
    # ── PEACH ────────────────────────────────────────────────────
    'Peach___Bacterial_spot': {
        'display_name': 'Peach Bacterial Spot',
        'hindi_name': 'आड़ू का जीवाणु धब्बा रोग',
        'disease_type': 'Bacterial',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Small water-soaked spots on leaves turning brown with yellow halos; fruit cracking.',
        'chemical_treatment': 'Copper hydroxide bactericide from dormancy through first cover sprays.',
        'organic_treatment': 'Copper sulfate sprays; avoid overhead irrigation.',
        'prevention': 'Plant resistant varieties; avoid working in orchard when wet; proper spacing.',
        'crop_loss_risk': '30–50 % fruit loss in wet seasons',
        'recovery_time': '4–6 weeks',
        'loss_pct': 40,
        'hindi_treatment': 'कॉपर हाइड्रॉक्साइड जीवाणुनाशक का छिड़काव करें। ऊपरी सिंचाई से बचें। प्रतिरोधी किस्में लगाएँ।',
    },
    'Peach___healthy': {
        'display_name': 'Healthy Peach Tree',
        'hindi_name': 'स्वस्थ आड़ू का पेड़',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Regular composting; balanced fertilization.',
        'prevention': 'Annual thinning; monitor for disease and pests.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। नियमित खाद और निगरानी जारी रखें।',
    },
    # ── PEPPER ───────────────────────────────────────────────────
    'Pepper,_bell___Bacterial_spot': {
        'display_name': 'Bell Pepper Bacterial Spot',
        'hindi_name': 'शिमला मिर्च का जीवाणु धब्बा रोग',
        'disease_type': 'Bacterial',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Small dark water-soaked spots on leaves and fruit; yellow halos; defoliation.',
        'chemical_treatment': 'Copper hydroxide (3 g/L) + Mancozeb (2 g/L) tank mix every 5–7 days.',
        'organic_treatment': 'Copper-based sprays; remove infected debris; avoid overhead watering.',
        'prevention': 'Certified disease-free seeds; crop rotation; avoid working plants when wet.',
        'crop_loss_risk': '20–50 % yield loss',
        'recovery_time': '3–5 weeks',
        'loss_pct': 35,
        'hindi_treatment': 'कॉपर हाइड्रॉक्साइड (3 ग्रा/ली) + मैंकोज़ेब (2 ग्रा/ली) का मिश्रण हर 5-7 दिन छिड़कें। संक्रमित अवशेष हटाएँ।',
    },
    'Pepper,_bell___healthy': {
        'display_name': 'Healthy Bell Pepper',
        'hindi_name': 'स्वस्थ शिमला मिर्च',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Mulching; drip irrigation; regular composting.',
        'prevention': 'Crop rotation every 2–3 years; balanced fertilization.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। मल्चिंग और ड्रिप सिंचाई जारी रखें।',
    },
    # ── POTATO ───────────────────────────────────────────────────
    'Potato___Early_blight': {
        'display_name': 'Potato Early Blight',
        'hindi_name': 'आलू का अगेती झुलसा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Dark brown spots with concentric rings (target pattern) on older leaves; yellowing.',
        'chemical_treatment': 'Chlorothalonil (2 g/L), Mancozeb (2.5 g/L), or Azoxystrobin every 7–10 days.',
        'organic_treatment': 'Copper fungicides; neem oil (5 ml/L); remove infected lower leaves.',
        'prevention': 'Balanced NPK; avoid water stress; use certified seed potato.',
        'crop_loss_risk': '20–30 % tuber yield reduction',
        'recovery_time': '3–4 weeks',
        'loss_pct': 25,
        'hindi_treatment': 'मैंकोज़ेब (2.5 ग्रा/ली) या क्लोरोथैलोनिल (2 ग्रा/ली) हर 7-10 दिन छिड़कें। नीम तेल (5 मिली/ली) भी उपयोगी। संक्रमित निचली पत्तियाँ हटाएँ।',
    },
    'Potato___Late_blight': {
        'display_name': 'Potato Late Blight',
        'hindi_name': 'आलू का पछेती झुलसा रोग',
        'disease_type': 'Fungal (Oomycete)',
        'severity_level': 'Critical',
        'urgency': 'URGENT — Act within 24–48 hours',
        'symptoms': 'Water-soaked dark lesions on leaves/stems; white fluffy growth on leaf undersides; rapid collapse.',
        'chemical_treatment': 'Metalaxyl + Mancozeb (Ridomil Gold MZ), Cymoxanil, or Dimethomorph every 5–7 days.',
        'organic_treatment': 'Copper-based fungicides as preventive; destroy affected plants immediately.',
        'prevention': 'Plant resistant varieties; avoid overhead irrigation; destroy cull piles; haulm destruction before harvest.',
        'crop_loss_risk': '75–100 % crop loss if untreated (caused the Irish Famine)',
        'recovery_time': 'If caught early: 2–3 weeks. Late: crop may be lost',
        'loss_pct': 87,
        'hindi_treatment': 'तुरंत कार्रवाई करें! मेटालैक्सिल + मैंकोज़ेब (रिडोमिल गोल्ड) हर 5-7 दिन छिड़कें। संक्रमित पौधे तुरंत नष्ट करें। ताँबा फफूंदनाशक का रोकथाम हेतु उपयोग करें।',
    },
    'Potato___healthy': {
        'display_name': 'Healthy Potato Plant',
        'hindi_name': 'स्वस्थ आलू का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Hill soil around stems; high K fertilization for tuber quality.',
        'prevention': 'Certified disease-free seed potato; crop rotation every 3–4 years.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। प्रमाणित बीज आलू का उपयोग करें।',
    },
    # ── RASPBERRY ────────────────────────────────────────────────
    'Raspberry___healthy': {
        'display_name': 'Healthy Raspberry Plant',
        'hindi_name': 'स्वस्थ रसभरी का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Annual cane management; mulching.',
        'prevention': 'Remove old canes after harvest; maintain proper spacing.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं।',
    },
    # ── SOYBEAN ──────────────────────────────────────────────────
    'Soybean___healthy': {
        'display_name': 'Healthy Soybean Plant',
        'hindi_name': 'स्वस्थ सोयाबीन का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Rhizobium inoculation; balanced PK fertilization.',
        'prevention': 'Crop rotation; pest monitoring; proper plant density.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। राइज़ोबियम टीकाकरण जारी रखें।',
    },
    # ── SQUASH ───────────────────────────────────────────────────
    'Squash___Powdery_mildew': {
        'display_name': 'Squash Powdery Mildew',
        'hindi_name': 'कद्दू का चूर्णिल आसिता रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'White powdery patches on leaf surfaces; yellowing of older leaves.',
        'chemical_treatment': 'Myclobutanil, Trifloxystrobin, or potassium bicarbonate every 7–10 days.',
        'organic_treatment': '1:9 milk-water solution; baking soda (5 g/L); neem oil every 7 days.',
        'prevention': 'Plant resistant varieties; space plants for air circulation; avoid high N fertilization.',
        'crop_loss_risk': '15–35 % fruit yield reduction',
        'recovery_time': '2–3 weeks',
        'loss_pct': 25,
        'hindi_treatment': 'दूध-पानी का घोल (1:9) या नीम तेल हर 7 दिन छिड़कें। बेकिंग सोडा (5 ग्रा/ली) भी प्रभावी है।',
    },
    # ── STRAWBERRY ───────────────────────────────────────────────
    'Strawberry___Leaf_scorch': {
        'display_name': 'Strawberry Leaf Scorch',
        'hindi_name': 'स्ट्रॉबेरी का पत्ती दाह रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Small dark purple spots on leaves; leaf margins turn brown (scorched); plant weakening.',
        'chemical_treatment': 'Captan, Thiram, or Myclobutanil every 10 days.',
        'organic_treatment': 'Copper hydroxide sprays; remove old leaves at renovation.',
        'prevention': 'Certified disease-free runners; mowing after harvest; avoid waterlogged soil.',
        'crop_loss_risk': '20–40 % plant weakening; reduced next-season yield',
        'recovery_time': '3–5 weeks',
        'loss_pct': 30,
        'hindi_treatment': 'कैप्टान या माइक्लोब्यूटानिल हर 10 दिन छिड़कें। कॉपर हाइड्रॉक्साइड स्प्रे भी उपयोगी। पुरानी पत्तियाँ हटाएँ।',
    },
    'Strawberry___healthy': {
        'display_name': 'Healthy Strawberry Plant',
        'hindi_name': 'स्वस्थ स्ट्रॉबेरी का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Mulch with straw; balanced P and K fertilization.',
        'prevention': 'Annual renovation; replace plants every 3–4 years.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं।',
    },
    # ── TOMATO ───────────────────────────────────────────────────
    'Tomato___Bacterial_spot': {
        'display_name': 'Tomato Bacterial Spot',
        'hindi_name': 'टमाटर का जीवाणु धब्बा रोग',
        'disease_type': 'Bacterial',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Small dark water-soaked spots on leaves and fruit; yellow halos; fruit blemishes.',
        'chemical_treatment': 'Copper hydroxide (3 g/L) + Mancozeb (2 g/L) tank mix every 5–7 days.',
        'organic_treatment': 'Copper-based sprays; remove infected leaves; avoid overhead watering.',
        'prevention': 'Certified seeds; crop rotation; avoid working plants when wet; stake plants.',
        'crop_loss_risk': '20–50 % marketable fruit loss',
        'recovery_time': '3–5 weeks',
        'loss_pct': 35,
        'hindi_treatment': 'कॉपर हाइड्रॉक्साइड (3 ग्रा/ली) + मैंकोज़ेब (2 ग्रा/ली) हर 5-7 दिन छिड़कें। संक्रमित पत्तियाँ हटाएँ। ऊपरी सिंचाई न करें।',
    },
    'Tomato___Early_blight': {
        'display_name': 'Tomato Early Blight',
        'hindi_name': 'टमाटर का अगेती झुलसा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Dark concentric target-pattern spots on older leaves; yellow halos; upward progression.',
        'chemical_treatment': 'Chlorothalonil, Mancozeb, or Azoxystrobin every 7 days during wet weather.',
        'organic_treatment': 'Copper sulfate spray; neem oil (5 ml/L); remove infected lower leaves.',
        'prevention': 'Mulch to prevent soil splash; stake plants; crop rotation; balanced NPK.',
        'crop_loss_risk': '25–50 % yield loss if defoliation severe',
        'recovery_time': '3–4 weeks',
        'loss_pct': 37,
        'hindi_treatment': 'मैंकोज़ेब या क्लोरोथैलोनिल हर 7 दिन छिड़कें। नीम तेल (5 मिली/ली) का भी उपयोग करें। संक्रमित निचली पत्तियाँ हटाएँ। मल्चिंग करें।',
    },
    'Tomato___Late_blight': {
        'display_name': 'Tomato Late Blight',
        'hindi_name': 'टमाटर का पछेती झुलसा रोग',
        'disease_type': 'Fungal (Oomycete)',
        'severity_level': 'Critical',
        'urgency': 'URGENT — Act within 24 hours',
        'symptoms': 'Greasy dark water-soaked patches on leaves/stems; white mold on undersides; rapid collapse.',
        'chemical_treatment': 'Metalaxyl-M + Chlorothalonil (Ridomil Gold), Cymoxanil + Mancozeb every 5–7 days.',
        'organic_treatment': 'Copper fungicides as preventive; immediately remove affected plants.',
        'prevention': 'Avoid overhead irrigation; remove infected debris; plant resistant varieties.',
        'crop_loss_risk': '80–100 % crop loss possible within days',
        'recovery_time': 'Caught early: 1–2 weeks. Late: crop likely lost',
        'loss_pct': 90,
        'hindi_treatment': 'तुरंत कार्रवाई! मेटालैक्सिल + मैंकोज़ेब हर 5-7 दिन छिड़कें। संक्रमित पौधे तुरंत हटाएँ और नष्ट करें। ऊपरी सिंचाई बंद करें।',
    },
    'Tomato___Leaf_Mold': {
        'display_name': 'Tomato Leaf Mold',
        'hindi_name': 'टमाटर का पत्ती फफूंद रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Pale greenish-yellow spots on leaf upper surface; olive-green velvety mold on undersides.',
        'chemical_treatment': 'Chlorothalonil, Mancozeb, or copper-based fungicides every 7–14 days.',
        'organic_treatment': 'Reduce humidity; improve ventilation; neem oil spray.',
        'prevention': 'Keep relative humidity below 85 %; improve air circulation; use resistant varieties.',
        'crop_loss_risk': '15–40 % yield loss (mainly greenhouse tomatoes)',
        'recovery_time': '2–3 weeks with humidity control',
        'loss_pct': 27,
        'hindi_treatment': 'हवा का संचार बढ़ाएँ और नमी कम करें। क्लोरोथैलोनिल या मैंकोज़ेब हर 7-14 दिन छिड़कें। नीम तेल भी सहायक।',
    },
    'Tomato___Septoria_leaf_spot': {
        'display_name': 'Tomato Septoria Leaf Spot',
        'hindi_name': 'टमाटर का सेप्टोरिया पत्ती धब्बा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Numerous small circular spots with dark borders and light centers on lower leaves; defoliation.',
        'chemical_treatment': 'Chlorothalonil (2 g/L), Mancozeb, or copper fungicides every 7–10 days.',
        'organic_treatment': 'Copper sulfate spray; remove affected lower leaves; mulch against soil splash.',
        'prevention': 'Crop rotation (3-year cycle); avoid overhead irrigation; stake plants.',
        'crop_loss_risk': '30–50 % yield reduction from defoliation',
        'recovery_time': '3–4 weeks',
        'loss_pct': 40,
        'hindi_treatment': 'क्लोरोथैलोनिल (2 ग्रा/ली) या मैंकोज़ेब हर 7-10 दिन छिड़कें। संक्रमित निचली पत्तियाँ हटाएँ। मल्चिंग करें।',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'display_name': 'Tomato Spider Mites',
        'hindi_name': 'टमाटर पर मकड़ी के कण',
        'disease_type': 'Pest (Arachnid)',
        'severity_level': 'Moderate',
        'urgency': 'Act within 5 days',
        'symptoms': 'Fine stippling on leaves; bronze/silver discoloration; fine webbing on undersides; leaf drop.',
        'chemical_treatment': 'Abamectin, Bifenazate, or Spiromesifen — rotate products to prevent resistance.',
        'organic_treatment': 'Neem oil (5 ml/L); insecticidal soap; predatory mites (Phytoseiulus persimilis).',
        'prevention': 'Avoid plant water stress; control dust; avoid broad-spectrum pesticides.',
        'crop_loss_risk': '20–50 % yield loss in hot dry conditions',
        'recovery_time': '2–3 weeks',
        'loss_pct': 35,
        'hindi_treatment': 'नीम तेल (5 मिली/ली) या कीटनाशक साबुन का छिड़काव करें। शिकारी कण (फाइटोसीलस) छोड़ें। पौधों को पानी का तनाव न होने दें।',
    },
    'Tomato___Target_Spot': {
        'display_name': 'Tomato Target Spot',
        'hindi_name': 'टमाटर का लक्ष्य धब्बा रोग',
        'disease_type': 'Fungal',
        'severity_level': 'Moderate',
        'urgency': 'Act within 7 days',
        'symptoms': 'Brown circular spots with concentric rings on leaves; similar lesions on fruit and stems.',
        'chemical_treatment': 'Azoxystrobin, Chlorothalonil, or Trifloxystrobin + Tebuconazole every 10 days.',
        'organic_treatment': 'Copper-based fungicides; reduce canopy humidity through pruning.',
        'prevention': 'Stake plants; prune lower leaves; crop rotation.',
        'crop_loss_risk': '20–40 % yield loss',
        'recovery_time': '3–4 weeks',
        'loss_pct': 30,
        'hindi_treatment': 'एज़ोक्सिस्ट्रोबिन या क्लोरोथैलोनिल हर 10 दिन छिड़कें। ताँबा फफूंदनाशक भी सहायक। निचली पत्तियों की छंटाई करें।',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'display_name': 'Tomato Yellow Leaf Curl Virus',
        'hindi_name': 'टमाटर का पीला पत्ती मरोड़ विषाणु रोग',
        'disease_type': 'Viral',
        'severity_level': 'High',
        'urgency': 'Act within 3 days',
        'symptoms': 'Upward curling of leaves; yellowing of margins; stunted growth; flower drop; small fruit.',
        'chemical_treatment': 'No cure. Control whitefly vector with Imidacloprid or Thiamethoxam soil drench.',
        'organic_treatment': 'Yellow sticky traps; neem oil (5 ml/L); remove infected plants.',
        'prevention': 'Resistant varieties; insect-proof netting; systemic insecticides at transplanting.',
        'crop_loss_risk': '70–100 % yield loss if infected early in season',
        'recovery_time': 'No recovery — manage vectors; replant with resistant varieties',
        'loss_pct': 85,
        'hindi_treatment': 'कोई इलाज नहीं। सफेदमक्खी नियंत्रण के लिए इमिडाक्लोप्रिड का उपयोग करें। पीले चिपचिपे जाल लगाएँ। संक्रमित पौधे तुरंत उखाड़ें। प्रतिरोधी किस्में लगाएँ।',
    },
    'Tomato___Tomato_mosaic_virus': {
        'display_name': 'Tomato Mosaic Virus',
        'hindi_name': 'टमाटर का मोज़ेक विषाणु रोग',
        'disease_type': 'Viral',
        'severity_level': 'High',
        'urgency': 'Act within 3 days',
        'symptoms': 'Mosaic pattern of light and dark green on leaves; leaf distortion; stunted growth.',
        'chemical_treatment': 'No cure. Disinfect tools with 10 % bleach to prevent spread.',
        'organic_treatment': 'Remove and destroy infected plants; wash hands and tools frequently.',
        'prevention': 'Virus-free certified seeds; control aphids; avoid tobacco near plants.',
        'crop_loss_risk': '25–50 % yield loss; fruit quality severely affected',
        'recovery_time': 'No recovery for infected plants — prevent spread',
        'loss_pct': 37,
        'hindi_treatment': 'कोई इलाज नहीं। संक्रमित पौधे उखाड़कर नष्ट करें। औज़ार 10% ब्लीच से साफ करें। प्रमाणित विषाणु-मुक्त बीज उपयोग करें। माहू कीट नियंत्रित करें।',
    },
    'Tomato___healthy': {
        'display_name': 'Healthy Tomato Plant',
        'hindi_name': 'स्वस्थ टमाटर का पौधा',
        'disease_type': 'None',
        'severity_level': 'None',
        'urgency': 'No action needed',
        'symptoms': 'No disease detected. Plant appears healthy.',
        'chemical_treatment': 'No treatment required.',
        'organic_treatment': 'Regular composting; consistent watering; balanced NPK.',
        'prevention': 'Monitor weekly; stake plants; maintain consistent soil moisture.',
        'crop_loss_risk': 'None',
        'recovery_time': 'N/A',
        'loss_pct': 0,
        'hindi_treatment': 'कोई उपचार आवश्यक नहीं। नियमित निगरानी और संतुलित खाद-पानी जारी रखें।',
    },
}

# ═════════════════════════════════════════════════════════════════
# CROP ECONOMIC DATA  (avg yield & market price — India context)
# ═════════════════════════════════════════════════════════════════
CROP_DATA = {
    'Apple':      {'yield_kg_per_acre': 9000,  'price_inr_per_kg': 60,  'hindi': 'सेब'},
    'Blueberry':  {'yield_kg_per_acre': 3000,  'price_inr_per_kg': 900, 'hindi': 'ब्लूबेरी'},
    'Cherry':     {'yield_kg_per_acre': 4000,  'price_inr_per_kg': 300, 'hindi': 'चेरी'},
    'Corn':       {'yield_kg_per_acre': 2500,  'price_inr_per_kg': 22,  'hindi': 'मक्का'},
    'Grape':      {'yield_kg_per_acre': 10000, 'price_inr_per_kg': 55,  'hindi': 'अंगूर'},
    'Orange':     {'yield_kg_per_acre': 9000,  'price_inr_per_kg': 35,  'hindi': 'संतरा'},
    'Peach':      {'yield_kg_per_acre': 5000,  'price_inr_per_kg': 80,  'hindi': 'आड़ू'},
    'Pepper':     {'yield_kg_per_acre': 6500,  'price_inr_per_kg': 45,  'hindi': 'शिमला मिर्च'},
    'Potato':     {'yield_kg_per_acre': 10000, 'price_inr_per_kg': 18,  'hindi': 'आलू'},
    'Raspberry':  {'yield_kg_per_acre': 1500,  'price_inr_per_kg': 500, 'hindi': 'रसभरी'},
    'Soybean':    {'yield_kg_per_acre': 1000,  'price_inr_per_kg': 48,  'hindi': 'सोयाबीन'},
    'Squash':     {'yield_kg_per_acre': 8000,  'price_inr_per_kg': 18,  'hindi': 'कद्दू'},
    'Strawberry': {'yield_kg_per_acre': 4000,  'price_inr_per_kg': 200, 'hindi': 'स्ट्रॉबेरी'},
    'Tomato':     {'yield_kg_per_acre': 12000, 'price_inr_per_kg': 20,  'hindi': 'टमाटर'},
}

# ═════════════════════════════════════════════════════════════════
# WEATHER SPREAD RISK  (disease type → conditions)
# ═════════════════════════════════════════════════════════════════
DISEASE_WEATHER = {
    'Fungal':            {'ht': 70, 'tmin': 15, 'tmax': 28,
                          'desc': 'Fungal spores spread rapidly in humid, moist conditions.',
                          'cond': 'High humidity + moderate temps + wet conditions'},
    'Fungal (Oomycete)': {'ht': 80, 'tmin': 10, 'tmax': 20,
                          'desc': 'Late-blight pathogens thrive in cool, wet conditions.',
                          'cond': 'High humidity + cool temps + rainfall'},
    'Bacterial':         {'ht': 60, 'tmin': 25, 'tmax': 35,
                          'desc': 'Bacteria spread through water splash in warm, humid weather.',
                          'cond': 'Warm temps + moderate humidity + rain splash'},
    'Viral':             {'ht': 50, 'tmin': 20, 'tmax': 35,
                          'desc': 'Viruses spread via insect vectors. Warm weather increases vector activity.',
                          'cond': 'Warm dry weather (vectors active)'},
    'Pest (Arachnid)':   {'ht': 40, 'tmin': 25, 'tmax': 40,
                          'desc': 'Spider mites thrive in hot, dry, dusty conditions.',
                          'cond': 'Hot and dry weather'},
    'None':              {'ht': 101, 'tmin': -999, 'tmax': 999,
                          'desc': 'Plant is healthy. Continue monitoring.',
                          'cond': 'N/A'},
}

# ═════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════════
def predict(image_file):
    image = Image.open(image_file).convert("RGB")
    arr = np.array(image.resize((128, 128))) / 255.0
    preds = model.predict(np.expand_dims(arr, 0), verbose=0)
    idx = int(np.argmax(preds))
    return idx, float(np.max(preds)), preds[0]


# def generate_gradcam(image_pil, class_idx):
#     print(model.summary())
#     try:
#         last_conv = None
#         # for layer in reversed(model.layers):
#         #     if isinstance(layer, tf.keras.layers.Conv2D):
#         #         last_conv = layer.name
#         #         break

#         last_conv = "conv2d_2"
#         if not last_conv:
#             return None, None

#         grad_model = tf.keras.models.Model(
#             model.inputs, [model.get_layer(last_conv).output, model.output])

#         arr = np.array(image_pil.resize((128, 128))) / 255.0
#         batch = tf.cast(np.expand_dims(arr, 0), tf.float32)

#         with tf.GradientTape() as tape:
#             conv_out, preds = grad_model(batch)
#             score = preds[:, class_idx]
#         grads = tape.gradient(score, conv_out)
#         pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
#         hm = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
#         hm = np.maximum(hm, 0)
#         if hm.max() > 0:
#             hm /= hm.max()

#         h_r = cv2.resize(hm, (128, 128))
#         h_c = (cm.jet(h_r)[:, :, :3] * 255).astype(np.uint8)
#         orig = np.array(image_pil.resize((128, 128)))
#         overlay = (0.55 * orig + 0.45 * h_c).astype(np.uint8)
#         return Image.fromarray(overlay), Image.fromarray(h_c)
#     except Exception:
#         return None, None

# def generate_gradcam(image_pil, class_idx):
#     try:
#         # ✅ HARDCODED last conv layer (based on your model)
#         last_conv = "conv2d_2"

#         grad_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=[model.get_layer(last_conv).output, model.output]
#         )

#         # Preprocess image
#         arr = np.array(image_pil.resize((128, 128))) / 255.0
#         batch = np.expand_dims(arr, axis=0)

#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(batch)
#             loss = predictions[:, class_idx]

#         # Compute gradients
#         grads = tape.gradient(loss, conv_outputs)

#         # Global average pooling
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#         # Multiply feature maps
#         conv_outputs = conv_outputs[0]
#         heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#         heatmap = np.maximum(heatmap, 0)
#         if np.max(heatmap) != 0:
#             heatmap /= np.max(heatmap)

#         # Resize heatmap
#         heatmap = cv2.resize(heatmap.numpy(), (128, 128))

#         # Apply color map
#         heatmap_color = cm.jet(heatmap)[:, :, :3]
#         heatmap_color = (heatmap_color * 255).astype(np.uint8)

#         # Overlay
#         original = np.array(image_pil.resize((128, 128)))
#         overlay = (0.6 * original + 0.4 * heatmap_color).astype(np.uint8)

#         return Image.fromarray(overlay), Image.fromarray(heatmap_color)

#     except Exception as e:
#         print("Grad-CAM Error:", e)
#         return None, None

# def generate_gradcam(image_pil, class_idx):
    # try:
    #     last_conv = "conv2d_2"

    #     grad_model = tf.keras.models.Model(
    #         inputs=model.input,
    #         outputs=[model.get_layer(last_conv).output, model.output]
    #     )

    #     # Preprocess
    #     arr = np.array(image_pil.resize((128, 128))) / 255.0
    #     batch = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)

    #     with tf.GradientTape() as tape:
    #         tape.watch(batch)
    #         conv_outputs, predictions = grad_model(batch, training=False)
    #         loss = predictions[:, class_idx]

    #     grads = tape.gradient(loss, conv_outputs)

    #     # 🛑 DEBUG CHECK
    #     if grads is None:
    #         print("❌ Gradients are None")
    #         return None, None

    #     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    #     conv_outputs = conv_outputs[0]
    #     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    #     heatmap = np.maximum(heatmap, 0)
    #     if np.max(heatmap) != 0:
    #         heatmap /= np.max(heatmap)

    #     heatmap = cv2.resize(heatmap.numpy(), (128, 128))

    #     heatmap_color = cm.jet(heatmap)[:, :, :3]
    #     heatmap_color = (heatmap_color * 255).astype(np.uint8)

    #     original = np.array(image_pil.resize((128, 128)))
    #     overlay = (0.6 * original + 0.4 * heatmap_color).astype(np.uint8)

    #     return Image.fromarray(overlay), Image.fromarray(heatmap_color)

    # except Exception as e:
    #     print("Grad-CAM Error:", e)
    #     return None, None

def generate_gradcam(image_pil, class_idx):
    """
    Generate Grad-CAM++ heatmap — Keras 3 compatible, high-accuracy.

    Improvements over vanilla Grad-CAM:
      • Grad-CAM++ (alpha-weighted 2nd-order gradients) → better localization
      • Gaussian smoothing → reduces noisy activations
      • Full-resolution output → matches original image size
      • Bicubic interpolation → smoother heatmap upscaling
      • Percentile normalization → better contrast in the heatmap
    """
    try:
        # ── Auto-detect last Conv2D layer ────────────────────
        last_conv_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break
        if last_conv_name is None:
            print("[Grad-CAM++] No Conv2D layer found in model.")
            return None, None

        # ── Preprocess image ─────────────────────────────────
        img_size = (128, 128)
        arr = np.array(image_pil.resize(img_size)) / 255.0
        batch = tf.convert_to_tensor(
            np.expand_dims(arr, axis=0), dtype=tf.float32
        )

        # ── Forward pass with second-order gradients ─────────
        # Grad-CAM++ needs both first and second derivatives,
        # so we use a nested GradientTape.
        conv_output = None
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                x = batch
                for layer in model.layers:
                    x = layer(x)
                    if layer.name == last_conv_name:
                        conv_output = x
                        tape1.watch(conv_output)
                        tape2.watch(conv_output)
                predictions = x
                loss = predictions[:, class_idx]

            # First-order gradients
            grads_1st = tape1.gradient(loss, conv_output)

        if conv_output is None:
            print(f"[Grad-CAM++] Could not capture output of '{last_conv_name}'.")
            return None, None
        if grads_1st is None:
            print("[Grad-CAM++] First-order gradients are None.")
            return None, None

        # Second-order gradients
        grads_2nd = tape2.gradient(grads_1st, conv_output)

        # ── Grad-CAM++ alpha weights ─────────────────────────
        # α_kc = (∂²y / ∂A²) / (2·(∂²y/∂A²) + Σ(A · ∂³y/∂A³) + ε)
        # Simplified: use ReLU(grads_1st) weighted by grads_2nd
        if grads_2nd is not None:
            # Compute alpha weights (Grad-CAM++ formula)
            grads_2nd_val = grads_2nd.numpy()
            grads_1st_val = grads_1st.numpy()
            conv_output_val = conv_output.numpy()

            # Numerator and denominator for alpha
            numerator = grads_2nd_val
            denominator = 2.0 * grads_2nd_val + np.sum(
                conv_output_val * (grads_2nd_val ** 2 + 1e-10),  # avoid gradient explosion
                axis=(1, 2), keepdims=True
            ) + 1e-10  # avoid division by zero

            alphas = np.maximum(numerator / denominator, 0)

            # Weight = alpha * ReLU(first-order gradient)
            weights = np.sum(alphas * np.maximum(grads_1st_val, 0), axis=(1, 2))
            weights = weights[0]  # remove batch dim
        else:
            # Fallback to vanilla Grad-CAM if 2nd order fails
            print("[Grad-CAM++] 2nd-order grads unavailable, falling back to vanilla.")
            weights = tf.reduce_mean(grads_1st, axis=(0, 1, 2)).numpy()

        conv_output_val = conv_output.numpy()[0]  # (H, W, C)

        # ── Weighted combination → raw heatmap ───────────────
        heatmap = np.sum(conv_output_val * weights, axis=-1)  # (H, W)
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # ── Percentile-based normalization ────────────────────
        # Clips outliers and stretches contrast so the heatmap
        # highlights truly important regions, not just the max.
        if heatmap.max() > 0:
            p_low = np.percentile(heatmap, 2)
            p_high = np.percentile(heatmap, 98)
            if p_high > p_low:
                heatmap = np.clip(heatmap, p_low, p_high)
                heatmap = (heatmap - p_low) / (p_high - p_low)
            else:
                heatmap = heatmap / heatmap.max()

        # ── Gaussian smoothing ───────────────────────────────
        # Reduces speckle noise from individual neurons, makes
        # the attention region smoother and more interpretable.
        ksize = max(3, (heatmap.shape[0] // 4) | 1)  # odd kernel, ~25% of map
        heatmap = cv2.GaussianBlur(
            heatmap.astype(np.float32), (ksize, ksize), sigmaX=0
        )
        # Re-normalize after blur
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # ── Upscale to original image resolution ─────────────
        orig_w, orig_h = image_pil.size
        # Cap at 512 to keep the overlay within Streamlit limits
        display_size = (min(orig_w, 512), min(orig_h, 512))

        heatmap_resized = cv2.resize(
            heatmap, display_size, interpolation=cv2.INTER_CUBIC
        )

        # ── Colorize and overlay ─────────────────────────────
        heatmap_color = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        original = np.array(image_pil.resize(display_size))
        overlay = (0.55 * original + 0.45 * heatmap_color).astype(np.uint8)

        return Image.fromarray(overlay), Image.fromarray(heatmap_color)

    except Exception as e:
        import traceback
        print(f"[Grad-CAM++] Error: {e}")
        traceback.print_exc()
        return None, None
def get_weather(city: str):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=8)
        if r.status_code != 200:
            return None
        d = r.json()['current_condition'][0]
        desc = d['weatherDesc'][0]['value']
        return {
            'temp_c': int(d['temp_C']),
            'humidity': int(d['humidity']),
            'wind_kmph': int(d['windspeedKmph']),
            'desc': desc,
            'feels': int(d['FeelsLikeC']),
            'is_wet': any(w in desc.lower() for w in ('rain', 'drizzle', 'shower', 'thunder')),
        }
    except Exception:
        return None


def spread_risk(disease_type, weather):
    cfg = DISEASE_WEATHER.get(disease_type, DISEASE_WEATHER['None'])
    score, reasons = 0, []
    t, h = weather['temp_c'], weather['humidity']

    if cfg['tmin'] <= t <= cfg['tmax']:
        score += 40
        reasons.append(f"Temperature ({t} C) is optimal for this pathogen")
    else:
        reasons.append(f"Temperature ({t} C) is outside the optimal spread range")

    if h >= cfg['ht']:
        score += 40
        reasons.append(f"Humidity ({h} %) exceeds danger threshold ({cfg['ht']} %)")
    else:
        reasons.append(f"Humidity ({h} %) is below spread threshold ({cfg['ht']} %)")

    if weather['is_wet'] and disease_type in ('Fungal', 'Fungal (Oomycete)', 'Bacterial'):
        score += 20
        reasons.append("Rain/wet conditions actively spreading spores/bacteria")

    if score >= 70:
        lvl, clr = 'HIGH', 'red'
        adv = f"SPREAD ALERT: Conditions highly favorable for {disease_type.lower()} spread. Treat immediately."
    elif score >= 40:
        lvl, clr = 'MEDIUM', 'orange'
        adv = "Weather moderately favors spread. Monitor closely; consider preventive treatment."
    else:
        lvl, clr = 'LOW', 'green'
        adv = "Weather conditions less favorable for spread. Continue routine monitoring."

    return {'level': lvl, 'color': clr, 'score': score,
            'reasons': reasons, 'advice': adv,
            'conditions': cfg['cond'], 'description': cfg['desc']}


def get_crop_name(class_name):
    """Extract crop name from class name for CROP_DATA lookup."""
    raw = class_name.split('___')[0]
    mapping = {
        'Apple': 'Apple', 'Blueberry': 'Blueberry',
        'Cherry_(including_sour)': 'Cherry',
        'Corn_(maize)': 'Corn', 'Grape': 'Grape',
        'Orange': 'Orange', 'Peach': 'Peach',
        'Pepper,_bell': 'Pepper', 'Potato': 'Potato',
        'Raspberry': 'Raspberry', 'Soybean': 'Soybean',
        'Squash': 'Squash', 'Strawberry': 'Strawberry',
        'Tomato': 'Tomato',
    }
    return mapping.get(raw, raw)


# ═════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════
# def build_pdf(result, treatment, weather_data=None, risk_data=None):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=20)
#     pdf.add_page()

#     # Header
#     pdf.set_font('Helvetica', 'B', 22)
#     pdf.cell(0, 12, 'AgriGuard - Crop Disease Report', ln=True, align='C')
#     pdf.set_font('Helvetica', '', 10)
#     pdf.cell(0, 6, f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")}', ln=True, align='C')
#     pdf.cell(0, 6, 'EPICS Project - VIT Bhopal University', ln=True, align='C')
#     pdf.ln(8)

#     # Leaf image
#     try:
#         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
#             result['img_pil'].resize((256, 256)).save(f, 'PNG')
#             img_path = f.name
#         pdf.image(img_path, x=30, w=60)
#         os.unlink(img_path)
#     except Exception:
#         pass

#     # Grad-CAM image
#     try:
#         if result.get('overlay'):
#             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
#                 result['overlay'].resize((256, 256)).save(f, 'PNG')
#                 gc_path = f.name
#             pdf.image(gc_path, x=110, w=60)
#             os.unlink(gc_path)
#     except Exception:
#         pass

#     pdf.ln(65)

#     # Diagnosis
#     pdf.set_font('Helvetica', 'B', 14)
#     pdf.cell(0, 8, 'DIAGNOSIS', ln=True)
#     pdf.set_draw_color(45, 138, 78)
#     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
#     pdf.ln(4)

#     pdf.set_font('Helvetica', '', 11)
#     pdf.cell(50, 7, 'Disease:', 0)
#     pdf.set_font('Helvetica', 'B', 11)
#     pdf.cell(0, 7, treatment['display_name'], ln=True)

#     pdf.set_font('Helvetica', '', 11)
#     pdf.cell(50, 7, 'Hindi Name:', 0)
#     pdf.set_font('Helvetica', '', 11)
#     # Hindi text may not render in Helvetica – just show transliteration note
#     pdf.cell(0, 7, f"[See Hindi translation in the app]", ln=True)

#     pdf.set_font('Helvetica', '', 11)
#     fields = [
#         ('Type', treatment['disease_type']),
#         ('Severity', treatment['severity_level']),
#         ('Urgency', treatment['urgency']),
#         ('AI Confidence', f"{result['confidence']:.1%}"),
#         ('Crop Loss Risk', treatment['crop_loss_risk']),
#         ('Recovery Time', treatment['recovery_time']),
#     ]
#     for label, value in fields:
#         pdf.cell(50, 7, f'{label}:', 0)
#         pdf.set_font('Helvetica', 'B', 11)
#         pdf.cell(0, 7, str(value), ln=True)
#         pdf.set_font('Helvetica', '', 11)

#     pdf.ln(6)

#     # Treatment
#     pdf.set_font('Helvetica', 'B', 14)
#     pdf.cell(0, 8, 'TREATMENT PLAN', ln=True)
#     pdf.set_draw_color(45, 138, 78)
#     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
#     pdf.ln(4)

#     sections = [
#         ('Symptoms', treatment['symptoms']),
#         ('Chemical Treatment', treatment['chemical_treatment']),
#         ('Organic Treatment', treatment['organic_treatment']),
#         ('Prevention', treatment['prevention']),
#     ]
#     for title, text in sections:
#         pdf.set_font('Helvetica', 'B', 11)
#         pdf.cell(0, 7, title, ln=True)
#         pdf.set_font('Helvetica', '', 10)
#         pdf.multi_cell(0, 5, text)
#         pdf.ln(3)

#     # Weather
#     if weather_data and risk_data:
#         pdf.ln(4)
#         pdf.set_font('Helvetica', 'B', 14)
#         pdf.cell(0, 8, 'WEATHER RISK ASSESSMENT', ln=True)
#         pdf.set_draw_color(45, 138, 78)
#         pdf.line(10, pdf.get_y(), 200, pdf.get_y())
#         pdf.ln(4)
#         pdf.set_font('Helvetica', '', 11)
#         pdf.cell(0, 7, f"Location: {weather_data.get('city','')}", ln=True)
#         pdf.cell(0, 7, f"Temperature: {weather_data['temp_c']} C  |  Humidity: {weather_data['humidity']}%  |  {weather_data['desc']}", ln=True)
#         pdf.set_font('Helvetica', 'B', 11)
#         pdf.cell(0, 7, f"Spread Risk: {risk_data['level']}", ln=True)
#         pdf.set_font('Helvetica', '', 10)
#         pdf.multi_cell(0, 5, risk_data['advice'])



#     # return pdf.output()
#     return pdf.output(dest='S').encode('latin-1', 'replace')


# def whatsapp_share_text(treatment, confidence, weather_data=None, risk_data=None):
#     lines = [
#         "--- AgriGuard Report ---",
#         f"Disease: {treatment['display_name']}",
#         f"Hindi: {treatment['hindi_name']}",
#         f"Type: {treatment['disease_type']}",
#         f"Severity: {treatment['severity_level']}",
#         f"AI Confidence: {confidence:.1%}",
#         f"Crop Loss Risk: {treatment['crop_loss_risk']}",
#         "",
#         f"Treatment: {treatment['chemical_treatment']}",
#         "",
#         f"Organic Option: {treatment['organic_treatment']}",
#     ]
#     if weather_data and risk_data:
#         lines += [
#             "",
#             f"Weather ({weather_data.get('city','')}): {weather_data['temp_c']}C, {weather_data['humidity']}% humidity",
#             f"Spread Risk: {risk_data['level']}",
#         ]
#     lines += ["", "Generated by AgriGuard | EPICS - VIT Bhopal"]
#     return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR (FIXED - NO FUNCTIONAL CHANGE)
# ═════════════════════════════════════════════════════════════════
def build_pdf(result, treatment, weather_data=None, risk_data=None):

    # 🔹 Clean text BEFORE sending to FPDF (prevents Unicode crash)
    def clean_text(text):
        return (
            str(text)
            .replace("–", "-")
            .replace("—", "-")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("•", "-")
            .replace("°", " deg")
        )

    # 🔹 Safe wrapper around FPDF (auto-cleans everything)
    class SafeFPDF(FPDF):
        def cell(self, *args, **kwargs):
            if len(args) >= 3:
                args = list(args)
                args[2] = clean_text(args[2])
            return super().cell(*args, **kwargs)

        def multi_cell(self, *args, **kwargs):
            if len(args) >= 3:
                args = list(args)
                args[2] = clean_text(args[2])
            return super().multi_cell(*args, **kwargs)

    pdf = SafeFPDF()   # ✅ only change (Safe wrapper)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Header
    pdf.set_font('Helvetica', 'B', 22)
    pdf.cell(0, 12, 'AgriGuard - Crop Disease Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")}', ln=True, align='C')
    pdf.cell(0, 6, 'AgriGuard - Crop Disease Intelligence', ln=True, align='C')
    pdf.ln(8)

    # Leaf image
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            result['img_pil'].resize((256, 256)).save(f, 'PNG')
            img_path = f.name
        pdf.image(img_path, x=30, w=60)
        os.unlink(img_path)
    except Exception:
        pass

    # Grad-CAM image
    try:
        if result.get('overlay'):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                result['overlay'].resize((256, 256)).save(f, 'PNG')
                gc_path = f.name
            pdf.image(gc_path, x=110, w=60)
            os.unlink(gc_path)
    except Exception:
        pass

    pdf.ln(65)

    # Diagnosis
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, 'DIAGNOSIS', ln=True)
    pdf.set_draw_color(45, 138, 78)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.cell(50, 7, 'Disease:', 0)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, treatment['display_name'], ln=True)

    pdf.set_font('Helvetica', '', 11)
    pdf.cell(50, 7, 'Hindi Name:', 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, "[See Hindi translation in the app]", ln=True)

    pdf.set_font('Helvetica', '', 11)
    fields = [
        ('Type', treatment['disease_type']),
        ('Severity', treatment['severity_level']),
        ('Urgency', treatment['urgency']),
        ('AI Confidence', f"{result['confidence']:.1%}"),
        ('Crop Loss Risk', treatment['crop_loss_risk']),
        ('Recovery Time', treatment['recovery_time']),
    ]
    for label, value in fields:
        pdf.cell(50, 7, f'{label}:', 0)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, str(value), ln=True)
        pdf.set_font('Helvetica', '', 11)

    pdf.ln(6)

    # Treatment
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, 'TREATMENT PLAN', ln=True)
    pdf.set_draw_color(45, 138, 78)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    sections = [
        ('Symptoms', treatment['symptoms']),
        ('Chemical Treatment', treatment['chemical_treatment']),
        ('Organic Treatment', treatment['organic_treatment']),
        ('Prevention', treatment['prevention']),
    ]
    for title, text in sections:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, text)
        pdf.ln(3)

    # Weather
    if weather_data and risk_data:
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, 'WEATHER RISK ASSESSMENT', ln=True)
        pdf.set_draw_color(45, 138, 78)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 7, f"Location: {weather_data.get('city','')}", ln=True)
        pdf.cell(0, 7, f"Temperature: {weather_data['temp_c']} C  |  Humidity: {weather_data['humidity']}%  |  {weather_data['desc']}", ln=True)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, f"Spread Risk: {risk_data['level']}", ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, risk_data['advice'])



    return pdf.output(dest='S').encode('latin-1', 'replace')

def whatsapp_share_text(treatment, confidence, weather_data=None, risk_data=None):
    lines = [
        "--- AgriGuard Report ---",
        f"Disease: {treatment['display_name']}",
        f"Hindi: {treatment['hindi_name']}",
        f"Type: {treatment['disease_type']}",
        f"Severity: {treatment['severity_level']}",
        f"AI Confidence: {confidence:.1%}",
        f"Crop Loss Risk: {treatment['crop_loss_risk']}",
        "",
        f"Treatment: {treatment['chemical_treatment']}",
        "",
        f"Organic Option: {treatment['organic_treatment']}",
    ]

    if weather_data and risk_data:
        lines += [
            "",
            f"Weather ({weather_data.get('city','')}): {weather_data['temp_c']}C, {weather_data['humidity']}% humidity",
            f"Spread Risk: {risk_data['level']}",
        ]

    lines += ["", "Generated by AgriGuard"]

    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════
# KVK LOCATOR — data loader + search
# ═════════════════════════════════════════════════════════════════
@st.cache_data
def load_kvk_data():
    """Load and parse kvk_centers.csv into a structured list."""
    kvks = []
    try:
        with open('kvk_centers.csv', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                addr_raw  = row.get('kvk_address', '')
                host_raw  = row.get('host_organization', '')
                state     = row.get('state_or_ut', '').strip()

                # Clean multi-line pipe-separated address
                addr_parts = [p.strip().rstrip(',') for p in addr_raw.split('|') if p.strip()]
                host_parts = [p.strip().rstrip(',') for p in host_raw.split('|') if p.strip()]

                # Extract district from address
                district = ''
                for part in addr_parts:
                    m = re.search(r'Dis[t]{1,2}[.\s]+(.+?)[\-\d]', part, re.IGNORECASE)
                    if m:
                        district = m.group(1).strip().rstrip(',').strip()
                        break
                if not district:
                    # fallback: last non-KVK part
                    for part in reversed(addr_parts):
                        if part and 'Vigyan' not in part and 'Kendra' not in part and len(part) > 3:
                            district = part.split('-')[0].strip().rstrip(',')
                            break

                # Build readable address (skip the first "Krishi Vigyan Kendra" line)
                display_addr = ', '.join(
                    p for p in addr_parts
                    if not re.match(r'[Kk]r?i?s?h?i?\s*[Vv]igyan\s*[Kk]endra', p)
                ) or ', '.join(addr_parts)

                host_clean = host_parts[-1] if host_parts else host_raw

                kvks.append({
                    'state':    state,
                    'district': district,
                    'address':  display_addr,
                    'host':     host_clean,
                    'serial':   row.get('serial_no', ''),
                })
    except FileNotFoundError:
        pass
    return kvks


def search_kvk(kvks, state_query, district_query):
    """Return best-matching KVKs for the given state + district."""
    state_q    = state_query.strip().lower()
    district_q = district_query.strip().lower()

    # Step 1 – filter by state (fuzzy, threshold 0.6)
    all_states = list({k['state'] for k in kvks})
    state_matches = difflib.get_close_matches(
        state_q, [s.lower() for s in all_states], n=3, cutoff=0.5
    )
    if state_matches:
        state_filtered = [k for k in kvks if k['state'].lower() in state_matches]
    else:
        # broad contains fallback
        state_filtered = [k for k in kvks if state_q in k['state'].lower()]

    if not state_filtered:
        state_filtered = kvks  # don't cut results to zero

    if not district_q:
        return state_filtered[:10]

    # Step 2 – rank by district similarity
    def score(k):
        d = k['district'].lower()
        if district_q == d:
            return 1.0
        if district_q in d or d in district_q:
            return 0.85
        ratio = difflib.SequenceMatcher(None, district_q, d).ratio()
        return ratio

    ranked = sorted(state_filtered, key=score, reverse=True)
    # Return top results with score > 0.3
    top = [k for k in ranked if score(k) > 0.3]
    return top[:8] if top else ranked[:5]


KVK_DATA = load_kvk_data()
KVK_STATES = sorted({k['state'] for k in KVK_DATA})

# ═════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    try:
        st.image("home_page.jpeg", use_container_width=True)
    except Exception:
        pass
    st.markdown("## AgriGuard")
    st.caption("Crop Disease Intelligence")
    st.markdown("---")
    st.markdown("### Mission")
    st.markdown(
        "Empowering farmers with AI-powered crop disease detection, "
        "treatment guidance, and economic impact awareness — bridging "
        "the gap between agricultural expertise and rural communities."
    )
    st.markdown("---")
    ca, cb = st.columns(2)
    ca.metric("Accuracy", "94.61%")
    cb.metric("Diseases", "38 Classes")
    st.metric("Training Images", "87,000+")
    st.markdown("---")
    show_hindi = st.toggle("Show Hindi translations", value=True)
    st.markdown("---")
    st.caption("Developer: Apoorv Aditya Tripathi")

# ═════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════
for key in ('result', 'weather', 'risk'):
    if key not in st.session_state:
        st.session_state[key] = None

# ═════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Detect Disease",
    "Treatment Plan",
    "Weather Risk",
    "Economic Impact",
    "KVK Locator",
    "About",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION + GRAD-CAM
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.header("Crop Disease Detection")
    st.markdown("Upload a clear photo of a plant leaf for AI diagnosis with visual explanation.")
    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Leaf", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing your leaf..."):
                    idx, conf, probs = predict(uploaded)
                    uploaded.seek(0)
                    pil = Image.open(uploaded).convert("RGB")
                    ov, hm = generate_gradcam(pil, idx)
                    st.session_state.result = {
                        'class_idx': idx, 'class_name': CLASS_NAMES[idx],
                        'confidence': conf, 'all_probs': probs,
                        'overlay': ov, 'heatmap': hm, 'img_pil': pil,
                    }

    with right:
        r = st.session_state.result
        if r:
            t = TREATMENT_DB.get(r['class_name'], {})
            name = t.get('display_name', r['class_name'])
            sev = t.get('severity_level', '-')
            ok = 'healthy' in r['class_name'].lower()

            if ok:
                st.success(f"**{name}** — Plant is Healthy!")
            elif sev == 'Critical':
                st.error(f"**{name}** — CRITICAL SEVERITY")
            elif sev == 'High':
                st.warning(f"**{name}** — HIGH SEVERITY")
            else:
                st.info(f"**{name}**")

            if show_hindi and t.get('hindi_name'):
                st.markdown(f'<div class="hindi-box"><b>हिंदी:</b> {t["hindi_name"]}</div>',
                            unsafe_allow_html=True)

            st.markdown(f"**AI Confidence: {r['confidence']:.1%}**")
            st.progress(r['confidence'])

            st.markdown("---")
            st.markdown("#### AI Attention Map (Grad-CAM++)")
            st.caption("Red/warm areas = what the AI focused on to diagnose this disease")
            if r['overlay']:
                c1, c2, c3 = st.columns(3)
                # Show original at same display size as the overlay
                disp = r['overlay'].size  # overlay is already at display_size
                c1.image(r['img_pil'].resize(disp), caption="Original", use_container_width=True)
                c2.image(r['overlay'], caption="AI Attention", use_container_width=True)
                c3.image(r['heatmap'], caption="Heatmap", use_container_width=True)
            else:
                st.warning("Grad-CAM could not be generated for this model format.")

            st.markdown("---")
            st.markdown("#### Top 3 Predictions")
            for i in np.argsort(r['all_probs'])[-3:][::-1]:
                lbl = CLASS_NAMES[i].replace('___', ' > ').replace('_', ' ')
                st.markdown(f"**{lbl}**: {r['all_probs'][i]:.1%}")
                st.progress(float(r['all_probs'][i]))
        else:
            st.info("Upload an image and click **Analyze Leaf** to see results")
            st.markdown("""
**How Grad-CAM++ works:**
Traditional AI is a black box. Grad-CAM++ makes it transparent by showing
*which pixels* the model used to make its prediction — with improved
spatial accuracy over standard Grad-CAM.
- Red/yellow = high attention (disease symptoms)
- Blue = low attention (healthy tissue / background)
            """)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — TREATMENT + HINDI + PDF + WHATSAPP
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.header("Treatment & Action Plan")
    r = st.session_state.result

    if not r:
        st.info("Run a detection first in the **Detect Disease** tab.")
    else:
        t = TREATMENT_DB.get(r['class_name'])
        if not t:
            st.warning("No treatment data for this class.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Disease", t['display_name'])
            c2.metric("Type", t['disease_type'])
            c3.metric("Severity", t['severity_level'])
            c4.metric("Urgency", t['urgency'])

            # Hindi treatment
            if show_hindi:
                st.markdown(f"""
<div class="hindi-box">
<b>हिंदी में उपचार:</b><br>{t.get('hindi_treatment', '')}
</div>""", unsafe_allow_html=True)

            st.markdown("---")

            with st.expander("Symptoms", expanded=True):
                st.markdown(t['symptoms'])

            ca, cb = st.columns(2)
            with ca:
                with st.expander("Chemical Treatment", expanded=True):
                    st.markdown(t['chemical_treatment'])
            with cb:
                with st.expander("Organic / Natural Treatment", expanded=True):
                    st.markdown(t['organic_treatment'])

            with st.expander("Prevention (Future Seasons)", expanded=True):
                st.markdown(t['prevention'])

            cc, cd = st.columns(2)
            cc.error(f"**Crop Loss Risk:** {t['crop_loss_risk']}")
            cd.info(f"**Recovery Time:** {t['recovery_time']}")

            # ── PDF & WHATSAPP ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### Share & Download")

            share_cols = st.columns(2)
            with share_cols[0]:
                pdf_bytes = build_pdf(
                    r, t,
                    weather_data=st.session_state.weather,
                    risk_data=st.session_state.risk,
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"AgriGuard_{t['display_name'].replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            with share_cols[1]:
                wa_text = whatsapp_share_text(
                    t, r['confidence'],
                    st.session_state.weather,
                    st.session_state.risk,
                )
                wa_url = f"https://wa.me/?text={urllib.parse.quote(wa_text)}"
                st.link_button(
                    "Share via WhatsApp",
                    url=wa_url,
                    use_container_width=True,
                )
            st.caption("Download the PDF and attach it in the WhatsApp chat for a complete report.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — WEATHER SPREAD RISK
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.header("Weather-Based Disease Spread Risk")
    st.markdown("Enter your location to check if current weather favours disease spread.")

    city = st.text_input("Enter your city / district",
                         placeholder="e.g., Bhopal, Pune, Nagpur, Lucknow")

    if city and st.button("Get Weather Risk", type="primary"):
        with st.spinner(f"Fetching weather for {city}..."):
            w = get_weather(city)

        if not w:
            st.error(f"Could not fetch weather for '{city}'. Check spelling and retry.")
        else:
            w['city'] = city
            st.session_state.weather = w
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{w['temp_c']} C", f"Feels {w['feels']} C")
            c2.metric("Humidity", f"{w['humidity']} %")
            c3.metric("Wind", f"{w['wind_kmph']} km/h")
            c4.metric("Condition", w['desc'])

            st.markdown("---")
            r = st.session_state.result
            if r:
                t = TREATMENT_DB.get(r['class_name'], {})
                dtype = t.get('disease_type', 'None')
                dname = t.get('display_name', r['class_name'])
                risk = spread_risk(dtype, w)
                st.session_state.risk = risk

                st.markdown(f"### Spread Risk for: **{dname}** ({dtype})")
                rc = {'HIGH': 'risk-high', 'MEDIUM': 'risk-medium', 'LOW': 'risk-low'}[risk['level']]
                emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[risk['level']]
                st.markdown(f"""
<div class="disease-card {rc}">
  <h3>{emoji} Spread Risk: {risk['level']}</h3>
  <p>{risk['advice']}</p>
</div>""", unsafe_allow_html=True)

                st.markdown("#### Why this risk level?")
                for reason in risk['reasons']:
                    st.markdown(f"- {reason}")
                st.info(f"**About {dtype}:** {risk['description']}")
                st.caption(f"Dangerous conditions: *{risk['conditions']}*")
            else:
                st.warning("Run a leaf detection first to get disease-specific spread risk.")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — ECONOMIC IMPACT CALCULATOR
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.header("Economic Impact Calculator")
    st.markdown("Estimate potential crop loss in **kg** and **INR** based on the detected disease.")

    r = st.session_state.result
    if not r:
        st.info("Run a detection first in the **Detect Disease** tab.")
    else:
        t = TREATMENT_DB.get(r['class_name'], {})
        crop_key = get_crop_name(r['class_name'])
        crop = CROP_DATA.get(crop_key)
        loss_pct = t.get('loss_pct', 0)

        if not crop:
            st.warning(f"No economic data for crop: {crop_key}")
        else:
            st.markdown(f"**Detected Disease:** {t['display_name']}  |  **Crop:** {crop_key}")
            if show_hindi:
                st.markdown(f'<div class="hindi-box"><b>फसल:</b> {crop["hindi"]}  |  <b>रोग:</b> {t["hindi_name"]}</div>',
                            unsafe_allow_html=True)

            area = st.number_input(
                "Enter your cultivated area (in acres)",
                min_value=0.1, max_value=10000.0, value=1.0, step=0.5,
            )

            if loss_pct == 0:
                st.success("Your plant is healthy! No estimated economic loss.")
                total_yield = crop['yield_kg_per_acre'] * area
                total_value = total_yield * crop['price_inr_per_kg']
                st.markdown(f"**Expected Healthy Yield:** {total_yield:,.0f} kg")
                st.markdown(f"**Expected Revenue:** Rs. {total_value:,.0f}")
            else:
                total_yield = crop['yield_kg_per_acre'] * area
                total_value = total_yield * crop['price_inr_per_kg']
                lost_yield = total_yield * (loss_pct / 100)
                lost_value = lost_yield * crop['price_inr_per_kg']
                saved_yield = total_yield - lost_yield
                saved_value = saved_yield * crop['price_inr_per_kg']

                st.markdown("---")

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Healthy Yield", f"{total_yield:,.0f} kg", f"Rs. {total_value:,.0f}")
                mc2.metric("Estimated Loss", f"{lost_yield:,.0f} kg",
                           f"- Rs. {lost_value:,.0f}", delta_color="inverse")
                mc3.metric("Remaining (if treated)", f"{saved_yield:,.0f} kg",
                           f"Rs. {saved_value:,.0f}")

                st.markdown("---")

                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Crop | {crop_key} ({crop['hindi']}) |
| Area | {area:.1f} acres |
| Avg Yield / Acre | {crop['yield_kg_per_acre']:,} kg |
| Market Price | Rs. {crop['price_inr_per_kg']} / kg |
| Disease Loss Factor | {loss_pct} % |
| **Total Potential Loss** | **Rs. {lost_value:,.0f}** ({lost_yield:,.0f} kg) |
                """)

                if show_hindi:
                    st.markdown(f"""
<div class="hindi-box">
<b>अनुमानित नुकसान:</b> {lost_yield:,.0f} किलो फसल ({loss_pct}% हानि)<br>
<b>आर्थिक नुकसान:</b> लगभग Rs. {lost_value:,.0f}<br>
<b>सलाह:</b> तुरंत उपचार करें ताकि शेष Rs. {saved_value:,.0f} की फसल बचाई जा सके।
</div>""", unsafe_allow_html=True)

                st.warning(
                    f"**Without treatment, you could lose approximately Rs. {lost_value:,.0f} "
                    f"worth of {crop_key} crop across {area:.1f} acres.** "
                    f"Early treatment can save up to Rs. {saved_value:,.0f}."
                )

# ══════════════════════════════════════════════════════════════════
# TAB 5 — KVK LOCATOR
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.header("Nearest Krishi Vigyan Kendra (KVK)")
    st.markdown(
        "Find your nearest government agricultural help centre. "
        "KVK officers provide free expert advice on crop diseases, pesticide use, and farming practices."
    )

    if not KVK_DATA:
        st.error("KVK data file not found. Please ensure `kvk_centers.csv` is in the project folder.")
    else:
        st.caption(f"Database: {len(KVK_DATA)} KVKs across {len(KVK_STATES)} States / UTs")

        col_s, col_d = st.columns(2)
        with col_s:
            selected_state = st.selectbox(
                "Select your State / UT",
                options=[""] + KVK_STATES,
                format_func=lambda x: "— Select State —" if x == "" else x,
            )
        with col_d:
            district_input = st.text_input(
                "Enter your District",
                placeholder="e.g., Pune, Nashik, Bhopal, Varanasi",
            )

        search_clicked = st.button("Find KVKs", type="primary", use_container_width=True)

        if search_clicked or (selected_state and district_input):
            if not selected_state and not district_input:
                st.warning("Please select a state or enter a district.")
            else:
                results = search_kvk(KVK_DATA, selected_state, district_input)

                if not results:
                    st.error("No KVKs found for this location. Try a broader search.")
                else:
                    r_det = st.session_state.result
                    if r_det:
                        t_det = TREATMENT_DB.get(r_det['class_name'], {})
                        st.info(
                            f"Detected disease: **{t_det.get('display_name', '')}** "
                            f"({t_det.get('severity_level', '')} severity) — "
                            "Show this page to the KVK officer for expert guidance."
                        )
                        if show_hindi and t_det.get('hindi_name'):
                            st.markdown(
                                f'<div class="hindi-box">KVK अधिकारी को दिखाएँ: '
                                f'<b>{t_det["hindi_name"]}</b> का पता चला है।</div>',
                                unsafe_allow_html=True,
                            )

                    st.markdown(f"### {len(results)} KVK(s) found")
                    for i, kvk in enumerate(results, 1):
                        with st.expander(
                            f"KVK #{i} — {kvk['district'] or 'N/A'}, {kvk['state']}",
                            expanded=(i == 1),
                        ):
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                st.markdown(f"**Address:**  \n{kvk['address']}")
                                st.markdown(f"**Host Organisation:**  \n{kvk['host']}")
                            with c2:
                                st.markdown(f"**State:** {kvk['state']}")
                                st.markdown(f"**District:** {kvk['district'] or '—'}")
                                # Google Maps search link
                                maps_q = urllib.parse.quote(
                                    f"Krishi Vigyan Kendra {kvk['district']} {kvk['state']} India"
                                )
                                st.link_button(
                                    "Search on Google Maps",
                                    f"https://www.google.com/maps/search/?api=1&query={maps_q}",
                                )

        st.markdown("---")
        st.markdown(
            "**What is a KVK?**  \n"
            "Krishi Vigyan Kendras are farm science centres established by ICAR across India. "
            "They offer free consultation, soil testing, seed distribution, and training to farmers. "
            "Visit your nearest KVK with the AgriGuard PDF report for immediate expert help."
        )
        if show_hindi:
            st.markdown(
                '<div class="hindi-box">'
                '<b>KVK क्या है?</b> कृषि विज्ञान केंद्र (KVK) ICAR द्वारा स्थापित कृषि सहायता केंद्र हैं। '
                'यहाँ किसान मुफ्त में मिट्टी परीक्षण, बीज, और कृषि सलाह प्राप्त कर सकते हैं।'
                '</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.header("About AgriGuard")
    st.markdown("""
### The Problem

India has **140 million farming households**. Most lack access to agronomists.
A delayed or wrong disease diagnosis can destroy an entire season's crop,
pushing families deeper into debt. AgriGuard bridges this gap with AI.

---

### Technology

| Component | Details |
|-----------|---------|
| Model | Custom CNN — 7.8 M parameters, 10 Conv layers |
| Dataset | PlantVillage (~87,000 images, 38 disease classes) |
| Accuracy | 94.61 % validation accuracy |
| Explainability | Grad-CAM attention visualization |
| Weather | Real-time data via wttr.in (no API key) |
| Languages | English + Hindi |

---

### What AgriGuard Does

**Detect** — Upload a leaf photo; AI predicts the disease in seconds.

**Explain** — Grad-CAM shows *exactly which part of the leaf* the AI analysed.

**Treat** — Curated chemical + organic treatment options for all 38 classes (English & Hindi).

**Assess Weather** — Real-time weather analysis shows spread risk.

**Calculate Loss** — Estimated crop loss in kg and INR per acre so farmers understand urgency.

**Share** — Download PDF report or share via WhatsApp for consultation.

---

### AgriGuard

**Crop Disease Intelligence**

Developer: Apoorv Aditya Tripathi
    """)
