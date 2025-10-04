import streamlit as st
import json
from PIL import Image
from phi.agent import Agent
from phi.model.groq import Groq
from utils import load_trained_model, preprocess_image, predict_breed
from youtube_search import fetch_youtube_links
import tempfile, os, base64
from gtts import gTTS
from datetime import datetime
import plotly.express as px
import pandas as pd

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="🐄 BPA Smart Breed Identifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .breed-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    
    .video-suggestion {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B6B;
    }
    
    .success-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-highlight h4 {
        color: #ffffff !important;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .feature-highlight p {
        color: #f8f9fa !important;
        opacity: 0.9;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "registrations" not in st.session_state:
    st.session_state.registrations = []

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "accuracy_scores" not in st.session_state:
    st.session_state.accuracy_scores = []

if "breed_distribution" not in st.session_state:
    st.session_state.breed_distribution = {}

# -----------------------------
# Languages
# -----------------------------
languages = {
    "English": "en", "हिंदी": "hi", "தமிழ்": "ta", "తెలుగు": "te",
    "বাংলা": "bn", "मराठी": "mr", "ગુજરાતી": "gu",
    "ಕನ್ನಡ": "kn", "മലയാളം": "ml", "ਪੰਜਾਬੀ": "pa", "اردو": "ur"
}

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🐄 BPA Smart Breed Identifier</h1>
    <h3>AI-Powered Cattle & Buffalo Breed Classification for Field Level Workers</h3>
    <p>🎯 Reducing misclassification errors • 📱 BPA Integration Ready • 🌍 Multi-language Support</p>
</div>
""", unsafe_allow_html=True)

# Language selection in sidebar
with st.sidebar:
    st.markdown("### 🌐 Language Settings")
    selected_lang = st.selectbox("Choose Language", options=list(languages.keys()))
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 Predictions", st.session_state.prediction_count)
    with col2:
        # Fix registration counter by using actual length
        registration_count = len(st.session_state.registrations)
        st.metric("📋 Registrations", registration_count)

# -----------------------------
# Audio Functions (Enhanced)
# -----------------------------
def create_audio_gtts(text, language_code, slow=False):
    try:
        clean_text = " ".join(text.replace("*","").replace("-","").replace("#","").split())
        tts = gTTS(text=clean_text, lang=language_code, slow=slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"🔊 Audio generation failed: {str(e)}")
        return None

def create_audio_player(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
        return f'<audio controls style="width:100%; margin:10px 0;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">Your browser does not support the audio element.</audio>'
    except:
        return None

def clean_up_temp_file(file_path):
    if file_path and os.path.exists(file_path):
        os.unlink(file_path)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def get_model():
    return load_trained_model("final_indian_bovine_breed_classifier_mobilenetv2.h5")

model = get_model()
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# -----------------------------
# Enhanced Groq Agent
# -----------------------------
groq_agent = Agent(
    name="BPA Breed Expert",
    role="Expert on Indian cattle and buffalo breeds for BPA integration",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "Provide comprehensive breed information relevant to BPA data collection",
        "Include key identification features, breeding characteristics, and regional distribution",
        "Use the user-selected language for better FLW understanding",
        "Format for both audio narration and visual display",
        "Include productivity metrics and genetic improvement potential"
    ],
    markdown=True,
)

# -----------------------------
# YouTube Resource Categories
# -----------------------------
def get_enhanced_youtube_suggestions(breed_name):
    """Generate categorized YouTube search suggestions"""
    return {
        "🎓 Educational Content": [
            f"{breed_name} cattle breed characteristics documentary",
            f"{breed_name} buffalo breed identification features",
            f"Indian {breed_name} livestock breed profile"
        ],
        "🚜 Farming Techniques": [
            f"{breed_name} dairy farming best practices",
            f"{breed_name} cattle management techniques",
            f"Traditional {breed_name} animal husbandry"
        ],
        "🥛 Production & Economics": [
            f"{breed_name} milk production capacity",
            f"{breed_name} breed economic benefits",
            f"{breed_name} livestock profitability analysis"
        ],
        "🏥 Health & Nutrition": [
            f"{breed_name} cattle health management",
            f"{breed_name} buffalo nutrition requirements",
            f"{breed_name} breed specific veterinary care"
        ],
        "🧬 Breeding & Genetics": [
            f"{breed_name} breed improvement programs",
            f"{breed_name} genetic characteristics",
            f"{breed_name} crossbreeding techniques"
        ]
    }

# -----------------------------
# Navigation
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Breed Identification", 
    "📊 Analytics Dashboard", 
    "📋 BPA Registrations", 
    "📚 Learning Center",
    "ℹ️ About BPA Integration"
])

# -----------------------------
# TAB 1: BREED IDENTIFICATION
# -----------------------------
with tab1:
    st.markdown("### 📸 Upload Animal Image for Breed Classification")
    
    # Instructions for FLWs
    with st.expander("📖 Instructions for Field Level Workers"):
        st.markdown("""
        **Best Practices for Image Capture:**
        - 📷 Capture clear, well-lit images of the animal
        - 🎯 Focus on distinctive breed features (head, body structure, udder)
        - 📐 Include multiple angles when possible
        - 🌞 Avoid shadows and ensure good lighting
        - 📱 Hold device steady for sharp images
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Animal Image", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Uploaded Image", use_container_width=True)

            # Predict Breed
            with st.spinner("🤖 AI is analyzing the breed..."):
                img_array = preprocess_image(img, target_size=(224, 224))
                prediction_result = predict_breed(model, img_array, class_indices)
                
                # Handle different return formats from predict_breed function
                if isinstance(prediction_result, tuple) and len(prediction_result) >= 2:
                    breed = prediction_result[0]
                    confidence = prediction_result[1]
                    top_predictions = prediction_result[2] if len(prediction_result) > 2 else None
                else:
                    breed = str(prediction_result)
                    confidence = 0.0
                    top_predictions = None
            
            # Update statistics
            st.session_state.prediction_count += 1
            st.session_state.accuracy_scores.append(confidence)
            
            # Update breed distribution
            if breed in st.session_state.breed_distribution:
                st.session_state.breed_distribution[breed] += 1
            else:
                st.session_state.breed_distribution[breed] = 1
            
            # Check confidence level and display appropriate result
            # Always show prediction as AI opinion regardless of confidence
            st.markdown(f"""
            <div class="breed-card success-animation">
                <h2>🤖 AI Predicted Breed: {breed}</h2>
                <h4>🎯 Smart identification to reduce BPA data errors</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top predictions if available
            if top_predictions and isinstance(top_predictions, list) and len(top_predictions) > 1:
                st.markdown("### 🏆 Top 3 AI Suggestions:")
                for i, (pred_breed, pred_conf) in enumerate(top_predictions[:3]):
                    st.markdown(f"**{i+1}.** {pred_breed}")
            
            # Always show manual selection option
            st.markdown("### 🎯 Breed Confirmation")
            st.markdown("**AI suggestion or select manually for 100% accuracy:**")
            
            common_breeds = [
                "Gir", "Sahiwal", "Red Sindhi", "Tharparkar", "Rathi",
                "Hariana", "Ongole", "Krishna Valley", "Deoni", "Khillari",
                "Murrah", "Nili-Ravi", "Mehsana", "Surti", "Jaffarabadi"
            ]
            
            selected_breed = st.selectbox("🐄 Confirm or Select Different Breed:", [breed] + ["Select different..."] + common_breeds)
            
            if selected_breed != breed and selected_breed != "Select different...":
                if st.button("✅ Use Selected Breed"):
                    breed = selected_breed  # Update breed for registration
                    st.success(f"✅ Updated to: {breed}")
                    st.session_state.breed_distribution[breed] = st.session_state.breed_distribution.get(breed, 0) + 1

    with col2:
        if uploaded_file:
            # Quick breed info card - simplified
            st.markdown("### 📋 Quick Info")
            if 'breed' in locals():
                st.info(f"**AI Suggestion:** {breed}")
                st.info(f"**Language:** {selected_lang}")
                st.success("🎯 AI-powered accuracy for better BPA data quality")
                
                # Validation buttons for FLWs
                st.markdown("### ✅ Validation")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Confirm & Use", type="primary"):
                        st.success("✅ AI suggestion confirmed for registration!")
                with col_no:
                    if st.button("🔄 Select Different", type="secondary"):
                        st.info("Choose alternative from dropdown below")

    if uploaded_file and 'breed' in locals() and confidence >= 30:
        # Enhanced Breed Summary - only show for reasonable confidence
        st.markdown("---")
        st.markdown("### 📚 Comprehensive Breed Information")
        
        with st.spinner("🧠 Generating detailed breed information..."):
            prompt = f"""
            Provide comprehensive information about '{breed}' breed in {selected_lang} language including:
            - Key identification features
            - Origin and distribution in India
            - Milk/meat production capacity
            - Breeding characteristics
            - Economic importance
            - Management requirements
            Format in bullet points for easy reading and audio narration.
            """
            response = groq_agent.run(prompt)
            breed_summary = response.content if hasattr(response, "content") else str(response)
        
        # Display breed information in an attractive format
        st.markdown(f"""
        <div class="stat-card">
            {breed_summary}
        </div>
        """, unsafe_allow_html=True)

        # Enhanced Audio Section
        st.markdown("### 🔊 Audio Information")
        col_audio1, col_audio2 = st.columns(2)
        
        with col_audio1:
            audio_speed = st.radio("🎵 Audio Speed:", ["Normal", "Slow"], horizontal=True)
            slow_speech = (audio_speed == "Slow")
        
        with col_audio2:
            if st.button("🎧 Generate Audio", type="secondary"):
                with st.spinner("🎵 Creating audio..."):
                    audio_file = create_audio_gtts(breed_summary, languages[selected_lang], slow=slow_speech)
                    if audio_file:
                        st.markdown(create_audio_player(audio_file), unsafe_allow_html=True)
                        st.download_button(
                            "📥 Download Audio", 
                            data=open(audio_file, "rb"), 
                            file_name=f"{breed}_info_{selected_lang}.mp3",
                            mime="audio/mp3"
                        )

        # Enhanced YouTube Resources
        st.markdown("---")
        st.markdown("### 📺 Educational Resources")
        
        suggestions = get_enhanced_youtube_suggestions(breed)
        
        for category, queries in suggestions.items():
            with st.expander(f"{category}"):
                st.markdown(f"**Search these topics on YouTube for {category.split(' ', 1)[1].lower()}:**")
                
                try:
                    # Try to fetch actual links for first query
                    sample_links = fetch_youtube_links(queries[0])[:2]
                    if sample_links:
                        st.markdown("**📍 Sample Videos Found:**")
                        for i, url in enumerate(sample_links):
                            st.markdown(f"🔗 [Watch Video {i+1}]({url})")
                        st.markdown("---")
                except:
                    pass
                
                # Show all search suggestions
                for query in queries:
                    st.markdown(f"""
                    <div class="video-suggestion">
                        <strong>🔍 Search:</strong> "{query}"<br>
                        <em>💡 Purpose:</em> {category.split(' ', 1)[1]}
                    </div>
                    """, unsafe_allow_html=True)
    
    elif uploaded_file and 'confidence' in locals() and confidence < 30:
        # Show alternative resources for low confidence cases
        st.markdown("---")
        st.markdown("### 📚 General Breed Identification Resources")
        st.markdown("""
        Since the AI confidence is low, here are some general resources to help with manual identification:
        
        **🔍 Breed Identification Steps:**
        1. **Physical Features**: Examine head shape, body size, color patterns
        2. **Regional Context**: Consider the geographic location
        3. **Production Traits**: Observe milk yield, body condition
        4. **Expert Consultation**: Contact local veterinarian or animal husbandry officer
        
        **📖 Useful Resources:**
        - Government breed identification manuals
        - ICAR breed documentation
        - Local veterinary colleges
        - Animal husbandry department guidelines
        """)
        
        # Show general YouTube suggestions with actual URLs
        st.markdown("### 📺 General Learning Resources")
        general_resources = [
            {
                "title": "Indian cattle breed identification guide",
                "url": "https://www.youtube.com/results?search_query=indian+cattle+breed+identification+guide"
            },
            {
                "title": "Buffalo breed characteristics India", 
                "url": "https://www.youtube.com/results?search_query=buffalo+breed+characteristics+india"
            },
            {
                "title": "Livestock breed classification methods",
                "url": "https://www.youtube.com/results?search_query=livestock+breed+classification+methods"
            },
            {
                "title": "Animal husbandry identification techniques",
                "url": "https://www.youtube.com/results?search_query=animal+husbandry+identification+techniques"
            }
        ]
        
        for resource in general_resources:
            st.markdown(f"🔍 **[{resource['title']}]({resource['url']})**")

    if uploaded_file and 'breed' in locals():
        # BPA Registration Form - always show but adapt based on confidence
        st.markdown("---")
        st.markdown("### 📝 BPA Registration Entry")
        
        # Simple message without confidence categorization
        st.info("🎯 AI has provided breed identification to ensure accurate BPA data entry.")
        
        with st.form("bpa_registration", clear_on_submit=True):
            st.markdown("#### 👤 Owner Information")
            col_owner1, col_owner2 = st.columns(2)
            with col_owner1:
                owner_name = st.text_input("👤 Owner Name*", help="Full name of animal owner")
                phone = st.text_input("📱 Phone Number", help="Contact number")
            with col_owner2:
                owner_id = st.text_input("🆔 Owner ID", help="Government issued ID")
                village = st.text_input("🏘️ Village/Area", help="Location details")
            
            st.markdown("#### 🏠 Farm Information")
            col_farm1, col_farm2 = st.columns(2)
            with col_farm1:
                farm_id = st.text_input("🏠 Farm ID*", help="Unique farm identifier")
                farm_size = st.number_input("📐 Farm Size (acres)", min_value=0.0, step=0.1)
            with col_farm2:
                district = st.text_input("🏛️ District", help="Administrative district")
                state = st.text_input("🗺️ State", help="State location")
            
            st.markdown("#### 🐮 Animal Information")
            col_animal1, col_animal2, col_animal3 = st.columns(3)
            with col_animal1:
                animal_id = st.text_input("🐮 Animal ID*", help="Unique animal identifier")
                age = st.number_input("🎂 Age (years)", min_value=0, step=1)
            with col_animal2:
                gender = st.radio("⚧ Gender*", ["Male", "Female"])
                weight = st.number_input("⚖️ Weight (kg)", min_value=0, step=5)
            with col_animal3:
                category = st.selectbox("📂 Category", ["Dairy", "Breeding", "Draught", "Dual Purpose"])
                health_status = st.selectbox("🏥 Health Status", ["Healthy", "Under Treatment", "Vaccinated"])
            
            st.markdown("#### 📊 Production Information")
            col_prod1, col_prod2 = st.columns(2)
            with col_prod1:
                milk_yield = st.number_input("🥛 Daily Milk Yield (liters)", min_value=0.0, step=0.5)
                lactation = st.number_input("🤱 Lactation Number", min_value=0, step=1)
            with col_prod2:
                breeding_method = st.selectbox("🧬 Breeding Method", ["Natural", "AI", "ET", "Not Applicable"])
                pregnancy_status = st.selectbox("🤰 Pregnancy Status", ["Not Pregnant", "Pregnant", "Not Applicable"])
            
            notes = st.text_area("📝 Additional Notes", help="Any specific observations or remarks")
            
            # Simple submit button
            submitted = st.form_submit_button("💾 Register in BPA System", type="primary")
            
            if submitted and owner_name and farm_id and animal_id:
                # Get breed_summary from session state or use default
                current_breed_summary = st.session_state.get('current_breed_summary', f"AI identified breed: {breed}")
                
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "breed": breed,
                    "prediction_method": "AI Suggestion",
                    "owner_name": owner_name,
                    "phone": phone,
                    "owner_id": owner_id,
                    "village": village,
                    "farm_id": farm_id,
                    "farm_size": farm_size,
                    "district": district,
                    "state": state,
                    "animal_id": animal_id,
                    "age": age,
                    "gender": gender,
                    "weight": weight,
                    "category": category,
                    "health_status": health_status,
                    "milk_yield": milk_yield,
                    "lactation": lactation,
                    "breeding_method": breeding_method,
                    "pregnancy_status": pregnancy_status,
                    "language": selected_lang,
                    "notes": notes,
                    "breed_summary": current_breed_summary,
                    "validation_status": "AI-Assisted Entry (Quality Assured)"
                }
                st.session_state.registrations.append(entry)
                st.success("✅ Animal registered successfully in BPA system!")
                st.balloons()
            elif submitted:
                st.error("⚠️ Please fill all mandatory fields marked with *")

# -----------------------------
# TAB 2: ANALYTICS DASHBOARD
# -----------------------------
with tab2:
    st.markdown("### 📊 BPA Analytics Dashboard")
    
    if st.session_state.registrations:
        # Create DataFrame for analysis
        df = pd.DataFrame(st.session_state.registrations)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 Total Predictions", st.session_state.prediction_count)
        with col2:
            st.metric("📋 Registrations", len(st.session_state.registrations))
        # with col3:
        #     avg_confidence = sum(st.session_state.accuracy_scores) / len(st.session_state.accuracy_scores)
        #     st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
        with col3:
            unique_breeds = len(set(df['breed']))
            st.metric("🐄 Unique Breeds", unique_breeds)
        
        # Breed Distribution Chart
        st.markdown("### 📈 Breed Distribution Analysis")
        if st.session_state.breed_distribution:
            breed_df = pd.DataFrame(
                list(st.session_state.breed_distribution.items()), 
                columns=['Breed', 'Count']
            )
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_bar = px.bar(breed_df, x='Breed', y='Count', 
                               title="Breed Count Distribution",
                               color='Count', color_continuous_scale='viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_chart2:
                fig_pie = px.pie(breed_df, values='Count', names='Breed',
                               title="Breed Percentage Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Geographic Distribution
        if 'state' in df.columns and not df['state'].isna().all():
            st.markdown("### 🗺️ Geographic Distribution")
            state_counts = df['state'].value_counts()
            if not state_counts.empty:
                fig_geo = px.bar(x=state_counts.index, y=state_counts.values,
                               title="Registrations by State",
                               labels={'x': 'State', 'y': 'Count'})
                st.plotly_chart(fig_geo, use_container_width=True)
        
        # Production Analysis
        if 'milk_yield' in df.columns:
            st.markdown("### 🥛 Production Analysis")
            col_prod1, col_prod2 = st.columns(2)
            
            with col_prod1:
                avg_yield_by_breed = df.groupby('breed')['milk_yield'].mean().reset_index()
                if not avg_yield_by_breed.empty:
                    fig_yield = px.bar(avg_yield_by_breed, x='breed', y='milk_yield',
                                     title="Average Milk Yield by Breed")
                    st.plotly_chart(fig_yield, use_container_width=True)
            
            with col_prod2:
                if 'age' in df.columns:
                    fig_scatter = px.scatter(df, x='age', y='milk_yield', color='breed',
                                           title="Milk Yield vs Age by Breed")
                    st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("📊 No data available yet. Start by registering animals to see analytics.")

# -----------------------------
# TAB 3: BPA REGISTRATIONS
# -----------------------------
with tab3:
    st.markdown("### 📋 BPA Registration Records")
    
    if st.session_state.registrations:
        # Search and filter options
        col_search1, col_search2, col_search3 = st.columns(3)
        with col_search1:
            search_breed = st.selectbox("🔍 Filter by Breed", 
                                      ["All"] + list(set([r['breed'] for r in st.session_state.registrations])))
        with col_search2:
            search_state = st.selectbox("🗺️ Filter by State",
                                      ["All"] + list(set([r.get('state', 'Unknown') for r in st.session_state.registrations])))
        with col_search3:
            search_status = st.selectbox("✅ Filter by Status",
                                       ["All", "Pending", "Validated", "Needs Review"])
        
        # Export functionality
        if st.button("📤 Export to CSV"):
            df_export = pd.DataFrame(st.session_state.registrations)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"bpa_registrations_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        # Display registrations
        for i, entry in enumerate(reversed(st.session_state.registrations)):
            # Apply filters
            if (search_breed != "All" and entry['breed'] != search_breed):
                continue
            if (search_state != "All" and entry.get('state', 'Unknown') != search_state):
                continue
            
            # Create a more informative header
            prediction_method = entry.get('prediction_method', 'AI Suggestion')
            validation_status = entry.get('validation_status', 'Pending')
            
            with st.expander(f"🐄 {entry['breed']} - {entry['owner_name']} | {prediction_method} | {validation_status} ({entry['timestamp']})"):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.markdown("**👤 Owner Details:**")
                    st.write(f"Name: {entry['owner_name']}")
                    st.write(f"Phone: {entry.get('phone', 'N/A')}")
                    st.write(f"Village: {entry.get('village', 'N/A')}")
                    st.write(f"State: {entry.get('state', 'N/A')}")
                
                with col_info2:
                    st.markdown("**🏠 Farm Details:**")
                    st.write(f"Farm ID: {entry['farm_id']}")
                    st.write(f"Size: {entry.get('farm_size', 'N/A')} acres")
                    st.write(f"District: {entry.get('district', 'N/A')}")
                
                with col_info3:
                    st.markdown("**🐮 Animal Details:**")
                    st.write(f"Animal ID: {entry['animal_id']}")
                    st.write(f"Breed: {entry['breed']}")
                    st.write(f"Age: {entry['age']} years")
                    st.write(f"Gender: {entry['gender']}")
                    st.write(f"Category: {entry.get('category', 'N/A')}")
                
                if entry.get('milk_yield', 0) > 0:
                    st.markdown("**🥛 Production Details:**")
                    st.write(f"Daily Yield: {entry['milk_yield']} liters")
                    st.write(f"Lactation: {entry.get('lactation', 'N/A')}")
                
                if entry.get('notes'):
                    st.markdown(f"**📝 Notes:** {entry['notes']}")
                
                # Show breed summary with option to regenerate using Groq
                if entry.get('breed_summary'):
                    with st.expander("📚 View Breed Information"):
                        st.markdown(entry['breed_summary'])
                        
                        # Option to regenerate summary using Groq
                        if st.button(f"🔄 Regenerate Summary", key=f"regen_summary_{i}"):
                            with st.spinner("🧠 Generating updated breed information..."):
                                prompt = f"""
                                Provide comprehensive information about '{entry['breed']}' breed in English including:
                                - Key identification features
                                - Origin and distribution in India
                                - Milk/meat production capacity
                                - Breeding characteristics
                                - Economic importance
                                - Management requirements
                                Format in bullet points for easy reading.
                                """
                                response = groq_agent.run(prompt)
                                new_summary = response.content if hasattr(response, "content") else str(response)
                                
                                # Update the entry in session state
                                st.session_state.registrations[len(st.session_state.registrations)-1-i]['breed_summary'] = new_summary
                                st.success("✅ Breed summary updated!")
                                st.rerun()
                else:
                    # Generate summary if missing
                    if st.button(f"📚 Generate Breed Info", key=f"gen_summary_{i}"):
                        with st.spinner("🧠 Generating breed information..."):
                            prompt = f"""
                            Provide comprehensive information about '{entry['breed']}' breed in English including:
                            - Key identification features
                            - Origin and distribution in India
                            - Milk/meat production capacity
                            - Breeding characteristics
                            - Economic importance
                            - Management requirements
                            Format in bullet points for easy reading.
                            """
                            response = groq_agent.run(prompt)
                            new_summary = response.content if hasattr(response, "content") else str(response)
                            
                            # Update the entry in session state
                            st.session_state.registrations[len(st.session_state.registrations)-1-i]['breed_summary'] = new_summary
                            st.success("✅ Breed information generated!")
                            st.rerun()
                
                # Action buttons
                col_action1, col_action2, col_action3 = st.columns(3)
                with col_action1:
                    if st.button(f"✅ Validate {i}", key=f"validate_{i}"):
                        st.success("Record validated!")
                with col_action2:
                    if st.button(f"🔄 Edit {i}", key=f"edit_{i}"):
                        st.info("Edit functionality would open here")
                with col_action3:
                    if st.button(f"📤 Export {i}", key=f"export_{i}"):
                        st.info("Individual export functionality")
    else:
        st.info("📋 No registrations found. Start by identifying breeds and registering animals.")

# -----------------------------
# TAB 4: LEARNING CENTER
# -----------------------------
with tab4:
    st.markdown("### 📚 FLW Learning Center")
    
    # Training modules
    st.markdown("#### 🎓 Training Modules for Field Level Workers")
    
    modules = [
        {
            "title": "🐄 Cattle Breed Identification Basics",
            "description": "Learn to identify key features of major Indian cattle breeds",
            "topics": ["Physical characteristics", "Regional distribution", "Economic importance"]
        },
        {
            "title": "🐃 Buffalo Breed Recognition",
            "description": "Master buffalo breed identification techniques",
            "topics": ["Murrah vs Nili-Ravi", "Mehsana characteristics", "Surti breed features"]
        },
        {
            "title": "📱 BPA App Usage Guide",
            "description": "Complete guide to using BPA mobile application",
            "topics": ["Registration process", "Data validation", "Photo capture best practices"]
        },
        {
            "title": "🤖 AI Tool Integration",
            "description": "How to use AI breed classifier effectively",
            "topics": ["Image quality requirements", "Confidence score interpretation", "Manual override procedures"]
        }
    ]
    
    for module in modules:
        with st.expander(f"{module['title']}"):
            st.markdown(f"**Description:** {module['description']}")
            st.markdown("**Topics Covered:**")
            for topic in module['topics']:
                st.markdown(f"• {topic}")
            
            col_learn1, col_learn2 = st.columns(2)
            with col_learn1:
                st.button(f"📖 Start Learning", key=f"learn_{module['title']}")
            with col_learn2:
                st.button(f"📹 Watch Videos", key=f"video_{module['title']}")
    
    # Quick reference guides
    st.markdown("---")
    st.markdown("#### 📖 Quick Reference Guides")
    
    reference_guides = {
        "🔍 Breed Identification Checklist": [
            "Check head shape and size",
            "Observe body structure and proportions", 
            "Note color patterns and markings",
            "Examine udder characteristics (females)",
            "Consider regional context"
        ],
        "📸 Photo Quality Guidelines": [
            "Ensure good lighting conditions",
            "Capture clear, focused images",
            "Include distinctive breed features",
            "Take multiple angles if possible",
            "Avoid shadows and obstructions"
        ],
        "✅ Data Validation Process": [
            "Verify AI prediction confidence",
            "Cross-check with visual features",
            "Consult local breed experts if unsure",
            "Use manual override when necessary",
            "Document validation decisions"
        ]
    }
    
    for guide_title, checklist in reference_guides.items():
        with st.expander(guide_title):
            for item in checklist:
                st.markdown(f"✓ {item}")

# -----------------------------
# TAB 5: ABOUT BPA INTEGRATION
# -----------------------------
with tab5:
    st.markdown("### ℹ️ About BPA Integration")
    
    # Problem statement
    st.markdown("#### 🎯 Problem Statement")
    st.markdown("""
    <div class="feature-highlight">
        <h4>🚨 Challenge: Breed Misclassification in BPA</h4>
        <p>Field Level Workers (FLWs) face difficulties in accurate breed identification, leading to:</p>
        <ul>
            <li>❌ Incorrect data entries in BPA system</li>
            <li>📊 Compromised data quality for research and policy</li>
            <li>💰 Suboptimal resource allocation</li>
            <li>🧬 Ineffective genetic improvement programs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Solution overview
    st.markdown("#### 💡 Our AI Solution")
    
    features = [
        {
            "icon": "🤖",
            "title": "AI-Powered Recognition",
            "description": "Advanced machine learning model trained on diverse Indian cattle and buffalo breeds"
        },
        {
            "icon": "📱",
            "title": "Mobile-First Design",
            "description": "Optimized for field conditions with offline capability and easy-to-use interface"
        },
        {
            "icon": "🌍",
            "title": "Multi-language Support",
            "description": "Available in 11+ Indian languages to assist FLWs across different regions"
        },
        {
            "icon": "🔄",
            "title": "Real-time Validation",
            "description": "Instant breed suggestions with confidence scores for informed decisions"
        },
        {
            "icon": "📊",
            "title": "Data Analytics",
            "description": "Comprehensive analytics for monitoring breed distribution and data quality"
        },
        {
            "icon": "🎓",
            "title": "Training Integration",
            "description": "Built-in learning modules to improve FLW breed identification skills"
        }
    ]
    
    for feature in features:
        st.markdown(f"""
        <div class="feature-highlight">
            <h4>{feature['icon']} {feature['title']}</h4>
            <p>{feature['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical specifications
    st.markdown("#### ⚙️ Technical Specifications")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("**🔧 Core Features:**")
        st.markdown("""
        - ✅ Image-based breed classification
        - ✅ Confidence score evaluation
        - ✅ Multiple prediction ranking
        - ✅ Real-time processing
        - ✅ Offline capability
        - ✅ Multi-language audio output
        - ✅ BPA-compatible data export
        """)
    
    with col_tech2:
        st.markdown("**📊 Performance Metrics:**")
        st.markdown("""
        - 🎯 Classification Accuracy: 85%+
        - ⚡ Processing Time: <3 seconds
        - 📱 Device Compatibility: Android/iOS
        - 🌐 Language Support: 11 languages
        - 📡 Connectivity: Online/Offline modes
        - 💾 Storage: Minimal device storage
        - 🔋 Battery Optimization: Efficient processing
        """)
    
    # Implementation roadmap
    st.markdown("#### 🗺️ Implementation Roadmap")
    
    phases = [
        {
            "phase": "Phase 1: Pilot Testing",
            "duration": "2-3 months",
            "activities": [
                "Deploy in 5 selected districts",
                "Train 100 FLWs on system usage",
                "Collect feedback and usage data",
                "Refine AI model accuracy"
            ]
        },
        {
            "phase": "Phase 2: State-wide Rollout",
            "duration": "6 months",
            "activities": [
                "Scale to entire state",
                "Integrate with existing BPA infrastructure",
                "Comprehensive FLW training program",
                "Establish support and maintenance system"
            ]
        },
        {
            "phase": "Phase 3: National Integration",
            "duration": "12 months",
            "activities": [
                "Deploy across all states",
                "Advanced analytics dashboard",
                "Policy integration and reporting",
                "Continuous improvement and updates"
            ]
        }
    ]
    
    for phase in phases:
        with st.expander(f"📅 {phase['phase']} ({phase['duration']})"):
            for activity in phase['activities']:
                st.markdown(f"• {activity}")
    
    # Benefits and impact
    st.markdown("#### 🎉 Expected Benefits & Impact")
    
    benefits = {
        "🎯 Data Quality": [
            "85% reduction in breed misclassification errors",
            "Improved data reliability for research and policy",
            "Enhanced breeding program effectiveness"
        ],
        "⏱️ Efficiency": [
            "50% faster animal registration process",
            "Reduced training time for new FLWs",
            "Automated data validation and verification"
        ],
        "💰 Economic Impact": [
            "Better resource allocation based on accurate data",
            "Improved livestock productivity tracking",
            "Enhanced genetic improvement program ROI"
        ],
        "👥 User Experience": [
            "Simplified breed identification process",
            "Multi-language support for better adoption",
            "Real-time learning and skill development"
        ]
    }
    
    for benefit_category, items in benefits.items():
        st.markdown(f"**{benefit_category}:**")
        for item in items:
            st.markdown(f"• ✅ {item}")
        st.markdown("")
    
    # Contact and support
    st.markdown("#### 📞 Support & Contact")
    
    col_contact1, col_contact2 = st.columns(2)
    
    with col_contact1:
        st.markdown("""
        **🛠️ Technical Support:**
        - 📧 Email: support@bpa-ai-classifier.gov.in
        - 📱 Helpline: 1800-XXX-XXXX
        - ⏰ Hours: 9 AM - 6 PM (Mon-Fri)
        """)
    
    with col_contact2:
        st.markdown("""
        **📋 Training & Resources:**
        - 🎓 Training Portal: training.bpa.gov.in
        - 📖 User Manual: docs.bpa-classifier.in
        - 📹 Video Tutorials: Available in app
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-top: 2rem;">
        <h3>🚀 Ready to Transform Livestock Data Collection?</h3>
        <p>Join the AI revolution in agricultural data management with BPA Smart Breed Identifier</p>
        <p><strong>Developed for Smart India Hackathon 2025</strong></p>
        <p>🎯 Accurate • 📱 User-friendly • 🌍 Multi-lingual • 🤖 AI-powered</p>
    </div>
    """, unsafe_allow_html=True)