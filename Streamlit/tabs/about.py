import streamlit as st
import os

def sidebar_choice():
    st.title("👥 L'Équipe Projet")
    
    # --- NOTRE DEMARCHE (PREMIUM CARD) ---
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e5631 0%, #2e8b57 100%); 
                padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px;
                box-shadow: 0 4px 15px rgba(46, 139, 87, 0.2); border-left: 8px solid #004d40;'>
        <h3 style='margin-top:0; color: #e8f5e9;'>💡 Notre Démarche</h3>
        <p style='font-size: 1.1em; line-height: 1.5; margin-bottom: 0;'>
            Ce projet s'inscrit dans le cadre de notre formation chez <b>DataScientest</b>, 
            couvrant un large éventail de techniques allant du Machine Learning classique 
            aux architectures de Deep Learning les plus avancées.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- TEAM IMAGE & GRATITUDE ---
    if os.path.exists("Streamlit/assets/team_collage.png"):
        col_img, col_txt = st.columns([1.2, 1], gap="large")
        
        with col_img:
            st.image("Streamlit/assets/team_collage.png", width="stretch")
            
        with col_txt:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); 
                        padding: 30px; border-radius: 20px; border: 1px solid #bcccdc;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.05); height: 100%; display: flex; 
                        flex-direction: column; justify-content: center;'>
                <h3 style='color: #243b53; margin-top: 0;'>Un mot de l'équipe</h3>
                <p style='font-style: italic; color: #486581; font-size: 1.1em; line-height: 1.6;'>
                    "Ce projet fut une aventure humaine et technique exceptionnelle. 
                    Nous sommes fiers du chemin parcouru ensemble et de la cohésion de notre groupe. 
                    Un immense merci à toute l'équipe pour cette collaboration inoubliable !"
                </p>
                <p style='text-align: right; font-weight: bold; color: #2E8B57; margin-bottom: 0;'>
                    — Merci à tous !
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Photo d'équipe à intégrer ici.")
