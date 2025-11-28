import streamlit as st


st.title("Streamlit Demo")

color = st.sidebar.selectbox("Pick Color", 
        options = ('Red', 'Blue', 'Green'))

name = st.text_input("Your Name", max_chars=50)

str = f"{name} has picked {color}"


if(len(name)> 0):
    str