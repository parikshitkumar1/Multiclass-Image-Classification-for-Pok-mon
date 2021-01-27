import streamlit as st
from PIL import Image
from model import pred
import requests
from bs4 import BeautifulSoup



def getdata(url):
    r = requests.get(url)
    return r.text




st.set_option('deprecation.showfileUploaderEncoding', False)
imag = Image.open('pika.jpeg')

st.image(imag, caption='', use_column_width=True)

st.header("""Find out whatever you want to about your Pokémon""")
st.subheader("Includes every first generation Pokémon and mega evolutions")
st.subheader("ϞϞ(๑⚈ ․̫ ⚈๑) Ƶƶ(￣▵—▵￣) ( ͡° ͜ʖ ͡°)⊃━☆ﾟ✧ ζ,,ﾟДﾟζ ╭<<◕°ω°◕>>╮")
st.write("")

imager = Image.open('dex.jpeg')

st.image(imager, caption='', use_column_width=True)
st.title("""Pokédex""")

	




ul = "https://github.com/pksenpai/Pokedex"
st.subheader("How this works: ")
st.write(ul)


file_up = st.file_uploader("Upload a picture of your Pokémon", type=["png","jpg","jpeg"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Loading Pokémon prediction...', use_column_width=True)
    st.write("")
    st.subheader("The Pokémon is: ")
    labels = pred(file_up)
    st.subheader(labels)
    x = "https://www.pokemon.com/us/pokedex/"
    htmldata = getdata(x+str(labels))
    soup = BeautifulSoup(htmldata, 'html.parser')
    data = ''
    text = []
    for data in soup.find_all("p"):
        text.append(data.get_text())
    st.write(text)
    
	





    