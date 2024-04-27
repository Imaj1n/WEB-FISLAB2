import streamlit as st

from streamlit_option_menu import option_menu


import W1,W2,W3,W4,W5,MP1,MP2,MP3,MP4,MP5
st.set_page_config(
        page_title="Judul Praktikum",
)



class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Praktikum',
                options=["W1","W2","W3","W4","W5","MP1","MP2","MP3","MP4","MP5"],
                menu_icon='chat-text-fill',
                default_index=1,
        #         styles={
        #             "container": {"padding": "5!important","background-color":'black'},
        # "icon": {"color": "white", "font-size": "23px"}, 
        # "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        # "nav-link-selected": {"background-color": "#02ab21"},}      
                )
        if app=="W1":
            W1.app()
        if app == 'W2':
            W2.app()
        if app == 'W3':
            W3.app()
        if app == 'W4':
            W4.app()
        if app == 'W5':
            W5.app()
        if app == 'MP1':
            MP1.app()
        if app == 'MP2':
            MP2.app()
        if app == 'MP3':
            MP3.app()
        if app =='MP4':
            MP4.app()
        if app == 'MP5':
            MP5.app()    
          
             
    run()            
         