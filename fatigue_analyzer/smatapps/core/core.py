# -*- coding: utf-8 -*-
# ****************************************************************************
# * (C) 2025 Johannes Moeller ST/HZA-CEA                                     *
# ****************************************************************************

__author__ = "J. Möller, Schaeffler Technologies, Schweinfurt"
__credits__ = ["J. Moeller", "R. Nuetzel"]
__copyright__ = "(C) 2019-2025 J. Moeller"
__license__ = ""
__version__ = "0.2"
__status__ = "Development"
__contact__ = "johannes.moeller@schaeffler.com"

import os
#from smat.SMatBase import SMatBase
import streamlit as st
import base64
import logging
from io import BytesIO
import platform
import pandas as pd
from PIL import Image
from datetime import datetime

class SMatApp(object):

    def __init__(self, name, loglevel=logging.INFO, layout="wide"):
        self.path = self.get_path()
        
        self.log = logging
        self.log.basicConfig(format='%(asctime)s %(levelname)s %(message)s', 
                             datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)
                             
        icon = Image.open(self.get_path()+os.sep+"smatapps_logo.jpg")
        st.set_page_config(page_title=name+"@materials-dev", page_icon=icon, layout=layout)

    def get_path(self, file=__file__):
        #return os.path.dirname(file)
        return os.path.split(os.path.realpath(__file__))[0]
        
    def get_baseUrl(self): 
        self.serverAddress=st.get_option("browser.serverAddress")
        self.serverPort=st.get_option("browser.serverPort")
        self.port=st.get_option("server.port")
        self.baseUrlPath=st.get_option("server.baseUrlPath")
        url="http"
        url+="s" if self.serverPort==443 else ""
        url+="://"+self.serverAddress
        url+=":"+str(self.serverPort) if self.serverPort!=443 else ""
        url+="/"+self.baseUrlPath+"/" if self.baseUrlPath!="" else "/"
        return url
    
    def init_design(self, title="Unnamed", menu=True):
   
        ###############################################################################
        #
        # Standard header for common layout
        #
        ###############################################################################
        #st.set_page_config(page_title=title,page_icon="🔍",initial_sidebar_state="expanded", menu_items={'Get Help': 'https://materials-dev.schaeffler.com/','Report a bug': "https://materials-dev.schaeffler.com/",'About': "Streamlit Materials Apps provided by Schaeffler Werkstofftechnik"})
        st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 100%;
                max-height: 100%
                padding-top: 3rem;
                padding-right: 3rem;
                padding-left: 3rem;
                padding-bottom: 3rem;
            }}
            .logo-img{{
                margin-bottom:0px;
                max-width:300px;
                display: block;
                margin-left: auto;
            }}
            
        </style>
        """,
                unsafe_allow_html=True,
            )

        schaeffler_logo_path=self.get_path()+os.sep+'Schaeffler_Logo_small.jpg'

        logo = st.columns(1)
        #logo = st.markdown(
        #    f"""
        #    <div class="logo-container">
        #        <img class="logo-img" src="data:image/jpg;base64,{base64.b64encode(open(schaeffler_logo_path, "rb").read()).decode()}">
        #    </div>
        #    """,
        #    unsafe_allow_html=True
        #)
        logo = st.html(f"""<div class="logo-container"><img class="logo-img" src="data:image/jpg;base64,{base64.b64encode(open(schaeffler_logo_path, "rb").read()).decode()}"></div>""")

        ###############################################################################
        #
        # End of standard header for common layout
        #
        ###############################################################################
        
        st.title(title)

        if menu:
            self.add_menu()

        self.add_navbar()

    def get_registered_apps(self):
        """ Retrieve a list of all apps registered in ´smatapps´ """

        registered_apps_file = self.get_path()+os.sep+".."+os.sep+"REGISTERED_APPS.txt"
        if os.path.exists(registered_apps_file):
            with open(registered_apps_file, "r") as f:
                registered_apps = f.read().splitlines()
            if "landing_page" in registered_apps:
                registered_apps.remove("landing_page")
            return registered_apps
        else:
            return []

    def add_menu(self):
        """ Add a menu to the sidebar to navigate to other apps """

        base_url = "https://materials-dev.schaeffler.com"
        st.sidebar.markdown(f"<h1><a href=\"{base_url}\" style=\"text-decoration:none;color:#01893E\">MaterialsDevApps</a></h1>", unsafe_allow_html=True)
        for app in self.get_registered_apps():
            url = base_url + "/" + app
            st.sidebar.markdown(f"&nbsp;&#8226;&nbsp;<a href=\"{url}\" style=\"text-decoration:none;color:#656062\"><b>{app}</b></a>", unsafe_allow_html=True)

    def add_navbar(self):
        """ Add a navigation bar """

        if hasattr(self, "modes"):
            for name, mode in self.modes.items():
                mode.append(self.url+"?mode="+name)
            navbar = f'<a href="{self.url}" target="_self">Home</a> | '
            navbar += " | ".join([f'<a href="{mode[2]}" target="_self">{mode[0]}</a>' for mode in self.modes.values()])
            st.markdown(navbar, unsafe_allow_html=True)

    def add_copyright(self):
        current_year = datetime.now().year
        st.markdown(f"© Schaeffler Technologies AG & Co. KG, 2021-{current_year}")

    def get_download_link(self, file, filename="download.xlsx", display_text="here"):
        """ Retrieve a download link for  """

        buffer = BytesIO()
    
        if type(file)==dict:
            with pd.ExcelWriter(buffer) as writer:
                for name, df in file.items():
                    df.to_excel(writer, sheet_name=name[:31])

        elif type(file)==pd.DataFrame:
            with pd.ExcelWriter(buffer) as writer:
                file.to_excel(writer)
        else:
            self.log.error("File type not implemented.")
            return

        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{display_text}</a>'
        
        return href

