# -*- coding: utf-8 -*-
# ****************************************************************************
# * (C) 2021 Johannes Moeller ST/SWE-CMWE                                           *
# ****************************************************************************

__author__ = "J. Möller, Schaeffler Technologies, Schweinfurt"
__credits__ = ["J. Moeller", "R. Nuetzel"]
__copyright__ = "(C) 2019-2021 J. Moeller"
__license__ = ""
__version__ = "0.1"
__status__ = "Development"
__contact__ = "johannes.moeller@schaeffler.com"

from smatapps.core.core import SMatApp
import streamlit as st

class ExampleApp(SMatApp):
    """ Example app class for demonstration purposes. Just run 

        $> cd path\to\smatapps
        $> streamlit run example\example.py

        in the terminal to launch. """

    def __init__(self):
        """ Initialize the application with some general attributes and the standard layout """
        
        self.appname="ExampleApp"
        super().__init__(self.appname)
        self.path=os.path.split(os.path.realpath(__file__))[0]
        self.url=self.get_baseUrl()
        self.init_design(self.appname)
        
    def build_gui(self):
        """ Build the graphical user interface of the actual app """
        st.write("Hello World!")
        self.log.info("Hello Log!")

if __name__ == '__main__':
    app=ExampleApp()
    app.build_gui()
