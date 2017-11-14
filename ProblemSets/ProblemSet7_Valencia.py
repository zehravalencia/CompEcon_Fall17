# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:34:46 2017

@author: zcagi
"""


#APPLICATION 1
#The realtionship between INFANT MORTALITY RATES - GDP GROWTH 
#Belgium, France, Germany, Italy, Luxembourg and the Netherlands --> THE FATHERS OF EU

import wbdata  #world bank data
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from bs4 import BeautifulSoup
import urllib.request
from pandas_datareader import wb
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt



#APPLICATION 1  : API world bank
#searching for variables
wb.search('growth').iloc[:,:2] #search for gdp growth
wb.search('mortality').iloc[:,:2] #search for mortality variables.I am interested in "infant mortality"

#download the gdp growth rates of first EU countries.
#https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
#https://europa.eu/european-union/about-eu/countries/member-countries_en
countries = ["BE", "FR" , "DE", "IT", "LU", "NL", ]  #EU COUNTRIES
gdp_growth = wb.download(indicator='NY.GDP.MKTP.KD.ZG',country=countries, start=1970, end=2015)
print(gdp_growth)
#visualization
#dictionary
indicators = {'NY.GDP.MKTP.KD.ZG':'GDP growth'}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)
variable = ['NY.GDP.MKTP.KD.ZG','SH.DYN.NMRT'] 
data = wb.download(indicator=variable, country=countries, start=1970, end=2015).dropna()
print(data)
#df is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
dfu = df.unstack(level=0)
dfu.plot(); 
plt.legend(loc='best'); 
plt.title("GDP growth (annual %)"); 
plt.xlabel('Date'); plt.ylabel('GDP growth (annual %)');
dfu.plot()
plt.show()


indicators2 = {'SH.DYN.NMRT':'infant mortality'}
df = wbdata.get_dataframe(indicators2, country=countries, convert_date=False)
dfu = df.unstack(level=0)
dfu.plot(); 
plt.legend(loc='best'); 
plt.title("infant mortality"); 
plt.xlabel('Date'); plt.ylabel('infant mortality');
dfu.plot()
plt.show()



#APPLICATION 2: WEB SCRAPING IMBD




#The Internet Movie Database (abbreviated IMDb) is an online database of information related to films, television programs and video games.

#I want to analyze the distributions of IMDb and Metacritic movie ratings in year 2015.
from requests import get  # we import the get() function from the requests module.

url = 'http://www.imdb.com/search/title?release_date=2015&ref_=rlm_yr' #we assign the address of the web page to a variable named

response = get(url)
print(response.text[:1000])


#google crome--> right click --> inspect --> brings html code.
#Every movie name starts with <div> 

from bs4 import BeautifulSoup

html_soup = BeautifulSoup(response.text, 'html.parser')
type(html_soup)


movie_information = html_soup.find_all('div', class_ = 'lister-item mode-advanced') #in html code  the class attribute has two values: lister-item and mode-advanced.
print(type(movie_information))
print(len(movie_information))   #number if <div> (there are 50 movies per page)

# the information on movies :  name of the movie,year, IMBD rating, the metascore

number_one_film = movie_information[0]
number_one_film


#  *** NAME***
# <a href="/title/tt4052886/?ref_=adv_li_tt">Lucifer</a>
number_one_film.h3.a.text


name_of_number_one_film = number_one_film.h3.a.text
name_of_number_one_film

# ***YEAR***
# <span class="lister-item-year text-muted unbold">(2015– )</span>
number_one_film_year = number_one_film.h3.find('span', class_ = 'lister-item-year text-muted unbold')
number_one_film_year


number_one_film_year = number_one_film_year.text
number_one_film_year


# ***IMDB SCORE***
# <strong>8.3</strong> --> it gave an error : 'str' object has no attribute 'strong'; so I converted it to float
 
number_one_film_imdb = float(number_one_film.strong.text)
number_one_film_imdb

##This movie does not have metascore or gross value, so I wanted to check another movie to get the html ...Avengers: Age of Ultron has this; this movie is #14 in the ranks of year 15.
number_fourteen_film = movie_information[13]
number_fourteen_film

# ***METASCORE***
# <span class="metascore favorable">66 
number_fourteen_film_metascore = number_fourteen_film.find('span', class_ = 'metascore favorable')

number_fourteen_film_metascore = int(number_fourteen_film_metascore.text)   #convert
print(number_fourteen_film_metascore)

#prepeare the lists to store the data
names = []
years = []
imdb_ratings = []
metascores = []


for information in movie_information: #we gather information from "movie_information"

    if information.find('div', class_ = 'ratings-metascore') is not None:   #but not all movies has metascore". Movies will get a Metascore only if at least four critics's reviews are collected. 
        
        # The name of the film
        name = information.h3.a.text
        names.append(name)
        
        # The year of the movie 
        year = information.h3.find('span', class_ = 'lister-item-year').text
        years.append(year)
        
        # The IMDB rating
        imdb = float(information.strong.text)
        imdb_ratings.append(imdb)
        
        # The Metascore
        m_score = information.find('span', class_ = 'metascore').text
        metascores.append(int(m_score))



data = pd.DataFrame({'movie': names,
                       'year': years,
                       'imdb': imdb_ratings,
                       'metascore': metascores})
print(data.info())
data

# # I could not get all the movies in one page...so Lucifer did not appear in the table.



import matplotlib.pyplot as plt
%matplotlib inline 
data.plot(y='imdb', kind='bar')

import matplotlib.pyplot as plt
%matplotlib inline 
data.plot(y='metascore', kind='bar')


import seaborn as sns
ax = sns.barplot(x="imdb", y="metascore", data=data)