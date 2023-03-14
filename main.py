import requests
from bs4 import BeautifulSoup
import json

from bassTab import BassTab


#%%
url = 'https://tabs.ultimate-guitar.com/tab/masego/tadow-bass-3148656'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

data_content = soup.find('div', {'class': 'js-store'})['data-content']
data = json.loads(data_content)

if not data['store']['page']['data']['tab_view']['tracking']['ctx']['type']=='Bass Tabs':
    print('Warning, not a bass tab that is being loaded')
else:   
    # Now you can access the data you are interested in using dictionary keys
    tab = data['store']['page']['data']['tab_view']['wiki_tab']['content']
    dic = data['store']['page']['data']['tab_view']['versions'][0]

    # Split the tablature text into individual lines
    tab_lines = tab.split('\n')
    
    # create class to save the lines in
    bassTab = BassTab(song_name=dic["song_name"], artist=dic["artist_name"], tonality=dic["tonality_name"])
    
    # Extract the relevant information from the lines we're interested in
    for line in tab_lines:
        if line.startswith(("[tab]G|")):
            G_new, _, Gisempty = bassTab.parse_tab_line(line[6:])
        if line.startswith(("D|")):
            D_new, _, Disempty  = bassTab.parse_tab_line(line[1:])
        if line.startswith(("A|")):
            A_new, _, Aisempty  = bassTab.parse_tab_line(line[1:])
        if line.startswith(("E|")):
            E_new, repeat, Eisempty  = bassTab.parse_tab_line(line[1:])
            
            # add only the bassline to the data if it contains notes
            if Gisempty and Disempty and  Aisempty and Eisempty:
                bassTab.append_empty(repeat) 
            else:            
                bassTab.append_bassline(E_new, A_new, D_new, G_new, repeat)