import requests
from bs4 import BeautifulSoup
import json

from bassTab import BassTab, parse_tab_line
from embeddings import Embeddings

def load_bassTab(url):
    '''
    Loads a bass tab from a URL and returns an instance of the BassTab class containing the tab data.

    Parameters
    ----------
    url : str
        The URL of the bass tab to be loaded.
    
    Returns
    -------
    bassTab : BassTab
        An instance of the BassTab class containing the loaded tab data.
    '''
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        data_content = soup.find('div', {'class': 'js-store'})['data-content']
        data = json.loads(data_content)
    except: 
        print("URL cannot be loaded")
        return None
    
    '''
    try:
        if (data['store']['page']['data']['tab_view']['meta']['tuning']['value']!='Eb Ab Db Gb' or 
            data['store']['page']['data']['tab_view']['meta']['tuning']['value']!='E A D G'):
            print('Unknown tuning.')
    except: print("Tuning not found")
    '''
    try:
        if not data['store']['page']['data']['tab']['type']=='Bass Tabs':
            print('Warning, not a bass tab that is being loaded. Returning None')
            return None
    except: 
        pass
    
    try:
        if data['store']['page']['data']['tab_view']['meta']['tuning']['value']=='D A D G':
            print(data['store']['page']['data']['tab']["artist_name"],data['store']['page']['data']['tab']["song_name"],
                  "in drop-D tuning. So we skip it!")
            return None
    except:
        pass
    
    try:
        # Now you can access the data you are interested in using dictionary keys
        tab = data['store']['page']['data']['tab_view']['wiki_tab']['content']
        dic = data['store']['page']['data']['tab']
    
        # Split the tablature text into individual lines
        tab_lines = tab.split('\n')
        
        # create class to save the lines in
        bassTab = BassTab(name=dic["song_name"], artist=dic["artist_name"], tonality=dic["tonality_name"])
        
        # Extract the relevant information from the lines we're interested in
        Efound=False
        Afound=False
        Dfound=False
        Gfound=False
        repeat_tot=0
        repeat=0
        for line in tab_lines:
            
            #  all the ways the program might fail
            line=line.replace(":","")
            line=line.replace(";","")
            line=line.replace("G-","G|-")
            line=line.replace("Gb-","Gb|-")
            line=line.replace("A-","A|-")
            line=line.replace("Ab-","Ab|-")
            line=line.replace("D-","D|-")
            line=line.replace("Db-","Db|-")
            line=line.replace("||","|")
            if "E-mail" not in line:
                line=line.replace("E-","E|-")
            line=line.replace("Eb-","Eb|-")
            if line[-2:]=="-\r":
                line=line.replace("-\r","-|\r")
            if line[-8:]=='-[/tab]\r':
                line=line.replace('-[/tab]\r','-|[/tab]\r')
            line = line.replace(' |','|')
            line = line.replace('| ','|')
            
            # we only have standard tuning and tuning a half step lower. Drop D is ignored
            if line.startswith(("[tab]Gb|")):
                G_new, repeat, Gisempty = parse_tab_line(line[7:])
                Gfound=True
            elif line.startswith(("[tab]G|")):
                G_new, repeat, Gisempty = parse_tab_line(line[6:])
                Gfound=True
            elif line.startswith(("Gb|")):
                G_new, repeat, Gisempty = parse_tab_line(line[2:])
                Gfound=True
            elif line.startswith(("G|")):
                G_new, repeat, Gisempty = parse_tab_line(line[1:])
                Gfound=True
            elif line.startswith(("Db|")):
                D_new, repeat, Disempty = parse_tab_line(line[2:])
                Dfound=True
            elif line.startswith(("D|")):
                D_new, repeat, Disempty = parse_tab_line(line[1:])
                Dfound=True
            elif line.startswith(("Ab|")):
                A_new, repeat, Aisempty = parse_tab_line(line[2:])
                Afound=True
            elif line.startswith(("A|")):
                A_new, repeat, Aisempty = parse_tab_line(line[1:])
                Afound=True
            elif line.startswith(("Eb|")):
                E_new, repeat, Eisempty = parse_tab_line(line[2:])
                Efound=True
            elif line.startswith(("E|")):
                E_new, repeat, Eisempty = parse_tab_line(line[1:])
                Efound=True
            if repeat is not None and repeat>repeat_tot: repeat_tot=repeat
    
            # add the basslines only if all have been found
            if Efound and Afound and Dfound and Gfound: 
                # add only the bassline to the data if it contains notes
                if Gisempty and Disempty and  Aisempty and Eisempty:
                    bassTab.append_empty(repeat_tot) 
                else:            
                    bassTab.append_bassline(E_new, A_new, D_new, G_new, repeat_tot)
                repeat_tot=0
                
                #reset
                Efound=False
                Afound=False
                Dfound=False
                Gfound=False
                        
        if len(bassTab.G)!=0:
            return bassTab
        else: 
            print("Cannot access",url)
            return None

    except:
        print('UNKNOWN ERROR! for',url)
        return None


#%%
from tqdm import tqdm

# Starting url
##TODO right now we only have rock
# filtered on best rating, bass tab, standard tuning and rock
url1 = "https://www.ultimate-guitar.com/explore?genres[]=4&order=rating_desc&page="
url2 = "&tuning[]=1&type[]=Bass%20Tabs"
N = 1 # total number of pages to scrape, 100 is maximum

# List to store all the bassTab objects
tokenized_inputs = []

for i in tqdm(range(N)):
    url = url1+str(i)+url2

    # Send a GET request to the current page
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    data_content = soup.find('div', {'class': 'js-store'})['data-content']
    data = json.loads(data_content)
    
    songs_dict = data['store']['page']['data']['data']['tabs']
    
    for songs in songs_dict:
        BT = load_bassTab(url=songs['tab_url'])
        if BT is not None:
            #BT.print_bassline_unique()
            print(BT.artist,':', BT.name)
            tokenized_inputs.append(Embeddings(BT))
            
            
#%% save the tokens
import pickle

# Assuming your tokenized inputs are stored in a list called 'tokenized_inputs'
# Save the list using Pickle
with open('tokenized_inputs.pickle', 'wb') as f:
    pickle.dump(tokenized_inputs, f)


#%%
# Load the list back from the Pickle file
#with open('tokenized_inputs.pickle', 'rb') as f:
#    tokenized_inputs = pickle.load(f)