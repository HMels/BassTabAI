# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:10:07 2023

@author: Mels
"""
class BassTab:
    def __init__(self, E=[], A=[], D=[], G=[], genre=[], name="", artist="", tonality=""):
        '''
        Initializes a new BassTab object with the specified basslines, genre, song name, and artist.

        Parameters
        ----------
        E : list, optional
            List of basslines on the E string, by default []
        A : list, optional
            List of basslines on the A string, by default []
        D : list, optional
            List of basslines on the D string, by default []
        G : list, optional
            List of basslines on the G string, by default []
        genre : list, optional
            List of genres for the song, by default []
        name : str, optional
            Name of the song, by default ""
        artist : str, optional
            Name of the artist, by default ""

        Returns
        -------
        None.
        '''
        self.E = [] # bassline on the E string
        self.A = [] # bassline on the A string
        self.D = [] # bassline on the D string
        self.G = [] # bassline on the G string
        self.genre = genre
        self.name = name
        self.artist = artist
        self.tonality = tonality
        self.repeat_indices = [] # references the bassline indexes in the total composition
        
        self.check_inputs()


    def check_inputs(self):
        if not all(isinstance(bassline, list) for bassline in [self.E, self.A, self.D, self.G]):
            raise TypeError("Basslines must be lists")
        
        if not all(isinstance(genre, str) for genre in self.genre):
            raise TypeError("Genres must be strings")
            
        if not isinstance(self.name, str):
            raise TypeError("Song name must be a string")
            
        if not isinstance(self.artist, str):
            raise TypeError("Artist must be a string")
            
        if not isinstance(self.tonality, str):
            raise TypeError("Tonality must be a string")
            
        # Check that all basslines are the same length.
        bassline_len = len(self.E)
        if len(self.A) != bassline_len or len(self.D) != bassline_len or len(self.G) != bassline_len:
            raise ValueError("All basslines must be the same length")
            
            
    def append_bassline(self, E_new, A_new, D_new, G_new, repeat=None):
        '''
        Appends a new bassline to the BassTab object.

        Parameters
        ----------
        E_new : list of str
            The new bassline on the E string.
        A_new : list of str
            The new bassline on the A string.
        D_new : list of str
            The new bassline on the D string.
        G_new : list of str
            The new bassline on the G string.
        repeat : int, optional
            The number of times the bassline is repeated, by default None.

        Returns
        -------
        bool
            True if the bassline is successfully appended, False otherwise.
        '''
        index = self.check_bassline(E_new, A_new, D_new, G_new)
        
        if index == -1: # this means the bassline is unique
            bassline_len = len(E_new) # Check that the length of the new basslines is the same as the existing ones.
            if len(A_new) != bassline_len or len(D_new) != bassline_len or len(G_new) != bassline_len:
                return False
            
            # If the new basslines are unique, append them to the corresponding arrays and update the repeat_indices array.
            self.E.append(E_new)
            self.A.append(A_new)
            self.D.append(D_new)
            self.G.append(G_new)
        
        # Add the index of the new bassline to the repeat_indices array.
        if repeat is None:
            if index == -1: self.repeat_indices.append(len(self.E)) 
            else: self.repeat_indices.append(index) 
        else:
            for i in range(repeat):
                if index == -1: self.repeat_indices.append(len(self.E)) 
                else: self.repeat_indices.append(index)
                
    def append_empty(self, repeat=None):
        if repeat is None: 
            self.repeat_indices.append(0)
        else:
            for i in range(repeat):
                self.repeat_indices.append(0)
        
        
    def check_bassline(self, E_new, A_new, D_new, G_new):
        '''
        Checks if the new basslines are already in the BassTab object.

        Parameters
        ----------
        E_new : list of str
            The new bassline on the E string.
        A_new : list of str
            The new bassline on the A string.
        D_new : list of str
            The new bassline on the D string.
        G_new : list of str
            The new bassline on the G string.

        Returns
        -------
        int
            The index of the matching bassline if it is already in the BassTab object, -1 otherwise.
        '''
        for i in range(len(self.E)):
            if E_new == self.E[i] and A_new == self.A[i] and D_new == self.D[i] and G_new == self.G[i]:
                # If the new basslines already exist in the corresponding arrays, return the index.
                return i
        return -1
    
    
    def print_bassline(self):
        print(self.artist,"-",self.name)
        for i in self.repeat_indices:
            if i!=0:
                print("G"+self.G[i-1])
                print("D"+self.D[i-1])
                print("A"+self.A[i-1])
                print("E"+self.E[i-1],'\n')
            else: print('SKIP LINE!')
            
    
    def print_bassline_unique(self):
        print(self.artist,"-",self.name)
        for i in range(len(self.A)):
            print("G"+self.G[i])
            print("D"+self.D[i])
            print("A"+self.A[i])
            print("E"+self.E[i],'\n')
            
            
    
    def tokenize_bar(self, G, D, A, E):
        strings = [E, A, D, G]
        bar_counts = []
        bar_len = len(G)
        
        i=0
        while i < bar_len:
            count = ["-", "-", "-", "-", ""]
            doubledigits = False
            containsspecial = False
            ghostnote = False
            
            for j in range(4):
                # double digits smaller than 20
                if (i < bar_len-1 and strings[j][i] in "0123456789" and strings[j][i+1] in "0123456789" 
                    and int(strings[j][i]+strings[j][i+1])<=20):
                    doubledigits=True
                    count[j] = strings[j][i]+strings[j][i+1] 
                    
                    # special characters
                    if i < bar_len-2 and strings[j][i+2] in "/\sShHpPbB^~*":
                        containsspecial = True
                        if strings[j][i+2] in "/\sS":
                            count[4] = "s"
                        elif strings[j][i+2] in"Hh":
                            count[4] = "h"
                        elif strings[j][i+2] in "pP":
                            count[4] = "p"
                        elif strings[j][i+2] in "bB^":
                            count[4] = "b"
                        else:
                            count[4] = strings[j][i+2]
                        
                # single digits
                elif strings[j][i] in "0123456789":
                    count[j] = strings[j][i]
                    
                    # special characters
                    if i < bar_len-1 and strings[j][i+1] in "/\sShHpPbB^~*":
                        containsspecial = True
                        if strings[j][i+1] in "/\sS":
                            count[4] = "s"
                        elif strings[j][i+1] in"Hh":
                            count[4] = "h"
                        elif strings[j][i+1] in "pP":
                            count[4] = "p"
                        elif strings[j][i+1] in "bB^":
                            count[4] = "b"
                        else:
                            count[4] = strings[j][i+1]
                    
                # dead note
                elif strings[j][i] in "x":
                    count[j] = "x"
                    
                # ghost note
                elif strings[j][i] in "(":
                    count[4] = "g"
                    ghostnote = True
                    if ( strings[j][i+1] in "0123456789" and strings[j][i+2] in "0123456789"
                        and int(strings[j][i+1]+strings[j][i+2])<=20):
                        doubledigits=True
                        count[j] = strings[j][i+1]+strings[j][i+2]                         
                    elif strings[j][i+1] in "0123456789":
                        count[j] = strings[j][i+1]
                        
                # bars 
                elif strings[j][i] in "|":
                    count[j] = "|"
                    
                
            bar_counts.append(count)
            
            # double digits and special character means an extra count needs to be skipped
            i+=1
            if doubledigits: i+=1
            if containsspecial: i+=1
            if ghostnote: i+=2     
            
        return bar_counts
    
    
    def tokenize(self):
        self.token = []
        for i in range(len(self.A)):
            self.token.append( self.tokenize_bar(self.G[i],self.D[i],self.A[i],self.E[i]) )
            
    
    
    def detokenize_bar(self, bar_counts):
        G = ""
        D = ""
        A = ""
        E = ""
        
        for count in bar_counts:
            if len(count[0])==2 or len(count[1])==2 or len(count[2])==2 or len(count[3])==2: dash="--"
            else: dash="-"
            
            # contains special characters
            if len(count[4])!=0 and count[4] in "/\sShHpPbB^~*":
                if count[0].isdigit(): E += count[0]+count[4]
                else: E += dash+'-'
                if count[1].isdigit(): A += count[1]+count[4]
                else: A += dash+'-'
                if count[2].isdigit(): D += count[2]+count[4]
                else: D += dash+'-'
                if count[3].isdigit(): G += count[3]+count[4]
                else: G += dash+'-'
                
            # does not contain special characters
            else:
                if count[0].isdigit(): E += count[0]
                elif count[0]=='|': E += '|'
                else: E += dash
                if count[1].isdigit() or count[1]=='|': A += count[1]
                elif count[0]=='|': A += '|'
                else: A += dash
                if count[2].isdigit() or count[2]=='|': D += count[2]
                elif count[0]=='|': D += '|'
                else: D += dash
                if count[3].isdigit() or count[3]=='|': G += count[3]
                elif count[0]=='|': G += '|'
                else: G += dash
                    
        return G, D, A, E


            
            
    
    
    
def parse_tab_line(line):
    """
    Parses a single line of a tab and returns the notes until the third "|".
    If there are characters after the third "|" sign, it returns them separately as a repeat.

    Parameters
    ----------
    line : str
        The tab line to be parsed.

    Returns
    -------
    output : str
        The notes in the line until the third "|".
    repeat : int or None
        The repeat count of the notes after the third "|" sign. If no repeat count is found, returns None.
    isempty : bool
        Is true when the complete bassline did not contain any notes
    """
    # Initialize variables
    output = ""
    pipe_count = 0
    remaining = ""
    isempty = True # will be used to check if there are digits in the bassline
    
    # delete double ||
    result = ''
    for i in range(len(line)):
        if line[i] == '|' and i < len(line)-1 and line[i+1] == '|':
            continue  # skip the second '|' character
        result += line[i]
    line = result
    
    maxpipe_count = line.count("|")
    
    # Loop through characters in the line
    for c in line:
        if c == "|":
            pipe_count += 1
            if pipe_count == maxpipe_count:
                # Save remaining characters after third "|" sign
                if maxpipe_count==2: remaining = line[line.index("|", line.index("|") + 1) + 1:]
                else: remaining = line[line.index("|", line.index("|", line.index("|") + 1) + 1) + 1:]
                output += c
                break
        output += c
        if c.isdigit(): isempty = False
        
    # Check for repeat count in remaining characters
    repeat = 1
    for r in remaining:
        if r == "x":
            if remaining[remaining.index("x")+2].isdigit():
                repeat = int(remaining[remaining.index("x")+1] + remaining[remaining.index("x")+2])
            elif remaining[remaining.index("x")+1].isdigit():
                repeat = int(remaining[remaining.index("x")+1])
            else: repeat = None
            break
                
    return output, repeat, isempty