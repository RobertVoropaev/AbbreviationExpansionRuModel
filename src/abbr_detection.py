import string

from collections import defaultdict

class AbbrDetection:
    def __init__(self):
        self.word2info = self.load_opencorpora()
        self.word2type = self.get_word2type()
        
    def load_opencorpora(self, opencorpora_path = "../input/dict.opcorpora.txt"):
        word2info = defaultdict(list)
        with open(opencorpora_path) as f:
            for i, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) > 1:
                    word = line[0]
                    info = line[1].split(",")
                    word2info[word.lower()].append(tuple(info))
        return word2info
    
    def get_word2type(self):
        word2type = {}
        for word, info_list in self.word2info.items():
            abbr_flag = False
            for info in info_list:
                if "Abbr" in " ".join(info):
                    abbr_flag = True
                    break

            if abbr_flag:
                word2type[word] = "Abbr"
            else:
                word2type[word] = "Word"
        return word2type
    
    def word_is_russian(self, word):
        for char in word:
            if char in string.ascii_letters:
                return False
        return True
    

    
    def word_is_abbr(self, word):
        is_abbr = self.word2type.get(word)
        if is_abbr == "Abbr":
            return True
        elif is_abbr == "Word":
            return False
        else: # None
            if word.isdigit():
                return False
            
            if self.word_is_russian(word):
                return True
            else:
                return False
                
    
    def word_is_full(self, word):
        return not self.word_is_abbr(word)