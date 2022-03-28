import os
import pickle
import random

from collections import defaultdict

class AbbrInfo:
    """
    Определения:
    * ('теория', 'механизм', 'и', 'машина') - desc
    * 'тмм' - abbr
    * 0 - id
    """
    
    def __init__(self, 
                 tokenized_abbr: dict = None, 
                 data_dir: str = None):
        if tokenized_abbr is not None:
            self.create_dicts(tokenized_abbr)
        elif data_dir is not None:
            self.load_dicts(data_dir)
        
    ### Dict tools ###
        
    def create_dicts(self, tokenized_abbr: dict):
        self.id2abbr = {id_: abbr for id_, (desc, abbr) in enumerate(tokenized_abbr.items())}
        self.id2desc = {id_: desc for id_, (desc, abbr) in enumerate(tokenized_abbr.items())}

        self.desc2abbr = {desc: abbr for id_, (desc, abbr) in enumerate(tokenized_abbr.items())}
        self.desc2id = {desc: id_ for id_, (desc, abbr) in enumerate(tokenized_abbr.items())}

        self.abbr2id_list = defaultdict(list)
        for id_, abbr in self.id2abbr.items():
            self.abbr2id_list[abbr].append(id_)

        self.abbr2desc_list = defaultdict(list)
        for desc, abbr in self.desc2abbr.items():
            self.abbr2desc_list[abbr].append(desc)
    
    def save_dicts(self, data_dir: str):
        with open(os.path.join(data_dir, "id2abbr.pickle"), "wb") as f:
            pickle.dump(self.id2abbr, f)
        with open(os.path.join(data_dir, "id2desc.pickle"), "wb") as f:
            pickle.dump(self.id2desc, f)
            
        with open(os.path.join(data_dir, "desc2abbr.pickle"), "wb") as f:
            pickle.dump(self.desc2abbr, f)
        with open(os.path.join(data_dir, "desc2id.pickle"), "wb") as f:
            pickle.dump(self.desc2id, f)
            
        with open(os.path.join(data_dir, "abbr2id_list.pickle"), "wb") as f:
            pickle.dump(self.abbr2id_list, f)
        with open(os.path.join(data_dir, "abbr2desc_list.pickle"), "wb") as f:
            pickle.dump(self.abbr2desc_list, f)
            
    def load_dicts(self, data_dir: str):
        with open(os.path.join(data_dir, "id2abbr.pickle"), "rb") as f:
            self.id2abbr = pickle.load(f)
        with open(os.path.join(data_dir, "id2desc.pickle"), "rb") as f:
            self.id2desc = pickle.load(f)
            
        with open(os.path.join(data_dir, "desc2abbr.pickle"), "rb") as f:
            self.desc2abbr = pickle.load(f)
        with open(os.path.join(data_dir, "desc2id.pickle"), "rb") as f:
            self.desc2id = pickle.load(f)
            
        with open(os.path.join(data_dir, "abbr2id_list.pickle"), "rb") as f:
            self.abbr2id_list = pickle.load(f)
        with open(os.path.join(data_dir, "abbr2desc_list.pickle"), "rb") as f:
            self.abbr2desc_list = pickle.load(f)

    
class AbbrTree:
    ID_KEY = "<id>"
    
    def __init__(self, abbr_info: AbbrInfo):
        self.abbr_info = abbr_info
        self.desc_tree = self.get_desc_tree()
            
    def get_desc_tree(self):
        desc_tree_root = dict()
        curr_node = desc_tree_root
        for desc, id_ in self.abbr_info.desc2id.items():

            for word in desc:
                if word not in curr_node:
                    curr_node[word] = dict()    
                curr_node = curr_node[word]

            curr_node[self.ID_KEY] = id_
            curr_node = desc_tree_root

        return desc_tree_root
    
    def get_text_labels(self, text: list):
        labels = [0 for i in range(len(text))]

        curr_node = self.desc_tree
        desc_start = None
        for word_i, word in enumerate(text):
            if word in curr_node: 
                if desc_start is None: 
                    desc_start = word_i
                curr_node = curr_node[word]

            else: 
                if self.ID_KEY in curr_node: 
                    for j in range(desc_start, word_i): 
                        labels[j] = curr_node[self.ID_KEY]

                desc_start = None
                curr_node = self.desc_tree

        return labels
    
    def replace_by_abbr(self, text: list, p: int = 0.5):
        labels = self.get_text_labels(text)
        
        new_text = []
        new_labels = []

        i = 0
        while i < len(text):
            label = labels[i]
            if label != 0:
                if random.choices([False, True], weights=[(1 - p), p])[0]:
                    new_text.append(self.abbr_info.id2abbr[label])
                    new_labels.append(label)
                    while label == labels[i] and i < len(text):
                        i += 1
                else:
                    while label == labels[i] and i < len(text):
                        new_text.append(text[i])
                        new_labels.append(0)
                        i += 1
            else:
                new_text.append(text[i])
                new_labels.append(0)        
                i += 1
                
        return new_text, new_labels