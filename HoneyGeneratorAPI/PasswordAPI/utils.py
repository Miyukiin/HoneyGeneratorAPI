import os
import sys
import numpy as np
import textdistance
import json
import random
import string

import tensorflow as tf
import keras
from keras.src.utils.sequence_utils import pad_sequences
from keras.src.utils import to_categorical, plot_model
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Embedding
from keras.src.callbacks import EarlyStopping, ModelCheckpoint

from django.conf import settings

class MLHoneywordGenerator:
    def __init__(self, password_list:list[str]=None, embedding_dim=50, lstm_units=100):
        """
        Initialize the Honeyword Generator with parameters.
        """
        self.threshold_damerau_levenshtein:int = 3
        self.num_of_honeywords:int = 5 # How many honeywords to be generated? E.g 5 honeywords -> 4 sweet words, 1 sugar word.
        self.password_list:list[str] = password_list
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.char_to_idx:dict[str,int] = {} 
        self.idx_to_char:dict[int,str] = {}
        self.model = None
        self.max_length_password_list:int = 0
        self.version:str = "honeyword_model_phpbb"
        self.dataset_name:str = "honeyword_dataset_phpbb"
        self.model_file_path:str = os.path.join(settings.BASE_DIR, 'PasswordAPI', 'static', 'PasswordAPI', 'tf_resources', "models", self.version) + ".keras"
        self.inp, self.out = None, None
        self.epochs = 100
        self.batch_size = 128
        self.seed_text_length = 3 # Zero-based, Splicing

        # Only prepare dataset if passwords are provided
        if password_list:
            self._prepare_dataset()
            self.build_model()
            self.train()
        else:
            self._prepare_dataset()
    
    def visualize_model(self):
        visualization_folder_path = os.path.join(settings.BASE_DIR, 'PasswordAPI', 'static', 'PasswordAPI', 'tf_resources', 'visualization')
        file_name = 'honey_model_architecture.png'
        
        if self.model == None:
            try:
                self.model = keras.models.load_model(self.model_file_path)
            except Exception as e:
                raise Exception(f"Can not load model. (Path:./tf_resources/models/{self.version}.keras) (Reason: {str(e)})")
        
        plot_model(
            self.model,
            to_file=os.path.join(visualization_folder_path, file_name),
            show_shapes=True,
            show_layer_names=True
        )

    def _prepare_dataset(self):
        """
        Prepare the dataset by analyzing passwords, generating sequences, 
        and creating input-output pairs.
        """
        dataset_file_path = os.path.join(settings.BASE_DIR, 'PasswordAPI', 'static', 'PasswordAPI', 'tf_resources', "datasets", self.dataset_name) + ".txt"
        
        def _save_dataset():
            """
            Save preprocessed dataset and mappings to files.
            """
            # Save mappings and metadata as JSON
            with open(dataset_file_path, "w") as f:
                json.dump({
                    "char_to_idx": self.char_to_idx,
                    "idx_to_char": self.idx_to_char,
                    "max_length_password_list": self.max_length_password_list,
                    "inp": self.inp.tolist(),
                    "out": self.out.tolist() 
                }, f)

        def _load_dataset():
            """
            Load preprocessed dataset and mappings from files if they exist.
            """
            try:
                with open(dataset_file_path, "r") as f:
                    mappings: dict[str, dict | int | list]  = json.load(f)
                    self.char_to_idx = mappings["char_to_idx"]
                    self.idx_to_char = {int(idx): char for idx, char in mappings["idx_to_char"].items()}
                    self.max_length_password_list = mappings["max_length_password_list"]
                    self.inp = np.array(mappings["inp"]) 
                    self.out = np.array(mappings["out"])  
                return True
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                return False
        
        if _load_dataset(): # Load, else run rest of code.
            return
        
        # Set all possible characters in a password.
        chars = sorted(
                    list(string.ascii_letters)  + 
                    list(string.punctuation) + 
                    list(string.digits)
                )

        # Generate character-to-index and index-to-character mappings
        self.char_to_idx:dict[str,int] = {char: idx for idx, char in enumerate(chars)} # {'char': idx}  
        self.idx_to_char:dict[int,str] = {idx: char for char, idx in self.char_to_idx.items()} # {'idx': char}

        # Convert passwords to sequences of indices
        sequences:list[list[int]] = [] 
        for password in self.password_list:
            seq:list[int] = [self.char_to_idx[char] for char in password] # example mapping is word = [25,23,52,32]
            sequences.append(seq)
            
        # Create input-output pairs
        self.max_length_password_list = max(len(seq) for seq in sequences) # Find the longest password length in given dataset
        inp:list[list[int]] =  []
        out:list[int] = []
        for seq in sequences: # each password-index sequence
            for i in range(1, len(seq)): # create subsequence
                inp.append(seq[:i])
                out.append(seq[i])

        # Pad sequences and one-hot encode outputs
        self.inp = pad_sequences(inp, maxlen=self.max_length_password_list, padding='post') # Ensure all inputs have same length by padding using 0
        self.out = to_categorical(out, num_classes=len(self.char_to_idx))
        
        # Save dataset for future use
        _save_dataset()

    def build_model(self):
        """
        Build the LSTM model for honeyword generation.
        """
        self.model = Sequential([
            Embedding(input_dim=len(self.char_to_idx), # Vocabulary Size, length of unique chars
                      output_dim=self.embedding_dim, # Embedding Dimensions, default 50 dimensional vector
                      input_length=self.max_length_password_list), # Consistent length
            LSTM(self.lstm_units, return_sequences=True), # Identify temporal dependencies or patterns, returns hidden state sequences
            LSTM(self.lstm_units), # Return a single vector representing the entire input sequence.
            Dense(len(self.char_to_idx), activation='softmax') # Generate probability distribution, probability of a specific character being the next in the sequence.
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy') # Specify optimizer and loss function

    def train(self, epochs=100, batch_size=32):
        """
        Train the LSTM model.
        """
        if not self.model:
            raise ValueError("Model has not been built. Call build_model() first.")
        
        # Define early stoppage, halt training when the validation loss stops improving.
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True, 
            verbose=1) 
        
        # Define checkpoint callback to save the model after each epoch
        checkpoint_file_path = os.path.join(settings.BASE_DIR, 'PasswordAPI', 'static', 'PasswordAPI', 'tf_resources', "model_checkpoints", f"{self.version}_epoch{{epoch:02d}}") + ".keras"
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=False,  # Save the entire model
            verbose=1,
            save_freq = "epoch"
        )
        
        # Training of model
        self.model.fit(
            self.inp, 
            self.out, 
            epochs=self.epochs, 
            validation_split = 0.2,
            batch_size=self.batch_size, 
            verbose=1, 
            callbacks=[early_stopping,checkpoint]
        )
        
        # Visualize Model
        self.visualize_model() 
        
        # Save after the entire training process finishes.
        self.model.save(self.model_file_path)


    def generate_honeyword(self, seed_text:str, sugarword_length:int, temperature=0.7) -> str:
        """
        Generate a honeyword based on the seed text with added randomness.
        """
        def sample_with_temperature(predictions, temperature=1.0):
            """
            Sample an index from a probability distribution with temperature scaling.
            """
            predictions = np.log(predictions + 1e-8) / temperature  # Avoid log(0)
            exp_preds = np.exp(predictions)
            probabilities = exp_preds / np.sum(exp_preds)
            return np.random.choice(len(probabilities), p=probabilities)
        
        def load_model():
            try:
                self.model = keras.models.load_model(self.model_file_path)
            except Exception as e:
                raise Exception(f"Can not load model. (Path:./tf_resources/models/{self.version}.keras) (Reason: {str(e)})")
            
        honeyword = seed_text
        
        if not self.model:
            load_model()  # Load the model only if not already loaded
        for _ in range(sugarword_length - len(seed_text)):
            input_seq = [self.char_to_idx[char] for char in honeyword[-self.max_length_password_list:]] # Negative Indexing in the case that entered Password Longer Case, and Password Shorter Case than longest assword in list.
            input_seq = pad_sequences([input_seq], maxlen=self.max_length_password_list, padding='post')
            predictions = self.model.predict(input_seq, verbose=0)[0]
            next_index = sample_with_temperature(predictions, temperature)
            next_char = self.idx_to_char[next_index]
            # print(next_char) #Debugging
            honeyword += next_char
        return honeyword

    def generate_honeywords(self, sugarword) -> tuple[list,int]:
        """
        Generate multiple honeywords for a given sugarword.
        """
        seed_text = sugarword[:self.seed_text_length]  # Start with a seed text
        honeyword_list = []

        while len(honeyword_list) < self.num_of_honeywords-1:
            
            honey_password_candidate = self.generate_honeyword(seed_text, len(sugarword))
            lev_distance_between_candidate_and_orig = textdistance.damerau_levenshtein(honey_password_candidate, sugarword) # Damerau-Levenshtein Assessment
            # print(honey_password_candidate, lev_distance_between_candidate_and_orig) # Debugging
            if lev_distance_between_candidate_and_orig >= self.threshold_damerau_levenshtein and honey_password_candidate not in honeyword_list:
                honeyword_list.append(honey_password_candidate)
            else:
                continue
        
        honeyword_list.append(sugarword)
        random.shuffle(honeyword_list) # Randomize positions
        sugarword_index = honeyword_list.index(sugarword) # Find the index of the sugarword for the API HoneyChecker 
        return honeyword_list, sugarword_index

    
class ExistingPasswordGeneration:
    def __init__(self, password:str):
        self.password:str = password
        self.num_of_honeywords = 5 # How many honeywords to be generated? E.g 5 honeywords -> 4 sweet words, 1 sugar word.
        self.alphabet_upper_list = list(string.ascii_uppercase)
        self.alphabet_lower_list = list(string.ascii_lowercase)
        self.symbol_list = list(string.punctuation)
        self.digit_list =  list(string.digits)
        self.honeyword_list:list[str] = []
    
    def is_unique_and_not_sugarword_or_list_empty(self, honeyword):
        # If list is empty, return true or if honeyword is not in self.honeyword_list and honeyword is not the password
        if not self.honeyword_list or (honeyword not in self.honeyword_list and honeyword != self.password):
            return True
        else:
            return False
        
    def choose_replacement_character(self, character: str) -> str:
        if character.isdigit():
            return str(random.choice(self.digit_list))
        elif character.isalpha():
            if character.isupper():
                return str(random.choice(self.alphabet_upper_list))
            else:
                return str(random.choice(self.alphabet_lower_list))
        else:
            return str(random.choice(self.symbol_list))
    
    def choose_method(self, choice: int):
        self.honeyword_list = []
        match choice:
            case 1: # Chaffing by Tail-Tweaking
                self.num_of_tweaks = 2 # How many characters from the end will be tweaked.
                
                while len(self.honeyword_list) < self.num_of_honeywords - 1: # Generate 4 sweet words
                    self.password_characters:list[str] = [*self.password] # Create a list of characters out of the password
                    
                    for i in range(self.num_of_tweaks):
                        self.password_characters[len(self.password_characters)-(i+1)] = (
                            self.choose_replacement_character(self.password_characters[len(self.password_characters)-(i+1)])
                        )
                        
                    self.possible_sweetword = "".join(self.password_characters)
                    if self.is_unique_and_not_sugarword_or_list_empty(self.possible_sweetword):
                        self.honeyword_list.append(self.possible_sweetword)
                        
                self.honeyword_list.append(self.password) # Append sugarword
                random.shuffle(self.honeyword_list) # Randomize positions
                sugarword_index = self.honeyword_list.index(self.password) # Find the index of the sugarword for the API HoneyChecker 
                return self.honeyword_list, sugarword_index
            
            case 2: # Take-a-tail
                self.num_of_tail_char = 3 # How many digits do you like to append to your password?
                self.password_characters:list[str] = [*self.password]
                
                for i in range(self.num_of_tail_char): # Append digits of length num_of_tail_char
                    self.password_characters.append(random.choice(self.digit_list))
                
                self.appended_password:str = "".join(self.password_characters) # join the characters with appended digits as the new password.
                
                while len(self.honeyword_list) < self.num_of_honeywords - 1: # Generate 4 sweet words
                    self.password_characters:list[str] = [*self.appended_password] # Create a list of characters out of the password
                    
                    for i in range(self.num_of_tail_char):
                        self.password_characters[len(self.password_characters)-(i+1)] = (
                            self.choose_replacement_character(self.password_characters[len(self.password_characters)-(i+1)])
                        )
                        
                    self.possible_sweetword = "".join(self.password_characters)
                    if self.is_unique_and_not_sugarword_or_list_empty(self.possible_sweetword):
                        self.honeyword_list.append(self.possible_sweetword)
                        
                self.honeyword_list.append(self.appended_password) # Append sugarword
                random.shuffle(self.honeyword_list) # Randomize positions
                sugarword_index = self.honeyword_list.index(self.appended_password) # Find the index of the sugarword for the API HoneyChecker 
                return self.honeyword_list, sugarword_index
                    
            case 3: # Chaffing with a Password-model
                file_path =  os.path.join('utilities', 'HPGResources', 'password lists', "password_list.txt")
                with open( file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    wordlist = [line.strip() for line in file]
                    
                while len(self.honeyword_list) < self.num_of_honeywords - 1:
                    w = random.choice(wordlist)
                    d = len(w)  # Determine the length of the chosen password
                    
                    # Initialize the new password (honeyword) as a list of characters
                    sweetword_candidate = [w[0]]  # Start with c1 = w1 (first character is directly copied)

                    # Step 2: Generate each character cj for j = 2, 3, ..., d
                    for j in range(1, d):  # Using range(1, d) because we've already added the first character
                        probability = random.random()  # Generate a random number between 0 and 1 for the probability.

                        if probability < 0.1:
                            # With probability 0.1, pick a new password of the same length randomly and take the jth character
                            new_password = random.choice([p for p in wordlist if len(p) == d])
                            sweetword_candidate.append(new_password[j])
                        elif probability < 0.5:
                            # With probability 0.4, pick a password that matches the previous character and take the jth character
                            new_password = random.choice([p for p in wordlist if len(p) == d and p[j - 1] == sweetword_candidate[j - 1]])
                            sweetword_candidate.append(new_password[j])
                        else:
                            # With probability 0.5, keep the jth character the same as in the original password
                            sweetword_candidate.append(w[j])
                    
                    self.possible_sweetword = "".join(sweetword_candidate)
                    if self.is_unique_and_not_sugarword_or_list_empty(self.possible_sweetword):
                        self.honeyword_list.append(self.possible_sweetword)
                        
                self.honeyword_list.append(self.password) # Append sugarword
                random.shuffle(self.honeyword_list) # Randomize positions
                sugarword_index = self.honeyword_list.index(self.password) # Find the index of the sugarword for the API HoneyChecker 
                return self.honeyword_list, sugarword_index
            case _:
                raise Exception("Choose Method. Choice not in range.")
        
# Used to generate the 4 methods' 5 password 5 honeywords each. Can also be used to test all or any of the methods.
def generate_survey_passwords_honeywords() -> dict:
    # Structure {'method_passwords: {'password1':[honeyword,honeyword,honeyword,honeyword,honeyword]},{'password2':[honeyword,honeyword,honeyword,honeyword,honeyword]}}
    methods_passwords_dict = {} 
    
    def generate_5_Juels(passwordlist, num):
        password_honeywords_dict = {}
        for password in passwordlist:
            instance = ExistingPasswordGeneration(password)
            honey_word_list, sugarindex = instance.choose_method(num)
            password_honeywords_dict[password] = [honey_word_list, sugarindex] 
        return password_honeywords_dict
    
    def generate_5_ML(passwordlist):
        password_honeywords_dict = {}
        for password in passwordlist:
            instance = MLHoneywordGenerator()
            honey_word_list, sugarindex = instance.generate_honeywords(password)
            password_honeywords_dict[password] = [honey_word_list, sugarindex]
        return password_honeywords_dict
            
    CBTW_passwords = ["altinordu52","HPw2207!@#","MIlkSHake1","Makkadeh!@#","jeff@farm1"]
    TAT_passwords =["riaflashmx846","1441262626027","lamurg0d694","#ector@1258","swordhunter583"]
    CPM_passwords = ["hazo996655","th3matr1x","papa072112","dovesciare9090","EAfiGI5389"]
    ML_passwords = ["DraiksRock251","diabolo666","letmeloveyou","password8169","rhfm281010"]
    
    methods_passwords_dict['CBTW'] = generate_5_Juels(CBTW_passwords, 1)
    methods_passwords_dict['TAT'] = generate_5_Juels(TAT_passwords, 2)
    methods_passwords_dict['CPM'] = generate_5_Juels(CPM_passwords, 3)
    methods_passwords_dict['ML'] = generate_5_ML(ML_passwords)
    
    # Printing
    for key1,value1 in methods_passwords_dict.items():
        print(f'\n{key1}')
        for key2,value2 in value1.items():
            print(f"   {key2} -> {value2}")
        
        
# Notes
    """
    # instance = ExistingPasswordGeneration("Ryan123")
    # honey_word_list, sugarindex = instance.choose_method(1)
    
    instance = ProposedPasswordGeneration("Ryan123")
    honey_word_list, sugarindex = instance.generate_honeyword_list()
    print(f"The honey word list is {honey_word_list}, and the sugarindex is {sugarindex}")
    """
    
    """
    # Temporary, reads passwordlist to pass to honeywordgenerator
    passwords = []
    file_path = Path.cwd() / 'static' / 'password lists' / 'phpbb-cleaned-up-listed-python.json'
    
    with file_path.open('r', encoding='utf-8', errors='ignore') as file:
        passwords = json.load(file)
    """
    
    """
    # Initialize and train the ML Honeyword Generator
    generator = MLHoneywordGenerator()
    print(generator.generate_honeywords("K02W0wG0dD4mN"))
    """
    
    """
    # Generate Honeywords for the survey
    generate_survey_passwords_honeywords()
    """

# The command "python manage.py run_utils" runs this function. Input the necessary logic to be called. By default contains Logic to create honeypassword
def run_utils():
    generate_survey_passwords_honeywords()
    