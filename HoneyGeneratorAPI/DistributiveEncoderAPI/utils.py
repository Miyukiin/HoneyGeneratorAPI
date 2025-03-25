import base64
import json
import os
from collections import Counter
import heapq
import math
import random
import bitarray
import itertools
import base64
from django.conf import settings
import hashlib
import argon2
from Crypto import Random
from .RBMRSA import try_generating_keys, try_eea_mod, try_decryption_crt, try_bitstuffing, try_destuffing, try_binary_conversion

class Compressor():  
    ### lzw METHODS ###
    def lzw_encode(self, text_to_encode:str):
        # Reference Section 5.2 Page 34 of Introduction to Data Compression by Blelloch Guy E.
        
        # Initialize dictionary with single character strings
        dictionary = {chr(i): i for i in range(256)}
        # Start of the Next Code. Immediately after 0-255 ASCII Codes.
        next_code = 256  
        
        w = ""
        lzw_compressed_data:list = []

        # Every unique pattern, add to the dictionary.
        for c in text_to_encode:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                lzw_compressed_data.append(dictionary[w])
                dictionary[wc] = next_code
                next_code += 1
                w = c
        if w:
            lzw_compressed_data.append(dictionary[w])
        return lzw_compressed_data # Returns a List of Integers where instead of characters, they are represented by the integer unique pattern as specified in the dictionary.

    def lzw_decode(self, compressed_lzw_data:list[int]):
        # Initialize dictionary with single character strings
        dictionary = {i: chr(i) for i in range(256)}
        # Start of the Next Code. Immediately after 0-255 ASCII Codes.
        next_code = 256 
        w = chr(compressed_lzw_data.pop(0))
        decompressed_data = w

        for code in compressed_lzw_data:
            if code in dictionary:
                entry = dictionary[code]
            elif code == next_code:
                entry = w + w[0]  # Special case for newly formed entries
            else:
                raise ValueError("Decompression error: Invalid LZW code")

            decompressed_data += entry
            dictionary[next_code] = w + entry[0]
            next_code += 1
            w = entry

        return decompressed_data
    
    ### HUFFMAN METHODS ###
    class HuffmanNode:
        counter = itertools.count()  # global class-level counter so that heap remains consistent; when frequencies are the same, they are not arbitrarily chosen when rebuilt.
        def __init__(self, symbol, freq):
            self.symbol = symbol
            self.freq = freq
            self.left = None
            self.right = None
            self.order = next(Compressor.HuffmanNode.counter)  # unique tiebreaker to becom deterministic when rebuilding tree during iteration process

        def __lt__(self, other): # heap ordering logic
            if self.freq == other.freq:
                return self.order < other.order  # break ties by insertion order (deterministic), because comparison by Int and None will yield error.
            return self.freq < other.freq
        
        def __repr__(self):
            return f"({self.symbol}:{self.freq}:{self.order})"
        
        def print_tree(self, node, indent="", branch="Root"):
            if node is not None:
                self.print_tree(node.right, indent + "     ", branch="R──")
                if node.symbol is not None:
                    print(f"{indent}{branch} ('{chr(node.symbol)}'/{node.symbol}:{node.freq})")
                else:
                    print(f"{indent}{branch} (None:{node.freq})")
                self.print_tree(node.left, indent + "     ", branch="L──")
            
    def build_frequency_table(self, data:list[int]):
        frequency_table = Counter(data) 
        return frequency_table
    
    def build_huffman_tree(self, freq_table:dict):
        # Initialize
        heap = [self.HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
        heapq.heapify(heap)

        # Iterate
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = self.HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)
            
        # Terminate
        return heap[0]

    def build_huffman_codes(self, node:HuffmanNode, prefix="", codebook=None, reverse_codebook=None):
        """Generate Huffman codes and store both forward and reverse mappings."""
        if codebook is None:
            codebook = {}
        if reverse_codebook is None:
            reverse_codebook = {}

        if node:
            if node.symbol is not None:  # We are at a Leaf node
                codebook[node.symbol] = prefix
                reverse_codebook[prefix] = node.symbol  # Store for decoding
            self.build_huffman_codes(node.left, prefix + "0", codebook, reverse_codebook)
            self.build_huffman_codes(node.right, prefix + "1", codebook, reverse_codebook) 
        return codebook, reverse_codebook
    
    def huffman_encode(self, lzw_compressed_data:list[int], codebook:dict):
        self.huffman_encoded_data = "".join(codebook[symbol] for symbol in lzw_compressed_data)
        return self.huffman_encoded_data # Returns a string containing binary numbers.
    
    def huffman_decode(self, huffman_encoded_data, reverse_codebook):
        """Decode Huffman-encoded bitstring using the reverse codebook."""
        current_code = ""
        decoded_output = []
        
        for bit in huffman_encoded_data:
            current_code += bit
            if current_code in reverse_codebook:  # Check if a valid symbol
                decoded_output.append(reverse_codebook[current_code])
                current_code = ""  # Reset for next symbol
        return decoded_output  # Returns list of LZW codes

    
    ### Binary Conversion Methods ###
    def huffman_to_bytes(self, binary_string:str):
        # Ensure the length is a multiple of 8 by padding
        padding_length = (8 - len(binary_string) % 8) % 8
        padded_binary = binary_string + "0" * padding_length

        # Store padding length in the first byte for decoding
        padding_info = "{:08b}".format(padding_length)  # 8-bit representation of padding length
        padded_binary = padding_info + padded_binary  # Store padding at the start

        # Convert binary string to bytearray
        byte_array = bytearray()
        for i in range(0, len(padded_binary), 8):
            byte_array.append(int(padded_binary[i:i+8], 2))
    
        bytes_huffman_output = bytes(byte_array)
        return bytes_huffman_output
    
    def bytes_to_huffman(self, byte_data):
        # print(base64.b64decode("ASCIIDATA HERE")) yields byte data.
        # Extract padding info from the first byte
        padding_length = byte_data[0]  # First byte stores padding length
        
        # Convert the remaining bytes to binary string
        binary_string = "".join(f"{byte:08b}" for byte in byte_data[1:])  # Skip padding byte, assumes 8 bits per byte chunk.
        return binary_string[:-padding_length] if padding_length > 0 else binary_string
    
    def lzw_to_bitstream(self, lzw_compressed_data):
        "Converts LZW Output into bytes form for accurate measurement of LZW output byte size."
        max_code = max(lzw_compressed_data)  # Get the highest dictionary code used
        bits_per_code = max(9, math.ceil(math.log2(max_code + 1)))  # Adjust bit size dynamically (min 9 bits) 256, 512, 1024, 2048, 4096 where 2^n where n is bits required to store the max code.

        bit_data = bitarray.bitarray()
        encoded_bits = ''.join(format(code, f'0{bits_per_code}b') for code in lzw_compressed_data)  # Pack each number

        #print(f"Bits Per LZW Code: {bits_per_code}")
        bit_data.extend(encoded_bits)  # Store as bitstream
        return bit_data.tobytes()
    
    def compress(self, text:str):
        # LZW Compress the string of text
        lzw_compressed_data = self.lzw_encode(text)
        
        # Build Huffman Frequency Table
        freq_table = self.build_frequency_table(lzw_compressed_data)
        
        # Build Huffman Tree
        huffman_root_node = self.build_huffman_tree(freq_table)
        huffman_codebook, huffman_reverse_codebook = self.build_huffman_codes(huffman_root_node)

        # Huffman Encode the LZW Output
        huffman_compressed_data = self.huffman_encode(lzw_compressed_data, huffman_codebook)
        
        # Binary Encode the Huffman Output
        binary_compressed_data = self.huffman_to_bytes(huffman_compressed_data)
        
        return binary_compressed_data, huffman_reverse_codebook
    
    def decompress(self, binary_compressed_data, huffman_reverse_codebook = None):
        
        # Convert Binary Input into Huffman Output
        huffman_compressed_data = self.bytes_to_huffman(binary_compressed_data)
        
        # Convert Huffman Input into LZW Output.
        if huffman_reverse_codebook:
            lzw_compressed_data = self.huffman_decode(huffman_compressed_data, huffman_reverse_codebook) # If stored huffman is available, different encoding and decoding session.
        
        # Decode LZW Input into String
        decompressed_data = self.lzw_decode(lzw_compressed_data)
        
        # Construct List, using whitespace as separator.
        decompressed_data = decompressed_data.split()
        
        return decompressed_data

def int_to_bytes(x: int, num_bytes) -> bytes:
    return x.to_bytes(num_bytes, "big")
    
def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, "big")

def dte_encode(uncompressed_wordlist:list[str], file_name:str):    
    # Set file paths for compressed and reverse_codebook
    compressed_file_path = os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', f'{file_name}', f'{file_name}_compressed_wordlist.txt')
    reverse_codebook_path = os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', f'{file_name}', f'{file_name}_reverse_codebook.json')
    
    # LZW -> Huffman Compression and Decompression
    # Check if a compressed text file already exists. If you want a fresh compression session, ensure both huffman metadata and Compressed text file are deleted.
    if not os.path.exists(compressed_file_path):
        # Construct a new compressed text file.
        with open(compressed_file_path, "w") as file:
            joined_wordlist = " ".join(uncompressed_wordlist)
            compressor_instance = Compressor()
            compressed_wordlist, huffman_reverse_codebook = compressor_instance.compress(joined_wordlist)
            ascii_compressed_wordlist = base64.b64encode(compressed_wordlist).decode("utf-8") # Convert to ASCII String
            file.write(ascii_compressed_wordlist)
            
            # Store Huffman metadata
            with open(reverse_codebook_path, "w") as f:
                json.dump(huffman_reverse_codebook, f)
          
    # Decompress text file
    with open(compressed_file_path, "r") as file:
        compressor_instance = Compressor()
        
        ascii_compressed_wordlist = file.read()
        # Read Huffman metadata
        with open(reverse_codebook_path, "r") as f:
            reverse_codebook = json.load(f)
        
        compressed_wordlist = base64.b64decode(ascii_compressed_wordlist.encode()) # Convert to bytes object

        decompressed_wordlist = compressor_instance.decompress(compressed_wordlist, reverse_codebook) 
    
    random_word = random.choice(decompressed_wordlist)
    index = decompressed_wordlist.index(random_word)
    
    chunk_size_bytes = 2 # Must be the same size in DTE encode and decode. Note that 2^chunk_size_bytes must be able to accommodate the largest seed integer in the DTE.
    byte_value = int_to_bytes(index, chunk_size_bytes) 
    plaintext_seed = byte_value

    # # print(byte_value) # Show Bytes in hexadecimal form (\x) of Test Seed Word Indices Individually.
    # # print(b"".join(plaintext)) # Show Bytes in hexadecimal form (\x) of Test Seed Word Indices Combined.

    return plaintext_seed

def dte_decode(seed_bytes: bytes) -> dict:
    wordlists = [
        "firstnames", "middlenames", "lastnames", "birthdate", "maritalstatus", "nationality",
        "occupation", "passportNo", "philid", "race", "religion", "sex", "sssNo", "suffixes"
    ]

    chunk_size_bytes = 2
    decoded_result = {}

    for i, wordlist_name in enumerate(wordlists):
        # Load corresponding compressed wordlist and reverse codebook
        compressed_file_path = os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', wordlist_name, f'{wordlist_name}_compressed_wordlist.txt')
        reverse_codebook_path = os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', wordlist_name, f'{wordlist_name}_reverse_codebook.json')

        # Decompress wordlist
        with open(compressed_file_path, "r") as file:
            ascii_compressed_wordlist = file.read()
        with open(reverse_codebook_path, "r") as f:
            reverse_codebook = json.load(f)

        compressor_instance = Compressor()
        compressed_wordlist = base64.b64decode(ascii_compressed_wordlist.encode())
        decompressed_wordlist = compressor_instance.decompress(compressed_wordlist, reverse_codebook)

        # Get 2-byte chunk for this wordlist
        byte_chunk = seed_bytes[i*chunk_size_bytes:(i+1)*chunk_size_bytes]
        index = int_from_bytes(byte_chunk) % len(decompressed_wordlist)
        decoded_result[wordlist_name] = decompressed_wordlist[index]

    return decoded_result

def encrypt(dte:bytes, honeypasswords:list[str], sugarword_index:int):
    bit_input = 32  # Bit length of RBMRSA. Adjust this as per bit length conventions. (256,512,1024,2048 etc.). Affects ciphertext, d length etc. See ciphertext. file
    bits = try_generating_keys.compute_bit(bit_input) # We just floor divide the bits by 4 - among the four prime numbers
    
    p, q, r, s = try_generating_keys.generating_keys(bits) # We produce 4 random-bit prime numbers with the divided bit length
    N, PHI, e = try_generating_keys.computation_keys(p, q, r, s)
    ### Testing only, Consistent Values for consistent output ###
    #p, q, r, s = 179, 139, 227, 137
    #N, PHI, e = 773774219, 754999104, 53131
    ### Test End ###
    y, x = try_eea_mod.gcd_checker(e, PHI)
    d = try_eea_mod.generating_d(x, y, e, PHI) # We compute for the private key.
    
        
    fake_passwords = honeypasswords.copy() # Make another copy, not a reference.
    fake_passwords.pop(sugarword_index) # Remove sugarword from list of honeypasswords.
    
    honey_keys: list[dict[str, int]] = [{} for _ in range(len(honeypasswords))]  # Ensures the list is pre-filled with empty dictionaries of length honeypasswords.
    # Attach a fake private key to every honey password.
    j=0
    for i in range (len(honeypasswords)):
        if i != sugarword_index:
            honey_keys[i] = {fake_passwords[j]: derive_fake_private_key(fake_passwords[j], d.bit_length(), PHI)}
            j = j + 1
            continue           
        honey_keys[i] = {honeypasswords[sugarword_index]: d} # Insert sugarword with actual d private key inside honey_keys list of dictionaries.
    
    for hp_pkey_pair in honey_keys:
        hp, pkey = next(iter(hp_pkey_pair.items()))

    
    dte_bytes:list[int] = list(dte)  # Convert the byte sequence into a list of int.
    encrypted_bytes:list[int] = [pow(byte, e, N) for byte in dte_bytes]  # Encrypt each element of the list and store it in another list.
    
    # Bitstuffing 
    binary_list = try_binary_conversion.decimal_to_binary(encrypted_bytes) # Convert integers in the list into binary for bitstuffing process.

    ################## Debugging ##################
    save_binary_list_initial = binary_list.copy()
    # print("Before Bitstuffing: ", binary_list[:20])
    ################## Debugging ##################

    bitX = try_bitstuffing.bitstuffX(binary_list)
    bitY = try_bitstuffing.bitstuffY(bitX)
    bitZ = try_bitstuffing.bitstuffZ(bitY)
    
    # Convert back each stuffed binary bits element in the list, into list of int.
    binary_list:list[int] = [try_binary_conversion.binary_to_decimal(element) for element in bitZ]
    
    ################## Debugging ##################
    desZ = try_destuffing.destuffZ(bitZ)
    desY = try_destuffing.destuffY(desZ)
    desX = try_destuffing.destuffX(desY)
    # print("Is desX == Initial Binary List before BitStuffing?: ", desX == save_binary_list_initial)
    # print("desX: ",desX[:20])
    ################## Debugging ##################
    
    # Convert all int elements in the list back into a single byte sequence.
    max_bits = max(c.bit_length() for c in binary_list)  # Get largest bit size in `binary_list`
    byte_list:list[bytes] = [c.to_bytes((max_bits + 7) // 8, "big") for c in binary_list]  # Ensures all numbers fit into a fixed byte size
    
    ciphertext: bytes = b''.join(byte_list)
    
    rmbrsa_parameters:dict = {"N": N, "e": e, "d": d, "p": p, "q": q, "r": r, "s": s, "PHI": PHI, "honey_keys": honey_keys, "chunk_size": max_bits}
    
    return ciphertext, rmbrsa_parameters

# Decrypt Message using RMBRSA - Debugged and Verified
def decrypt(ciphertext: bytes, rbmrsa_parameters: dict, password_hash:str):
    
    N:int = rbmrsa_parameters["N"]
    p:int = rbmrsa_parameters["p"]
    q:int = rbmrsa_parameters["q"]
    r:int = rbmrsa_parameters["r"]
    s:int = rbmrsa_parameters["s"]
    honey_keys: dict[str, int] = {k: v for element in rbmrsa_parameters["honey_keys"] for k, v in element.items()} 
    # For reference: [{"pass123":fake d},{"pass456":fake d}] -> {"pass123":fake d, "pass456":fake d}
    
    # Retrieve honey_key using input password.
    d:int = honey_keys[password_hash]
    
    # Compute Modular Inverses for CRT Optimization
    pInv, qInv, rInv, sInv = try_decryption_crt.modInv_Computation(N, p, q, r, s)
    dp, dq, dr, ds = try_decryption_crt.crt_equations(p, q, r, s, N, d)
    
    # We convert the single byte sequence into a list of bytes.
    chunk_size = rbmrsa_parameters['chunk_size'] 

    byte_list = []
    i = 0
    while i < len(ciphertext):
        encrypted_int = int.from_bytes(ciphertext[i:i+((chunk_size + 7) // 8)], "big")
        byte_list.append(encrypted_int)
        i += ((chunk_size + 7) // 8) # Use stored chunk size and do the same formula.
        
    binary_list = try_binary_conversion.decimal_to_binary(byte_list)

    # Bit destuffing 
    # print("Decrypted Integers Binary List: ", binary_list[:20])
    desZ = try_destuffing.destuffZ(binary_list)
    desY = try_destuffing.destuffY(desZ)
    desX = try_destuffing.destuffX(desY)
    # print("After Destuffing: ", desX[:20])
    
    # Convert back each stuffed binary bits element in the list into list of int.
    binary_list:list[int] = [try_binary_conversion.binary_to_decimal(element) for element in desX]

    # Decrypt each integer in the list.
    decrypted_integers:list[int] = try_decryption_crt.four_parts(
        binary_list, p, q, r, s, N, pInv, qInv, rInv, sInv, dp, dq, dr, ds
    )

    # Converts the list of integers back to a single byte sequence.
    dte = bytes(c % 256 for c in decrypted_integers)  # Ensure values fit in the valid byte range of 0-255, because a d_fake may produce result in decrypted_integers that are outside this range.
    return dte

def derive_key(password:str, salt=None):
    password_bytes = password.encode("utf-8") # String to bytes
    if salt is None:
        salt = Random.new().read(16)
        
    argon2id_hash = argon2.low_level.hash_secret_raw(
        password_bytes,  # String to bytes
        salt, 
        time_cost=2, 
        memory_cost=102400, 
        parallelism=8, 
        hash_len=64, 
        type=argon2.low_level.Type.ID
    )
    return argon2id_hash, salt

def derive_fake_private_key(password: str, d_bit_length: int, PHI: int):
    """Derives a fake private key with the same bit length as the real private key `d`."""
    hashed = hashlib.sha256(password.encode()).hexdigest()  # Hash password to hex string
    fake_key_int = int(hashed, 16)  # Convert hexa hash to integer
    
    # Keeps d_fake within valid range of modular arithmetic, not negative or too large.
    d_fake = fake_key_int % PHI  
    
    # Force `d_fake` to have exactly the same bit length as `d`
    bit_mask = 1 << (d_bit_length - 1) # 100...000 (2048 bits long)
    d_fake |= bit_mask  # Sets the highest bit of d_fake to 1 so it is the same length
    return d_fake