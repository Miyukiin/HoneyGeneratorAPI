from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from rest_framework.decorators import api_view
import json
import os
from .utils import dte_encode, dte_decode, encrypt, decrypt
from django.conf import settings
import base64

# Create your views here.

@api_view(['POST'])
def generate_dte_seeds(request:HttpRequest):
    """
    Generate DTE seeds for each of the wordlists.

    Args:
        request (HttpRequest): The request object. Must be a POST request.

    Returns:
        JsonResponse: A JSON response with the DTE seeds in a single ASCII-encoded string, or an error message.
    """
    try:     
        wordlists_path = {
            "firstnames": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'firstnames', 'firstnames_wordlist.txt'),
            "middlenames": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'middlenames', 'middlenames_wordlist.txt'),
            "lastnames": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'lastnames', 'lastnames_wordlist.txt'),
            "birthdate": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'birthdate', 'birthdate_wordlist.txt'),
            "maritalstatus": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'maritalstatus', 'maritalstatus_wordlist.txt'),
            "nationality": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'nationality', 'nationality_wordlist.txt'),
            "occupation": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'occupation', 'occupation_wordlist.txt'),
            "passportNo": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'passportNo', 'passportNo_wordlist.txt'),
            "philid": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'philid', 'philid_wordlist.txt'),
            "race": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'race', 'race_wordlist.txt'),
            "religion": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'religion', 'religion_wordlist.txt'),
            "sex": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'sex', 'sex_wordlist.txt'),
            "sssNo": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'sssNo', 'sssNo_wordlist.txt'),
            "suffixes": os.path.join(settings.BASE_DIR, 'DistributiveEncoderAPI', 'static', 'DistributiveEncoderAPI', 'suffixes', 'suffixes_wordlist.txt'),
        }
        
        plaintext_seeds = b""
        for wordlist_name, path in wordlists_path.items():
            with open(path, "r") as wordlist_file:
                # Split by newline or space.
                wordlist = wordlist_file.read().split()
            # Pass wordlist and identifier
            plaintext_seed = dte_encode(wordlist, wordlist_name)
            plaintext_seeds += plaintext_seed
        
        return JsonResponse({"plaintext_seeds": base64.b64encode(plaintext_seeds).decode("utf-8")}, status=200)
    
    except Exception as e:
        # Return error response
        return JsonResponse({"error": f"Unable to generate DTE Seeds: {str(e)}"}, status=500)

@api_view(['GET'])
def decode_dte_seeds(request:HttpRequest):
    try:
        dte_seeds:str = request.data.get('dte_seeds')
        field_message_dict :dict = dte_decode(base64.b64decode(dte_seeds))
        
        print(dte_seeds)
        print(field_message_dict)
        
        return JsonResponse({"field_message_dict": field_message_dict}, status=200)
    except Exception as e:
        # Return error response
        return JsonResponse({"error": f"Unable to decode DTE Seeds: {str(e)}"}, status=500)

@api_view(['POST'])   
def decrypt_dte_seeds(request:HttpRequest):
    try:
        rbmrsa_parameters = request.data.get('rbmrsa_parameters')
        password_hash:str = request.data.get('password_hash')
        encrypted_seed:str = request.data.get('encrypted_seed')
        dte_seeds = decrypt(base64.b64decode(encrypted_seed), rbmrsa_parameters, password_hash)
        
        return JsonResponse({"dte_seeds": base64.b64encode(dte_seeds).decode("utf-8")}, status=200)
        
    except Exception as e:
        # Return error response
        return JsonResponse({"error": f"Unable to decrypt DTE Seeds: {str(e)}"}, status=500)  
    
@api_view(['POST'])   
def encrypt_dte_seeds(request:HttpRequest):
    try:
        user_index = request.data.get('user_index')
        sugarword_index:int = request.data.get('sugarword_index')
        honey_hashes:list[str] = request.data.get('honey_hashes')
        ascii_seed:str = request.data.get('seed')
        ciphertext, rmbrsa_parameters = encrypt(base64.b64decode(ascii_seed),honey_hashes,sugarword_index)
        
        return JsonResponse({"ciphertext": base64.b64encode(ciphertext).decode("utf-8"), "rbmrsa_parameters": rmbrsa_parameters}, status=200)
        
    except Exception as e:
        # Return error response
        return JsonResponse({"error": f"Unable to encrypt DTE Seeds: {str(e)}"}, status=500)
        