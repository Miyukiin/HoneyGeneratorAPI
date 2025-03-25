from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from .utils import MLHoneywordGenerator
from rest_framework.decorators import api_view
import argon2
from Crypto import Random
import base64
import json
import logging
logger = logging.getLogger(__name__)

# Create your views here.
@api_view(['GET'])
def generate_honeypasswords(request:HttpRequest):
    """
    Generates Honey Passwords based on the method specified.

    Args:
        request (HttpRequest): The request object. Must be a GET request containing the base password as a query.

    Returns:
        JsonResponse: A JSON response with the generated honeywords and sugarword index, or an error message.
    """
    if request.method == 'GET':
        try:
            # Retrieve the password parameter
            base_password = request.GET.get("password")
            if not base_password:
                return JsonResponse({"error": "The 'password' parameter is required."}, status=400)

            # Generate honeywords
            honey_password_generator = MLHoneywordGenerator()
            honeyword_list, sugarword_index = honey_password_generator.generate_honeywords(base_password)

            # Return success response
            return JsonResponse({
                "honeyword_list": honeyword_list,
                "sugarword_index": sugarword_index
            }, status=200)

        except Exception as e:
            # Return error response
            return JsonResponse({"error": f"Unable to generate honeypasswords: {str(e)}"}, status=500)
    else:
        # Handle unsupported HTTP methods
        return JsonResponse({"error": "Only GET requests are allowed."}, status=405)
    
@api_view(['POST'])  
def hash_honeypasswords(request:HttpRequest):
    """
    Hashes Honeypasswords using Argon2 and Generated Salt.
    Args:
        request (HttpRequest): The request object. Must be a POST request containing the honeypasswords as data.

    Returns:
        JsonResponse: A JSON response with the hashed honeywords and sugarword index, or an error message.
    """
    if request.method == 'POST':
        try:
            honey_passwords:list[str] = request.data.get('honeyword_list')
            honey_hashes = []
        
            encoded_salt:str = request.data.get("salt", None)

            if not isinstance(honey_passwords, list):
                return JsonResponse({"error": "honeyword_list must be a list of strings."}, status=400)
            
            if encoded_salt is None:
                salt = Random.new().read(16)
            else:
                salt = base64.b64decode(encoded_salt)
                    
            for honey_password in honey_passwords:
                password_bytes = honey_password.encode("utf-8") # String to bytes
                
                argon2id_hash = argon2.low_level.hash_secret_raw(
                    password_bytes,  # String to bytes
                    salt, 
                    time_cost=2, 
                    memory_cost=102400, 
                    parallelism=8, 
                    hash_len=64, 
                    type=argon2.low_level.Type.ID
                )
                honeypassword_hash = base64.b64encode(argon2id_hash).decode('utf-8')
                honey_hashes.append(honeypassword_hash)
                
            # Return success response
            return JsonResponse({
                "honeyword_hashes": honey_hashes,
                "salt": base64.b64encode(salt).decode('utf-8')
            }, status=200)

        except Exception as e:
            # Return error response
            return JsonResponse({"error": f"Unable to hash honeypasswords: {str(e)}"}, status=500)
    else:
        # Handle unsupported HTTP methods
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)