from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from .utils import MLHoneywordGenerator, ExistingPasswordGeneration
from rest_framework.decorators import api_view

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