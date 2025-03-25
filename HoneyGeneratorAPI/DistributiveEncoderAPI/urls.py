"""
URL configuration for HoneyGeneratorAPI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from .views import generate_dte_seeds, encrypt_dte_seeds, decrypt_dte_seeds, decode_dte_seeds

urlpatterns = [
    path("generate_seeds/", generate_dte_seeds, name="generate_dte_seeds"),
    path("encrypt_dte_seeds/", encrypt_dte_seeds, name="encrypt_dte_seeds"),
    path("decrypt_dte_seeds/", decrypt_dte_seeds, name="decrypt_dte_seeds"),
    path("decode_dte_seeds/", decode_dte_seeds, name="decode_dte_seeds"),
]
