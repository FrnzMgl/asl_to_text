from django.urls import path, re_path
from django.urls import include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('lstm', views.lstm, name='lstm'),
    path('knight', views.knight, name='knight'),
    re_path(r'^.*\.*', views.pages, name='pages'),
]
