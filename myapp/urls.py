from django.urls import path, re_path
from django.urls import include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('lstm', views.lstm, name='lstm'),
    path('v2t', views.v2t, name='v2t'),
    path('l2t', views.l2t, name='l2t'),
      path('video_feed', views.video_feed, name='video_feed'),
    re_path(r'^.*\.*', views.pages, name='pages'),
]
