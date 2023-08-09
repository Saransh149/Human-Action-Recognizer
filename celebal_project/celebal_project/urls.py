# http://localhost:8000/api/predict/
from django.urls import path
from har_app.views import predict

urlpatterns = [
    path('api/predict/', predict, name='predict'),
]
