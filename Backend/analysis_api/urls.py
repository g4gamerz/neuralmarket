from django.urls import path
from . import views 

urlpatterns = [
    path('analyze/', views.analyze_stock_view, name='analyze_stock'),
    path('auth/register/', views.register_view, name='register'),
    path('auth/login/', views.login_view, name='login'),
    path('auth/logout/', views.logout_view, name='logout'),
    path('auth/forgot-password/', views.forgot_password_view, name='request_password_reset'), 
    path('auth/reset-password-confirm/', views.reset_password_confirm_view, name='reset_password_confirm'), 
    path('stocks/', views.get_stock_list_view, name='get_stock_list'),

]
