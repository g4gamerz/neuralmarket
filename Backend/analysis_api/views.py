# analysis_api/views.py

from django.http import JsonResponse, HttpRequest
from django.contrib.auth import authenticate, login as django_login, logout as django_logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.db import IntegrityError
import json
import requests 
import os
import logging
from .pipeline import STOCK_MAPPING_FILE
# For password reset
from django.contrib.auth.tokens import default_token_generator 
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.core.mail import send_mail
from django.urls import reverse 
from django.conf import settings 

from .pipeline import run_news_sentiment_pipeline, PSX_SYMBOL_TO_NAME, PSX_NAME_TO_SYMBOL 

logger = logging.getLogger(__name__)

EODHD_API_KEY = "6839ee1a3b7e51.24083684" 
EODHD_EXCHANGE_CODE = "KAR" 

def fetch_stock_price_eodhd(stock_symbol):
    if not EODHD_API_KEY or EODHD_API_KEY == "YOUR_EODHD_API_KEY":
        logger.warning("EODHD_API_KEY is not set. Cannot fetch stock price.")
        return None, "EODHD API key not configured."
    if not stock_symbol:
        return None, "Stock symbol not provided for price fetching."
    url = f"https://eodhd.com/api/real-time/{stock_symbol}.{EODHD_EXCHANGE_CODE}?api_token={EODHD_API_KEY}&fmt=json"
    logger.info(f"Fetching real-time price for {stock_symbol}.{EODHD_EXCHANGE_CODE} from EODHD: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        logger.info(f"EODHD Real-time Response for {stock_symbol}: {data}")
        price = data.get('close') 
        if price is not None: return float(price), None 
        if data.get('last') is not None: return float(data.get('last')), None
        if data.get('currentPrice') is not None: return float(data.get('currentPrice')), None
        logger.warning(f"Could not find price in EODHD real-time response for {stock_symbol}: {data}")
        return None, "Price data ('close', 'last', or 'currentPrice') not found in EODHD real-time response."
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching price for {stock_symbol}: {http_err} - Response: {response.text if response else 'No response'}")
        return None, f"Could not fetch price (HTTP {response.status_code if response else 'Unknown'})."
    except Exception as e:
        logger.error(f"Error fetching price for {stock_symbol}: {e}", exc_info=True)
        return None, "An unexpected error occurred while fetching price."

def get_stock_list_view(request: HttpRequest):
    """
    An endpoint to serve the full list of stocks from the names.json file
    for the frontend autocomplete feature.
    """
    if request.method == 'GET':
        try:
            with open(STOCK_MAPPING_FILE, 'r', encoding='utf-8') as f:
                stock_data = json.load(f)
            return JsonResponse(stock_data, safe=False) # safe=False is needed for sending a list
        except FileNotFoundError:
            logger.error(f"Stock mapping file not found at: {STOCK_MAPPING_FILE}")
            return JsonResponse({'error': 'Stock data source not found on server.'}, status=500)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from stock mapping file: {STOCK_MAPPING_FILE}")
            return JsonResponse({'error': 'Could not read stock data from source.'}, status=500)
        except Exception as e:
            logger.error(f"An unexpected error occurred in get_stock_list_view: {e}", exc_info=True)
            return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method. Use GET.'}, status=405)

def analyze_stock_view(request: HttpRequest):
    if request.method == 'GET':
        user_query = request.GET.get('stock_query', None) 
        if not user_query:
            return JsonResponse({'error': 'stock_query (symbol) parameter is required.'}, status=400)
        logger.info(f"Received analysis request for query: '{user_query}'")
        stock_symbol_for_api = None; company_name_for_display = user_query 
        normalized_user_query = user_query.strip().lower()
        if normalized_user_query.upper() in PSX_SYMBOL_TO_NAME:
            stock_symbol_for_api = normalized_user_query.upper()
            company_name_for_display = PSX_SYMBOL_TO_NAME[stock_symbol_for_api]
        elif normalized_user_query in PSX_NAME_TO_SYMBOL:
            stock_symbol_for_api = PSX_NAME_TO_SYMBOL[normalized_user_query]
            company_name_for_display = PSX_SYMBOL_TO_NAME[stock_symbol_for_api] 
        if not stock_symbol_for_api and company_name_for_display == user_query:
            logger.warning(f"Could not reliably identify a stock symbol from query '{user_query}' for price fetching.")
            stock_symbol_for_price_check = user_query.strip().upper()
        else: stock_symbol_for_price_check = stock_symbol_for_api
        news_sentiment_data = run_news_sentiment_pipeline(user_stock_query=user_query, desired_final_results=10)
        current_price = "N/A"; price_error = "Could not determine stock symbol for price."
        if stock_symbol_for_price_check:
            logger.info(f"Identified symbol for EODHD: {stock_symbol_for_price_check}")
            price_data, price_err_msg = fetch_stock_price_eodhd(stock_symbol_for_price_check)
            if price_data is not None: current_price = f"{price_data:.2f}"; price_error = None
            else: price_error = price_err_msg 
        else: logger.warning(f"No symbol identified for EODHD price check based on query '{user_query}'.")
        response_data = {'stock_query': user_query, 'identified_company_name': company_name_for_display, 'identified_symbol': stock_symbol_for_api if stock_symbol_for_api else "N/A", 'current_price': current_price, 'price_error': price_error, 'news_sentiment_data': news_sentiment_data, 'overall_sentiment_label': "N/A", 'overall_sentiment_score': 0.0}
        if news_sentiment_data:
            all_scores = [item['sentiment_score'] for item in news_sentiment_data if item.get('sentiment_label') != 'N/A' and isinstance(item.get('sentiment_score'), (int, float))]
            if all_scores:
                response_data['overall_sentiment_score'] = sum(all_scores) / len(all_scores)
                if response_data['overall_sentiment_score'] > 0.15: response_data['overall_sentiment_label'] = 'Positive'
                elif response_data['overall_sentiment_score'] < -0.15: response_data['overall_sentiment_label'] = 'Negative'
                else: response_data['overall_sentiment_label'] = 'Neutral'
            else: response_data['overall_sentiment_label'] = 'Neutral' 
        else: response_data['overall_sentiment_label'] = 'N/A' 
        return JsonResponse(response_data)
    else: return JsonResponse({'error': 'Invalid request method. Use GET.'}, status=405)

@csrf_exempt
def register_view(request: HttpRequest):
    if request.method == 'POST':
        try:
            data = json.loads(request.body); username = data.get('email'); email = data.get('email')
            password = data.get('password'); full_name = data.get('fullName', '')
            if not all([username, email, password]): return JsonResponse({'error': 'Email and password are required.'}, status=400)
            if User.objects.filter(username=username).exists(): return JsonResponse({'error': 'User with this email already exists.'}, status=400)
            user = User.objects.create_user(username=username, email=email, password=password)
            if full_name:
                first_name, last_name = (full_name.split(' ', 1) + [''])[:2]
                user.first_name = first_name; user.last_name = last_name; user.save()
            logger.info(f"User '{username}' registered successfully.")
            return JsonResponse({'success': 'User registered successfully. Please login.'}, status=201)
        except json.JSONDecodeError: return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        except IntegrityError: return JsonResponse({'error': 'User with this email already exists (db error).'}, status=400)
        except Exception as e: logger.error(f"Error during registration: {e}", exc_info=True); return JsonResponse({'error': 'An error occurred.'}, status=500)
    else: return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

@csrf_exempt 
def login_view(request: HttpRequest):
    if request.method == 'POST':
        try:
            data = json.loads(request.body); username = data.get('email'); password = data.get('password')
            if not username or not password: return JsonResponse({'error': 'Email and password are required.'}, status=400)
            user = authenticate(request, username=username, password=password)
            if user is not None:
                django_login(request, user); logger.info(f"User '{username}' logged in.")
                return JsonResponse({'success': 'Login successful.', 'user': {'email': user.email, 'fullName': user.get_full_name() or user.username}})
            else: logger.warning(f"Login failed for '{username}'."); return JsonResponse({'error': 'Invalid credentials.'}, status=401)
        except json.JSONDecodeError: return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        except Exception as e: logger.error(f"Error during login: {e}", exc_info=True); return JsonResponse({'error': 'An error occurred.'}, status=500)
    else: return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

@csrf_exempt 
def logout_view(request: HttpRequest):
    if request.method == 'POST':
        if request.user.is_authenticated:
            logger.info(f"User '{request.user.username}' logging out."); django_logout(request)
            return JsonResponse({'success': 'Logout successful.'})
        else: return JsonResponse({'error': 'User not authenticated.'}, status=401)
    else: return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

@csrf_exempt
def forgot_password_view(request: HttpRequest):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            if not email:
                return JsonResponse({'error': 'Email is required.'}, status=400)

            logger.info(f"Password reset requested for email: {email}")
            associated_users = User.objects.filter(email__iexact=email) 

            if associated_users.exists():
                for user in associated_users: 
                    uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
                    token = default_token_generator.make_token(user)
                    
                  
                    reset_url = f"http://localhost:8001/reset_password_confirm.html?uidb64={uidb64}&token={token}"
                    
                    subject = "Password Reset Requested for NeuralMarket"
                    message_body = (
                        f"Hello {user.username},\n\n"
                        f"You requested a password reset for your NeuralMarket account.\n"
                        f"Please click the link below to set a new password:\n\n"
                        f"{reset_url}\n\n"
                        f"If you did not request this, please ignore this email.\n\n"
                        f"Thanks,\nThe NeuralMarket Team"
                    )
                    
                    try:
                        send_mail(
                            subject,
                            message_body,
                            settings.DEFAULT_FROM_EMAIL or 'noreply@neuralmarket.com', 
                            [user.email], 
                            fail_silently=False,
                        )
                        logger.info(f"Password reset email sent to {user.email} (link: {reset_url})")
                    except Exception as e:
                        logger.error(f"Error sending password reset email to {user.email}: {e}", exc_info=True)
                
                return JsonResponse({'success': 'If an account with that email exists, password reset instructions have been sent.'})
            else:
                logger.info(f"No user found with email {email} for password reset request.")
                return JsonResponse({'success': 'If an account with that email exists, password reset instructions have been sent.'})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        except Exception as e:
            logger.error(f"Error during forgot password request: {e}", exc_info=True)
            return JsonResponse({'error': 'An unexpected error occurred.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)

@csrf_exempt
def reset_password_confirm_view(request: HttpRequest):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            uidb64 = data.get('uidb64')
            token = data.get('token')
            new_password = data.get('new_password')

            if not all([uidb64, token, new_password]):
                return JsonResponse({'error': 'All fields (uidb64, token, new_password) are required.'}, status=400)

            try:
                from django.utils.http import urlsafe_base64_decode
                uid = urlsafe_base64_decode(uidb64).decode()
                user = User.objects.get(pk=uid)
            except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                user = None
            
            if user is not None and default_token_generator.check_token(user, token):
                user.set_password(new_password) 
                user.save()
                logger.info(f"Password reset successful for user {user.username}")
                return JsonResponse({'success': 'Password has been reset successfully. You can now login.'})
            else:
                logger.warning(f"Password reset failed. Invalid token or user ID. UIDB64: {uidb64}, Token: {token}")
                return JsonResponse({'error': 'Invalid or expired password reset link.'}, status=400)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        except Exception as e:
            logger.error(f"Error during password reset confirmation: {e}", exc_info=True)
            return JsonResponse({'error': 'An unexpected error occurred during password reset.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)
