#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:17:31 2023

@author: tangshuo
"""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
from datetime import datetime, timedelta
from selenium.webdriver.support.select import Select
from itertools import cycle
import imaplib
import email
import pickle
import random
from sqlalchemy import create_engine
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#%%
def get_mail_verified(username, password):

    def get_server_address(username):
        server = re.search(r'@(hotmail\.com|outlook\.com)$', username)
        if server:
            server_address = "imap-mail.outlook.com"
        else:
            server = re.search(r'(@gmail\.com)$', username)
            if server:
                server_address = "imap.gmail.com"
            else:
                server = re.search(r'(@yahoo\.com)$', username)
                if server:
                    server_address = "imap.mail.yahoo.com"
                else:
                    server = re.search(r'(@autorambler.ru)$', username)
                    if server:
                        server_address = "imap.rambler.ru"
        return server_address
            
    server_address = get_server_address(username)   
    mail = imaplib.IMAP4_SSL(server_address)
    
    try:
        mail.login(username, password)
    
        # inbox代表收件箱
        mail.select("inbox")
        
        # 搜索邮件
        result, data = mail.uid('search', None, "ALL")
        
        # 获取邮件列表
        email_list = data[0].split()
        
        if email_list:
            # 获取最新的邮件
            latest = email_list[-1]
            # fetch the email body (RFC822) for the given ID
            result, email_data = mail.uid('fetch', latest, '(BODY.PEEK[TEXT])')
            raw_email = email_data[0][1].decode("utf-8")
            # continue parsing the raw email
            email_message = email.message_from_string(raw_email)
        
            # 输出邮件内容
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    html_content = part.get_payload(decode=True)
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                    clean_text = re.sub(r'\s+', ' ', text)
                    verified_code = re.search(r'single-use code. (\w+)', clean_text)
                    if verified_code:
                        code = verified_code.group(1)
                    else:
                        code = None
    except:
        code = None
    return code
#%%
#%%
def twitter_login(email, username, password, email_password, proxy_username, proxy_password, proxy_ip, proxy_port):
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--headless=new")
    # If ip proxy is used
    '''
    proxy_dict = {
    'proxy':{
        'http': f'http://{proxy_username}:{proxy_password}@{proxy_ip}:{proxy_port}',
        'https': f'http://{proxy_username}:{proxy_password}@{proxy_ip}:{proxy_port}'
    }
    }
    '''
    webdriver_service = Service('./chromedriver-linux64/chromedriver')
    webdriver_service.log_path = './chromedriver.log'  # 设置日志路径
    driver = webdriver.Chrome(service = webdriver_service, options=chrome_options)
    #webdriver_service = Service(ChromeDriverManager().install())
    #driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    driver.get('https://twitter.com/login')
    print(driver.title)
    time.sleep(7.5)

    # Error
    xpath_expr_1 = (
    '//span[text()="出错了，但别担心，这不是你的错"] | '
    '//span[text()="Something went wrong, but don’t fret — let’s give it another shot."] | '
    '//span[text()="出错了，请重试"] | '
    '//span[text()="Something went wrong. Try reloading."]'
    )
    try:
        login_error = driver.find_element(By.XPATH, xpath_expr_1)
        if login_error:
            login_error.click()
            time.sleep(3)
    except NoSuchElementException:
        pass
    time.sleep(5)

    # Login email
    xpath_expr_2 = (
    '//span[text()="下一步"] | '
    '//span[text()="Next"]'
    )
    try:
        username_field = driver.find_element(By.NAME, "text")
        username_field.send_keys(email)
        next_button = driver.find_element(By.XPATH, xpath_expr_2)
        next_button.click()
    except NoSuchElementException:
        time.sleep(10)
        try:
            username_field = driver.find_element(By.NAME, "text")
            username_field.send_keys(email)
            next_button = driver.find_element(By.XPATH, xpath_expr_2)
            next_button.click()
        except NoSuchElementException:
            time.sleep(15)
            username_field = driver.find_element(By.NAME, "text")
            username_field.send_keys(email)
            next_button = driver.find_element(By.XPATH, xpath_expr_2)
            next_button.click()
    print("Enter Email")
    time.sleep(5)

    # Login username
    xpath_expr_3 = (
    '//span[text()="输入你的手机号码或用户名"] |'
    '//span[text()="Enter your phone number or username"]'
    )
    retry = 0
    max_retry = 3
    error_message = None
    while retry <= max_retry:
        try:
            error_message = driver.find_element(By.XPATH, xpath_expr_3)
            break
        except:
            time.sleep(5)
            retry += 1
    if error_message:
        new_username_field = driver.find_element(By.NAME, 'text')
        new_username_field.send_keys(username)
        next_button = driver.find_element(By.XPATH, xpath_expr_2)
        next_button.click()
        print("Enter username")
    time.sleep(5)

    # Login password
    xpath_expr_4 = (
    '//span[text()="登录"] |'
    '//span[text()="Log in"]')
    retry = 0
    max_retry = 3
    while retry <= max_retry:
        try:
            password_field = driver.find_element(By.NAME, "password")
            password_field.send_keys(password)
            next_button = driver.find_element(By.XPATH, xpath_expr_4)
            next_button.click()
            break
        except NoSuchElementException:
            time.sleep(5)
            retry += 1
    print("Enter password")
    time.sleep(5)

    # If email is inspected
    xpath_expr_5 = (
    '//span[text()="检查你的邮箱"] | '
    '//span[text()="Confirm your email address"]')
    try:
        error_message = driver.find_element(By.XPATH, xpath_expr_5)
        if error_message:
            time.sleep(10)
            new_email_field = driver.find_element(By.NAME, 'text')
            verified_code = get_mail_verified(email, email_password)
            if verified_code is None:
                try:
                    driver.quit()
                except:
                    pass
                driver = None
            else:
                new_email_field.send_keys(verified_code)
                next_button = driver.find_element(By.XPATH, xpath_expr_2)
                next_button.click()
    except NoSuchElementException:
        pass
    except TimeoutException:
        try:
            driver.quit()
        except:
            pass
        driver = None

    # If Extra Authentication is needed, try next account.
    time.sleep(3)
    try:
        heading1 = driver.find_element(By.XPATH, '//h2[contains(text(),"Authenticate your account")]')
        heading2 = driver.find_element(By.XPATH, "//button[contains(text(), 'Authenticate')]")
        if heading1 or heading2:
            try:
                driver.quit()
            except:
                pass
            driver = None
    except:
        pass
    time.sleep(5)
    return driver
#%%
def login_and_check_account(account):
    email_, username, password, email_password = account
    try:
        driver = twitter_login(email_, username, password, email_password, None, None, None, None)
        driver.maximize_window()
        time.sleep(6)
        return driver
    except:
        try:
            driver.quit()
        except:
            pass
        return None
#%%
def login_checking(all_accounts):
    account = next(all_accounts)
    driver = login_and_check_account(account)
    while driver is None:
        account = next(all_accounts)
        driver = login_and_check_account(account)
    return driver
#%%
def convert_to_int(input_string):
    input_string = input_string.replace(',', '')
    if 'K' in input_string:
        number = float(input_string.replace('K', '')) * 1000
        return int(number) 
    elif 'M' in input_string:
        number = float(input_string.replace('M', '')) * 1000000
        return int(number)
    elif input_string == '':
        return 0
    else:
        return int(input_string)

def get_user_following(driver, following_name):
    try:
        following_name = following_name.replace("@", "")
        url = f'https://twitter.com/{following_name}'
        driver.get(url)
        if "These tweets are protected" in driver.page_source:
            return "next"
        if "Something went wrong. Try reloading." in driver.page_source:
            return "next account"
    except:
        return "next account"
    try:
        search_error = driver.find_element(By.CSS_SELECTOR, '.css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0')
        if search_error.text == 'Something went wrong. Try reloading.':
            return "next account"
    except:
        pass
        
    try:
        wait = WebDriverWait(driver, 15)  # 例如，最大等待时间为15秒
        
        # 等待followers元素出现
        followers = wait.until(EC.presence_of_element_located((By.XPATH, f'//a[@href="/{following_name}/verified_followers" and @role= "link"]//span[contains(translate(text(), "0123456789", ""),text())]')))
        num_followers = convert_to_int(followers.text)
        
        # 等待followings元素出现
        followings = wait.until(EC.presence_of_element_located((By.XPATH, f'//a[@href="/{following_name}/following" and @role= "link"]//span[contains(translate(text(), "0123456789", ""),text())]')))
        num_followings = convert_to_int(followings.text)
        return following_name, num_followers, num_followings
    except:
        return "next"
