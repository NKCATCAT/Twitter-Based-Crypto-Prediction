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
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
from itertools import cycle
import imaplib
import email
import pickle
from sqlalchemy import create_engine
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from collections import deque,Counter
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
    driver.get('https://twitter.com/i/flow/login')
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
    '//span[text()="Enter your phone number or username"] |'
    '//span[text()="Phone or username"]'
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
    '//span[text()="Confirm your email address"] |'
    '//span[text()="Check your email"] |'
    '//span[contains(text(),"a confirmation code")]')
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
# convert text to number for views, retweets, replies, likes
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
#%%
# check if the user has followers >= 10000, following > 0
def check_popularity(driver, following_name):
    try:
        following_name = following_name.replace("@", "")
        url = f'https://twitter.com/{following_name}'
        driver.get(url)
        if "These tweets are protected" in driver.page_source:
            return "next"
        if "Something went wrong. Try reloading." in driver.page_source:
            return "next"
    except:
        return "next"
    try:
        search_error = driver.find_element(By.CSS_SELECTOR, '.css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0')
        if search_error.text == 'Something went wrong. Try reloading.':
            return "next"
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
        if num_followers >= 10000 and num_followings > 0:
            return following_name, num_followers, num_followings
        else:
            return None
    except:
        return "next"

#%%
# parse
def parse_page_sources(page_sources):
    following_names = [] 
    for page_source in page_sources:
        page_source = BeautifulSoup(page_source, 'html.parser')
        user_cells = page_source.find_all('div', attrs = {'data-testid':'cellInnerDiv'})
        for cell in user_cells:
            following_name = cell.find('a', attrs = {'role':'link','tabindex':'-1', 'href':True,
                                                     'class':"css-4rbku5 css-18t94o4 css-1dbjc4n r-1loqt21 r-1wbh5a2 r-dnmrzs r-1ny4l3l"})
            if following_name:
                names = following_name.find('span')
                following_names.append(names.text)
    return list(set(following_names))
#%%
def get_starting_point_followings_verified_followers(driver, start_point):
    def collect_page_sources(driver):
        page_sources = []
        previous_page_source = driver.page_source
        page_sources.append(previous_page_source)
        while True:
            actions = ActionChains(driver)
            actions.send_keys(Keys.PAGE_DOWN)
            actions.perform()
            time.sleep(3)
            current_page_source = driver.page_source # 获取当前页面的源码或内容
            # 比较前后两次页面源码或内容
            if current_page_source == previous_page_source:
                # 没有新内容加载，说明滑动到了页面底部，停止滑动
                break
            page_sources.append(current_page_source)
            previous_page_source = current_page_source
        driver.back()
        return page_sources
    try:
        driver.get(f'https://twitter.com/{start_point}/following') 
        time.sleep(7.5) #等待页面加载
        driver.maximize_window()
        if "Something went wrong. Try reloading." in driver.page_source:
            return "next"
        page_sources_following = collect_page_sources(driver)
        driver.get(f'https://twitter.com/{start_point}/verified_followers') 
        page_sources_followers = collect_page_sources(driver)
        following_names = parse_page_sources(page_sources_following)
        followers_names = parse_page_sources(page_sources_followers)
        split_users = followers_names + following_names
        return list(set(split_users))
    except:
        return "next"
#%%
def login_and_check_account(account):
    email_, username, password, email_password = account
    try:
        driver = twitter_login(email_, username, password, email_password, None, None, None, None)
        if driver:
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
def find_common_users(driver,split_map,threshold = 3):
    kols_followings = []
    kols_followers = []
    kols = []
    all_users = [item for sublist in split_map.values() for item in sublist]
    counter = Counter(all_users)
    filtered_users = [key.replace("@","") for key,value in counter.items() if value >= threshold]
    for user in filtered_users:
        popular = check_popularity(driver, user)
        if popular and popular != "next":
            following_name, num_followers, num_followings = popular
            kols.append(following_name)
            kols_followings.append(num_followings)
            kols_followers.append(num_followers)
    data = {
    'username': kols,
    'followers': kols_followers,
    'following': kols_followings
    }
    return kols,pd.DataFrame(data)
#%%
def process_account(driver, accounts, start_point):
    visited = set()
    queue = deque([(start_point, 0)])
    split_map = {}
    max_depth = 20
    filtered_users = None
    i = 0
    while queue:
        following_name, depth = queue.popleft()
        if following_name in visited or depth > max_depth:
            continue
        visited.add(following_name)
        i +=1

        if i % 300 == 0:
            driver.delete_all_cookies()
            time.sleep(5)
            driver.quit()
            account = next(accounts)
            driver = login_and_check_account(account)
            while driver is None:
                account = next(accounts)
                driver = login_and_check_account(account)
        popular = check_popularity(driver, following_name)
        if popular == "next":
            account = next(accounts)
            driver = login_and_check_account(account)
            while driver is None:
                account = next(accounts)
                driver = login_and_check_account(account)
            popular = check_popularity(driver, following_name)
        elif popular != 'next' and popular is not None:
            popular_name, popular_follower, popular_following = popular
            if "These tweets are protected" not in driver.page_source:
                split_users = get_starting_point_followings_verified_followers(driver, popular_name)
                while split_users == "next":
                    account = next(accounts)
                    driver = login_and_check_account(account)
                    while driver is None:
                        account = next(accounts)
                        driver = login_and_check_account(account)
                    split_users = get_starting_point_followings_verified_followers(driver, popular_name)
                split_map[popular_name] = split_users
                start_data = {"username":popular_name, "followers":popular_follower,"following":popular_following}
                start_df = pd.DataFrame([start_data])  
                for split_user in split_users:
                    if split_user not in visited:
                        queue.append((split_user, depth + 1))
                        filtered_users,dataframe = find_common_users(driver,split_map,threshold = 3) 
                        if split_user == split_users[-1]:
                            # Using concat to append new data
                            dataframe = pd.concat([dataframe,start_df], ignore_index=True)
                        if len(dataframe) > 0:
                            save_todatabase(dataframe,sql_connector, database_tosave)
    driver.quit()
    if filtered_users:
        return filtered_users
def save_todatabase(dataframe,sql_connector, database_tosave):
    engine = create_engine(sql_connector)
    try:
        existing_usernames = pd.read_sql(f"SELECT username FROM {database_tosave}", engine)
        dataframe = dataframe[~dataframe['username'].isin(existing_usernames['username'])]
    except:  
        pass
    dataframe.to_sql(name = database_tosave, con = engine, index = False, if_exists = 'append')
    engine.dispose()
#%% 
start_point = "HsakaTrades"
account_filepath = r"./all_accounts"
sql_connector = "your_database_connector"
database_tosave = "expanding_kols"
with open(account_filepath, 'rb') as f:
    all_accounts = pickle.load(f)
all_accounts = list(all_accounts)[:25]
def processing(all_accounts,start_point,sql_connector, database_tosave):
    accounts = cycle(all_accounts)
    account = next(accounts)
    driver = login_and_check_account(account)
    while driver is None:
        account = next(accounts) 
        driver = login_and_check_account(account) 
    kols = process_account(driver, accounts, start_point)
    start_points = [kol for kol in kols if kol != start_point]
    for continue_start_point in start_points:
        try:
            processing(all_accounts,continue_start_point,sql_connector, database_tosave)
        except:
            try:
                driver.quit()
            except:
                pass
            account = next(accounts)
            driver = login_and_check_account(account)
            while driver is None:
                account = next(accounts)
                driver = login_and_check_account(account)
            processing(all_accounts,continue_start_point,sql_connector, database_tosave)
processing(all_accounts,start_point,sql_connector, database_tosave)
