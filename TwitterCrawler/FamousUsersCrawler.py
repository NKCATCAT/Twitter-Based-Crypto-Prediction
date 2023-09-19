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
from collections import deque
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
    
    # If ip proxy is used
    '''
    proxy_dict = {
    'proxy':{
        'http': f'http://{proxy_username}:{proxy_password}@{proxy_ip}:{proxy_port}',
        'https': f'http://{proxy_username}:{proxy_password}@{proxy_ip}:{proxy_port}'
    }
    }
    '''
    webdriver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    driver.get('https://twitter.com/login')
    
    time.sleep(7.5)
    try:
        login_error = driver.find_element(By.XPATH, '//span[text()="出错了，但别担心，这不是你的错"]')
        if login_error is None:
            login_error = driver.find_element(By.XPATH, '//span[text()="Something went wrong, but don’t fret — let’s give it another shot."]')
            if login_error:
                login_error.click()
                time.sleep(3)
                driver.refresh()
    except NoSuchElementException:
        try:
            login_error = driver.find_element(By.XPATH, '//span[text()="出错了，但别担心，这不是你的错"]')
            if login_error is None:
                login_error = driver.find_element(By.XPATH, '//span[text()="Something went wrong, but don’t fret — let’s give it another shot."]')
                if login_error:
                    login_error.click()
                    time.sleep(3)
                    driver.refresh()
        except:
            pass
    time.sleep(5)
    try:
        login_error = driver.find_element(By.XPATH, '//span[text()="出错了，请重试"]')
        if login_error:
            login_error.click()
            login_error = driver.find_element(By.XPATH, '//span[text()="出错了，请重试"]')
            if login_error:
                login_error.click()
    except NoSuchElementException:
        try:
            login_error = driver.find_element(By.XPATH, '//span[text()="出错了，请重试"]')
            if login_error:
                login_error.click()
                login_error = driver.find_element(By.XPATH, '//span[text()="出错了，请重试"]')
                if login_error:
                    login_error.click()
        except:
            pass
    # Login email
    try:
        username_field = driver.find_element(By.NAME, "text")
        username_field.send_keys(email)
        next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
        next_button.click()
    except NoSuchElementException:
        driver.refresh()
        time.sleep(10)
        try:
            username_field = driver.find_element(By.NAME, "text")
            username_field.send_keys(email)
            next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
            next_button.click()
        except NoSuchElementException:
            time.sleep(15)
            username_field = driver.find_element(By.NAME, "text")
            username_field.send_keys(email)
            next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
            next_button.click()
    # Login username
    time.sleep(5)
    try:
        error_message = driver.find_element(By.XPATH, '//span[text()="输入你的手机号码或用户名"]')
        if error_message:
            try:
                new_username_field = driver.find_element(By.NAME, 'text')
                new_username_field.send_keys(username)
                next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
                next_button.click()
            except NoSuchElementException:
                time.sleep(10)
                try:
                    new_username_field = driver.find_element(By.NAME, 'text')
                    new_username_field.send_keys(username)
                    next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
                    next_button.click()
                except NoSuchElementException:
                    time.sleep(20)
                    new_username_field = driver.find_element(By.NAME, 'text')
                    new_username_field.send_keys(username)
                    next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
                    next_button.click()       
    except NoSuchElementException:
        pass
    # Login password
    time.sleep(5)
    try:
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys(password)
        next_button = driver.find_element(By.XPATH, '//span[text() = "登录"]')
        next_button.click()
    except NoSuchElementException:
        time.sleep(10)
        try:
            password_field = driver.find_element(By.NAME, "password")
            password_field.send_keys(password)
            next_button = driver.find_element(By.XPATH, '//span[text() = "登录"]')
            next_button.click()
        except NoSuchElementException:
            time.sleep(15)
            password_field = driver.find_element(By.NAME, "password")
            password_field.send_keys(password)
            next_button = driver.find_element(By.XPATH, '//span[text() = "登录"]')
            next_button.click()
    # If email is inspected 
    time.sleep(5)
    try:
        error_message = driver.find_element(By.XPATH, '//span[text()="检查你的邮箱"]')
        if error_message:
            time.sleep(10)
            new_email_field = driver.find_element(By.NAME, 'text')
            verified_code = get_mail_verified(email, email_password)
            if verified_code is None:
                driver = None
            else:
                new_email_field.send_keys(verified_code)
                next_button = driver.find_element(By.XPATH, '//span[text()="下一步"]')
                next_button.click()
    except NoSuchElementException:
        pass
    except TimeoutException:
        driver = None
    # If Extra Authentication is needed, try next account.
    time.sleep(3)
    try:
        heading1 = driver.find_element(By.XPATH, '//h2[contains(text(),"Authenticate your account")]')
        heading2 = driver.find_element(By.XPATH, "//button[contains(text(), 'Authenticate')]")
        if heading1 or heading2:
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
    following_name = following_name.replace("@", "")
    url = f'https://twitter.com/{following_name}'
    driver.get(url)
    if "These tweets are protected" in driver.page_source:
        return "next"
    if "Something went wrong. Try reloading." in driver.page_source:
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
        followers = wait.until(EC.presence_of_element_located((By.XPATH, f'//a[@href="/{following_name}/followers" and @role= "link"]//span[contains(translate(text(), "0123456789", ""),text())]')))
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
def get_starting_point_followings(driver, start_point):
    page_sources = []
    driver.get(f'https://twitter.com/{start_point}/following') 
    time.sleep(7.5) #等待页面加载
    driver.maximize_window()
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
    following_names = parse_page_sources(page_sources)
    return following_names
#%%
def login_and_check_account(account):
    email_, username, password, email_password = account
    try:
        driver = twitter_login(email_, username, password, email_password, None, None, None, None)
        return driver
    except:
        try:
            driver.close()
        except:
            pass
        return None
#%%
def process_account(driver, accounts, start_point,sql_connector, database_tosave):
    visited = set()
    queue = deque([(start_point, 0)])
    max_depth = 20
    popular_names = []
    popular_followers = []
    popular_followings = []
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
        if popular == "next account":
            account = next(accounts)
            driver = login_and_check_account(account)
            while driver is None:
                account = next(accounts)
                driver = login_and_check_account(account)
            popular = check_popularity(driver, following_name)
        elif popular != 'next' and popular is not None:
            popular_name, popular_follower, popular_following = popular
            popular_names.append(popular_name)
            popular_followers.append(popular_follower)
            popular_followings.append(popular_following)
            
            if "These tweets are protected" not in driver.page_source:
                followers = get_starting_point_followings(driver, popular_name)
                for follower in followers:
                    if follower not in visited:
                        queue.append((follower, depth + 1))
        if len(popular_names) % 10 == 0:
            save_to_database(popular_names, popular_followers, popular_followings)
            popular_names = []
            popular_followers = []
            popular_followings = []
def save_to_database(popular_names, popular_followers, popular_followings):
    dataframe = pd.DataFrame()
    dataframe['usernames'] = popular_names
    dataframe['followers'] = popular_followers
    dataframe['followings'] = popular_followings

    engine = create_engine(sql_connector)
    existing_usernames = pd.read_sql(f"SELECT usernames FROM {database_tosave}", engine)
    dataframe = dataframe[~dataframe['usernames'].isin(existing_usernames['usernames'])]
    dataframe.to_sql(name = database_tosave, con = engine, index = False, if_exists = 'append')
    engine.dispose()
#%%
start_point = "CryptoZhaoX"     
account_filepath = r"./all_accounts"
sql_connector = "your_database"
database_tosave = "popular_accounts_df"
with open(account_filepath, 'rb') as f:
    all_accounts = pickle.load(f)
all_accounts = list(all_accounts)[:25]
accounts = cycle(all_accounts)
account = next(accounts)
driver = login_and_check_account(account)
while driver is None:
    account = next(accounts)
    email_, username, password, email_password = account
    driver = login_and_check_account(account) 
process_account(driver, accounts, start_point, sql_connector, database_tosave)
driver.close()
