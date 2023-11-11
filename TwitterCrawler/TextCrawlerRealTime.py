#!/usr/bin/python3
#%%
# Packages
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
from datetime import datetime, timedelta
from selenium.webdriver.support.select import Select
from itertools import cycle
import imaplib
import email
import pickle
from sqlalchemy import create_engine
import pandas as pd
import multiprocessing
import random
#%%
## Deal mail inspection if encountered
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
        mail.select("inbox")
        # Search Mails
        result, data = mail.uid('search', None, "ALL")
        # Fetch the Mail List
        email_list = data[0].split()
        if email_list:
            # Fetch the latest mail
            latest = email_list[-1]
            # Fetch the body of the latest mail
            result, email_data = mail.uid('fetch', latest, '(BODY.PEEK[TEXT])')
            raw_email = email_data[0][1].decode("utf-8")
            # Parse raw mail
            email_message = email.message_from_string(raw_email)
        
            # Get Verification Code
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    html_content = part.get_payload(decode=True)
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                    clean_text = re.sub(r'\s+', ' ', text)
                    verified_code = re.search(r'single-use code. (\w+)', clean_text)
                    if verified_code:
                        code = verified_code.group(1)
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
# get kols_list if lists are provided
'''
kols_lists = ['https://twitter.com/i/lists/1464058952732184580/members',
              'https://twitter.com/i/lists/1414566684506140672/members',
              'https://twitter.com/i/lists/1628015350351798272/members',
              'https://twitter.com/i/lists/1627578122332958720/members',
              'https://twitter.com/i/lists/1494560668200046592/members',
              'https://twitter.com/i/lists/1626952281638965251/members',
              'https://twitter.com/i/lists/1626943570379960320/members']
def kol_list(driver, kols_lists):
    usernames_ = []
    for kols in kols_lists:
        driver.get(kols)
        time.sleep(5)
        previous_page_source = driver.page_source
        # 获取 "Who to follow" 元素
        who_to_follow = driver.find_element(By.XPATH, '//span[contains(text(), "Who to follow")]')
        # 获取所有的用户元素
        all_user_cells = driver.find_elements(By.XPATH, '//a[@aria-hidden="true" and @tabindex="-1"]')
        # 初始化一个列表来存放 "Who to follow" 之前的用户元素
        user_cells_before = []
        
        for cell in all_user_cells:
            # 比较每个用户元素和 "Who to follow" 元素的位置
            # 如果用户元素的位置在 "Who to follow" 元素之前，则添加到列表中
            if cell.location['y'] < who_to_follow.location['y']:
                user_cells_before.append(cell)
        
        # 获取这些元素的 outerHTML
        htmls = [e.get_attribute('outerHTML') for e in user_cells_before]
        new_usernames_ = [re.search(r'href="/(\w+)"', text).group(1) for text in htmls if re.search(r'href="/(\w+)"', text)]
        
        for username in new_usernames_:
            if username not in usernames_:
                usernames_.append(username)
        
        scroll_count = 0
        while True:

            actions = ActionChains(driver)
            actions.send_keys(Keys.PAGE_DOWN)
            actions.perform()
            time.sleep(3) 
            
            current_page_source = driver.page_source # 获取当前页面的源码或内容
            
            # 比较前后两次页面源码或内容
            if current_page_source != previous_page_source:
                # 页面有新内容，获取新内容
                # 获取 "Who to follow" 元素
                who_to_follow = driver.find_element(By.XPATH, '//span[contains(text(), "Who to follow")]')
                
                # 获取所有的用户元素
                all_user_cells = driver.find_elements(By.XPATH, '//a[@aria-hidden="true" and @tabindex="-1"]')
                
                # 初始化一个列表来存放 "Who to follow" 之前的用户元素
                user_cells_before = []
                
                for cell in all_user_cells:
                    # 比较每个用户元素和 "Who to follow" 元素的位置
                    # 如果用户元素的位置在 "Who to follow" 元素之前，则添加到列表中
                    if cell.location['y'] < who_to_follow.location['y']:
                        user_cells_before.append(cell)
                
                # 获取这些元素的 outerHTML
                htmls = [e.get_attribute('outerHTML') for e in user_cells_before]
                new_usernames_ = [re.search(r'href="/(\w+)"', text).group(1) for text in htmls if re.search(r'href="/(\w+)"', text)]
        
        
                for username in new_usernames_:
                    if username not in usernames_:
                        usernames_.append(username)
        
                previous_page_source = current_page_source
                scroll_count += 1
        
                # 根据需要调整滑动次数的上限
                if scroll_count >= 50:
                    break
            else:
                # 没有新内容加载，说明滑动到了页面底部，停止滑动
                break
    usernames_.append("WuBlockchain")
    return usernames_
'''
#%%
# Tab = "Top" or "Latest"
def top_or_latest(driver, tab):
    try:
        element = driver.find_element(By.XPATH, f'//span[text() = "{tab}"]')
        driver.execute_script("arguments[0].click();", element)
    except NoSuchElementException:
        time.sleep(20)
        try:
            element = driver.find_element(By.XPATH, f'//span[text() = "{tab}"]')
            driver.execute_script("arguments[0].click();", element)
        except NoSuchElementException:
            time.sleep(20)
            try:
                element = driver.find_element(By.XPATH, f'//span[text() = "{tab}"]')
                driver.execute_script("arguments[0].click();", element)
            except NoSuchElementException:
                time.sleep(60)
                element = driver.find_element(By.XPATH, f'//span[text() = "{tab}"]')
                driver.execute_script("arguments[0].click();", element)
    time.sleep(5)
    return driver  
#%%
# Set Crawler Time Scope
def time_scope(i, start_date, end_date, days_per_call):
    extra_day = 1 if i > 1 else 0
    current_date = min(start_date + timedelta((i-1) * days_per_call) + timedelta(extra_day), end_date)
    call_end_date = min(current_date + timedelta(days_per_call), end_date)
    return str(current_date.month), str(current_date.day), str(current_date.year), str(call_end_date.month), str(call_end_date.day), str(call_end_date.year)
#%%
# Search specific user's tweets between specific dates (Not available under Headless mode)
def advanced_search_1(query, driver, f_m, f_d, f_y, t_m, t_d, t_y):
    try:
        advanced_search = driver.find_element(By.XPATH, '//span[text() = "Advanced search"]')
        driver.execute_script("arguments[0].click();", advanced_search)
    except NoSuchElementException:
        time.sleep(5)
        try:
            advanced_search = driver.find_element(By.XPATH, '//span[text() = "Advanced search"]')
            driver.execute_script("arguments[0].click();", advanced_search)
        except NoSuchElementException:
            time.sleep(20)
            try:
                advanced_search = driver.find_element(By.XPATH, '//span[text() = "Advanced search"]')
                driver.execute_script("arguments[0].click();", advanced_search)
            except NoSuchElementException:
                time.sleep(20)
                advanced_search = driver.find_element(By.XPATH, '//span[text() = "Advanced search"]')
                driver.execute_script("arguments[0].click();", advanced_search)
                
    time.sleep(5)
    try:
        input_box = driver.find_element(By.NAME, 'fromTheseAccounts')
        driver.execute_script("arguments[0].value = arguments[1];", input_box, query)
        
        from_elements_div = driver.find_elements(By.XPATH,"//div[@aria-label='From']")
        to_elements_div = driver.find_elements(By.XPATH, "//div[@aria-label='To']")
    except NoSuchElementException:
        time.sleep(10)
        try:
            input_box = driver.find_element(By.NAME, 'fromTheseAccounts')
            driver.execute_script("arguments[0].value = arguments[1];", input_box, query)
            from_elements_div = driver.find_elements(By.XPATH,"//div[@aria-label='From']")
            to_elements_div = driver.find_elements(By.XPATH, "//div[@aria-label='To']")
        except NoSuchElementException:
            time.sleep(10)
            input_box = driver.find_element(By.NAME, 'fromTheseAccounts')
            driver.execute_script("arguments[0].value = arguments[1];", input_box, query)
            
            from_elements_div = driver.find_elements(By.XPATH,"//div[@aria-label='From']")
            to_elements_div = driver.find_elements(By.XPATH, "//div[@aria-label='To']")

    from_elements = []
    for item in from_elements_div:
        elements = item.find_elements(By.XPATH,".//select[@aria-invalid='false' and starts-with(@id, 'SELECTOR_')]")
        from_elements.extend(elements)
    
    to_elements = []
    for item in to_elements_div:
        elements = item.find_elements(By.XPATH,".//select[@aria-invalid='false' and starts-with(@id, 'SELECTOR_')]")
        to_elements.extend(elements)

    from_month = from_elements[0]
    from_day = from_elements[1]
    from_year = from_elements[2]
    to_month = to_elements[0]
    to_day = to_elements[1]
    to_year = to_elements[2]
    
    select_month_from = Select(from_month)
    select_day_from = Select(from_day)
    select_year_from = Select(from_year) 
    select_month_to = Select(to_month)
    select_day_to = Select(to_day)
    select_year_to = Select(to_year)
    
    select_month_from.select_by_value(f_m)
    select_day_from.select_by_value(f_d)
    select_year_from.select_by_value(f_y)
    select_month_to.select_by_value(t_m)
    select_day_to.select_by_value(t_d)
    select_year_to.select_by_value(t_y)
    
    time.sleep(2)
    search = driver.find_element(By.XPATH, '//span[text() = "Search"]')
    search.click()
    time.sleep(3)
#%%
# Search specific user's tweets between specific dates (Headless mode)
def advanced_search_2(query, driver, f_m, f_d, f_y, t_m, t_d, t_y,tab):
    if tab == "Latest":
        search_query = f'https://twitter.com/search?q=(from%3A{query})%20until%3A{t_y}-{t_m}-{t_d}%20since%3A{f_y}-{f_m}-{f_d}&src=typed_query&f=live'
    elif tab == "Top":
        search_query = f'https://twitter.com/search?q=(from%3A{query})%20until%3A{t_y}-{t_m}-{t_d}%20since%3A{f_y}-{f_m}-{f_d}&src=typed_query'
    driver.get(search_query)
#%%
# Leave out long tweets (Long Tweets are saved in another container).
def filter_show_more(html):
    soup = BeautifulSoup(html, 'html.parser')
    tweet_texts = soup.find_all(attrs = {"data-testid": "tweet"})
    
    for tweet_text in tweet_texts:
        if tweet_text.find(attrs = {"data-testid": "tweet-text-show-more-link"}):
            tweet_text.decompose()
    new_soup = soup
    return new_soup
#%%
# Click to show full text of long tweets.
def find_show_more_buttons(driver, page_sources_2):
    i = 0
    while True:
        try:
            show_more_buttons = driver.find_elements(By.XPATH, '//span[@data-testid="tweet-text-show-more-link"]')
            if i >= len(show_more_buttons) - 1:
                break
            show_more_button = show_more_buttons[i]
        except NoSuchElementException:
            break
        
        # Execute JavaScript to click the button
        driver.execute_script("arguments[0].click();", show_more_button)
        time.sleep(6)

        show_more_page_source = driver.page_source
        page_sources_2.append(BeautifulSoup(show_more_page_source, 'html.parser'))

        driver.back()
        i += 1
    return driver, page_sources_2
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
# parse short tweets
def parse_tweets_1(page_source,query):
    tweets = page_source.find_all('article', attrs={'data-testid': 'tweet'})
    tweets_info = []
    if len(tweets) > 0:
        for tweet_element in tweets:
            tweet_text_element = tweet_element.find('div', attrs={'data-testid': 'tweetText'})
            tweet_time_element = tweet_element.find('time')
            tweet_stats_element = tweet_element.find('div', attrs={'role': 'group'})
            try:
                tweet_view_element = tweet_element.find('a', attrs={'role': 'link', 'class': 'css-4rbku5 css-18t94o4 css-1dbjc4n r-1loqt21 r-1777fci r-bt1l66 r-1ny4l3l r-bztko3 r-lrvibr'})
                tweet_view = tweet_view_element.find('span', class_='css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0')
            except:
                pass
            tweet_author_elements = tweet_element.find('div', attrs={'data-testid': 'User-Name'})
            if tweet_text_element is not None and tweet_time_element is not None:
                tweet_text = tweet_text_element.get_text()
                tweet_time = tweet_time_element['datetime']
            else:
                break            

            for item in tweet_author_elements:
                names = item.find('span', attrs={'class': 'css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0'})
                if names:
                    author_name = names.get_text()
                    author_name = author_name.replace('@', '')
                else:
                    author_name = query
    
            if tweet_view is not None:
                views = convert_to_int(tweet_view.get_text())
            else:
                views = 0
        
            replies, retweets, likes = 0, 0, 0

            if tweet_stats_element is not None:
                stats_text = tweet_stats_element['aria-label']
                if stats_text != "":
                    stats_parts = stats_text.split(',')
            
                    for part in stats_parts:

                        part = part.lstrip()
                        stat_type = part.split(' ')[1]
                        stat_value = int(part.split(' ')[0])
                    
                        if stat_type.lower() == 'replies' or stat_type.lower() == "reply":
                            replies = stat_value
                        elif stat_type.lower() == 'retweets' or stat_type.lower() == "retweet":
                            retweets = stat_value
                        elif stat_type.lower() == 'likes' or stat_type.lower() == 'like':
                            likes = stat_value
                
                tweets_info.append((author_name,tweet_text, tweet_time, replies, retweets, likes, views))
        return tweets_info
    else:
        return None
#%%
# parse long tweets
def parse_tweets_2(page_source,query): 
    tweets = page_source.find('article', attrs = {'data-testid': 'tweet'})
    tweets_info = []
    if len(tweets) > 0:
        for tweet_element in tweets:
            tweet_text_element = tweet_element.find('div', attrs={'data-testid': 'tweetText'})
            tweet_time_element = tweet_element.find('time')
            tweet_stats_elements = tweet_element.find_all('div', {'class': 'css-1dbjc4n r-xoduu5 r-1udh08x'})
            tweet_author_elements = tweet_element.find('div', attrs={'data-testid': 'User-Name'})
        
            if tweet_text_element is not None and tweet_time_element is not None:
                tweet_text = tweet_text_element.get_text()
                tweet_time = tweet_time_element['datetime']
            else:
                break
            for item in tweet_author_elements:
                names = item.find('span', attrs={'class': 'css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0'})
                if names:
                    author_name = names.get_text()
                    author_name = author_name.replace('@', '')
                else:
                    author_name = query
            stats_values = [convert_to_int(e.get_text()) for e in tweet_stats_elements]

            while len(stats_values) < 5:
                stats_values.append(0)

            views, replies, retweets, likes, bookmarks = stats_values

            tweets_info.append((author_name, tweet_text, tweet_time, replies, retweets, likes, views))
        return tweets_info
    else:
        return None
#%%
# organize tweets data
def get_dataframe(page_sources_1, page_sources_2, query):
    all_tweets = []
    for page_source in page_sources_1:
        tweets_info = parse_tweets_1(page_source,query)
        if tweets_info is not None:
            all_tweets.append(tweets_info)
    for page_source in page_sources_2:
        tweets_info = parse_tweets_2(page_source,query)
        if tweets_info is not None:
            all_tweets.append(tweets_info)
    if len(all_tweets) > 0:
        all_tweets_ = [item for sublist in all_tweets for item in sublist]
        tweets_df = pd.DataFrame(all_tweets_, columns = ['username', 'text','date', 'replies', 'retweets', 'likes', 'views'])
        tweets_df['date'] = pd.to_datetime(tweets_df['date'])
        tweets_df.sort_values(by = 'date',ascending = False, inplace = True)
        tweets_df = tweets_df.drop_duplicates(subset=['text','date','username']) 
        return tweets_df
#%%
# avoid error
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
def get_query(driver, query, tab, counts, start_date, end_date, days_per_call):
    try:
        driver.get(f'https://twitter.com/search?q={query}&src=typed_query')
        time.sleep(7.5)
        counts += 1
        try:
            f_m, f_d, f_y, t_m, t_d, t_y = time_scope(counts, start_date, end_date, days_per_call)
            advanced_search_2(query, driver, f_m, f_d, f_y, t_m, t_d, t_y,tab)
        except:
            try:
                driver.quit()
            except:
                pass
            counts -= 1
            return None, counts
        return driver, counts
    except:
        try:
            driver.quit()
        except:
            pass
        return None, counts
#%%
import pytz
def safely_localize_to_utc(series, timezone=pytz.UTC):
    if hasattr(series.iloc[0], 'tz') and series.iloc[0].tz:  # if already timezone-aware
        return series.dt.tz_convert(timezone)
    else:
        return series.apply(lambda x: x.tz_localize(timezone) if pd.notna(x) else x)
#%%
def save_to_database(sql_connector, tweets_df, database_tosave):
    engine = create_engine(sql_connector)
    tweets_df['date'] = safely_localize_to_utc(tweets_df['date'])
    try:
        existing_data_query = f"SELECT username, text, date FROM {database_tosave}"
        existing_data = pd.read_sql(existing_data_query, engine)
        existing_data['date'] = safely_localize_to_utc(existing_data['date'])
        # Merge dataframes
        merged = pd.merge(tweets_df, existing_data, on=['username', 'text', 'date'], how='left', indicator=True)

        # Rows to insert
        to_insert = merged[merged['_merge'] == 'left_only'][tweets_df.columns]

        # Rows to potentially update
        to_update = merged[merged['_merge'] == 'both']

        # Update only if the other columns differ
        for index, row in to_update.iterrows():
            # Fetch the existing row from the database
            existing_row = pd.read_sql(f"SELECT * FROM {database_tosave} WHERE username = %s AND text = %s AND date = %s", engine, params=(row['username'], row['text'], row['date'])).iloc[0]
            # Compare the non-key columns and update if they differ
            if not (existing_row['likes'] == row['likes'] and existing_row['retweets'] == row['retweets'] and existing_row['views'] == row['views'] and existing_row['replies'] == row['replies']):
                update_query = f"""
                UPDATE {database_tosave} SET
                replies = %s,
                retweets = %s,
                likes = %s,
                views = %s
                WHERE username = %s AND text = %s AND date = %s
                """
                engine.execute(update_query, (row['replies'], row['retweets'], row['likes'], row['views'], row['username'], row['text'], row['date']))

        # Insert new rows
        to_insert.to_sql(name=database_tosave, con=engine, index=False, if_exists='append')
        engine.dispose()
    except:
        tweets_df.to_sql(name=database_tosave, con=engine, index=False, if_exists='append')
        engine.dispose()
#%%
def get_twitter_page(queries, account_filepath, all_accounts, i,tab,sql_connector,database_tosave,start_date, end_date, days_per_call, proxy_username = None, proxy_passwords = None, proxy_ip = None, proxy_port = None):
    page_sources_1 = [] #用来存储不包含show_more的源代码
    page_sources_2 = [] #用来存储包含show_more的源代码
    used_accounts = []
    counts_for_queries = 0
    accounts = cycle(all_accounts)
    for query in queries:
        if len(all_accounts) < 2:
            with open(account_filepath, 'rb') as f:
                all_accounts = pickle.load(f)
            all_accounts = list(all_accounts)
            all_accounts.reverse()
            used_accounts = []
            accounts = cycle(all_accounts)
        counts = 0
        if counts_for_queries % i == 0 or counts_for_queries == 0:
            try:
                driver.quit()
            except:
                pass
            all_accounts = [account for account in all_accounts if account not in used_accounts]
            account = next(accounts)
            driver = login_and_check_account(account)
            while driver is None:
                used_accounts.append(account)
                account = next(accounts)
                driver = login_and_check_account(account)
                used_accounts.append(account)
        driver, counts = get_query(driver, query, tab, counts, start_date, end_date, days_per_call)
        while driver is None:
            while driver is None:
                used_accounts.append(account)
                account = next(accounts)
                driver = login_and_check_account(account)
                used_accounts.append(account)
            driver, counts = get_query(driver, query, tab, counts, start_date, end_date, days_per_call)
        counts_for_queries += 1
        
        previous_page_source = driver.page_source #previous源代码
        html_1 = driver.page_source
        soup = BeautifulSoup(html_1, 'html.parser')
#        driver, page_sources_2 = find_show_more_buttons(driver, page_sources_2)
    
#        filtered_html = filter_show_more(html_1)
        page_sources_1.append(soup)
        
        scroll_count = 0
        while True:
            actions = ActionChains(driver)
            actions.send_keys(Keys.PAGE_DOWN)
            actions.perform()
            actions.send_keys(Keys.PAGE_DOWN)
            actions.perform()
            
            time.sleep(5)
                
            current_page_source = driver.page_source # 获取当前页面的源码或内容
            
            # 比较前后两次页面源码或内容
            if current_page_source == previous_page_source:
                # 没有新内容加载，说明滑动到了页面底部，停止滑动
                break
            
            # 有新内容加载，继续滑动
            scroll_count += 1
            if scroll_count >= 500:  # 根据需要调整滑动次数的上限
                break
        
            html_1 = driver.page_source
            soup = BeautifulSoup(html_1, 'html.parser')
#            driver, page_sources_2 = find_show_more_buttons(driver, page_sources_2)
            
#            filtered_html = filter_show_more(html_1)
            page_sources_1.append(soup)
            previous_page_source = current_page_source
        time.sleep(3)
        tweets_df = get_dataframe(page_sources_1, page_sources_2,query)
        if tweets_df is not None:
            try:
                save_to_database(sql_connector, tweets_df, database_tosave)
                page_sources_1 = [] 
                page_sources_2 = []
            except:
                pass 
        print(query)
#%%
# Define const
account_filepath = r"./all_accounts"
kol_filepath = r"./kol_list.pickle" 
tab = "Latest"
start_date = datetime.now() - timedelta(days = 4) #爬取推文的起始时间
end_date = datetime.now() + timedelta(days = 1) #爬取推文的结束时间
days_per_call = 5 #一次所爬推文的时间跨度
i = 15 #爬15次换一个账户
sql_connector = "your_database_connector"
database_tosave = "you_database_name"
def get_expanding_kols(sql_connector):
    engine = create_engine(sql_connector)
    expanding_kols = pd.read_sql("SELECT username FROM xxx(your_database_name_for_expanding_kols)", engine)
    expanding_kols = expanding_kols['username'].tolist()
    return expanding_kols
# Load Data
# get twitter accounts
with open(account_filepath, 'rb') as f:
    all_accounts = pickle.load(f)
all_accounts = list(all_accounts)
# get kol list
with open(kol_filepath, "rb") as f:
    queries = pickle.load(f)
expanding_kols = get_expanding_kols(sql_connector)
queries = list(set(expanding_kols + queries))
# Shuffle the lists
random.shuffle(all_accounts)
random.shuffle(queries)
# Split data into chunks
num_processes = 10  # Set the number of processes as needed
chunk_size_1 = len(all_accounts) // num_processes
chunk_size_2 = len(queries) // num_processes
account_chunks = [all_accounts[i:i + chunk_size_1] for i in range(0, len(all_accounts), chunk_size_1)]
queries_chunks = [queries[i:i + chunk_size_2] for i in range(0, len(queries), chunk_size_2)]

# Create and start processes
processes = []
for accounts_chunk, queries_chunk in zip(account_chunks, queries_chunks):
    process = multiprocessing.Process(target=get_twitter_page, args=(queries_chunk,account_filepath, accounts_chunk, i, tab, sql_connector, database_tosave, start_date, end_date, days_per_call))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()