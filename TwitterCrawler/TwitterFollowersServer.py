import pickle
from itertools import cycle
from flask import Flask, jsonify, request
from TwitterFollowersHTTP import twitter_login, get_user_following,login_checking

app = Flask(__name__)
account_filepath = r"./all_accounts"
with open(account_filepath, 'rb') as f:
    all_accounts = pickle.load(f)
all_accounts = list(all_accounts)
all_accounts = cycle(all_accounts)
driver = login_checking(all_accounts)
@app.route('/get_user_following', methods=['GET'])
def get_user_following_route():
    global driver
    following_name = request.args.get('username')
    if not following_name:
        return jsonify({"error": "Username not provided"}), 400
    result = get_user_following(driver, following_name)  
    
    if isinstance(result, tuple):
        return jsonify({
            "username": result[0],
            "followers_count": result[1],
            "followings_count": result[2]
        })
    elif isinstance(result, str) and result == "next":
        return jsonify({"error": "This is a private account"}), 400
    elif isinstance(result, str) and result == "next account":
        try:
            driver.quit()
        except:
            pass
        driver = login_checking(all_accounts)
        result = get_user_following(driver, following_name)  
        while result == "next account":
            driver = login_checking(all_accounts)
            result = get_user_following(driver, following_name)  
        if isinstance(result, tuple):
            return jsonify({
                "username": result[0],
                "followers_count": result[1],
                "followings_count": result[2]
            })
        elif isinstance(result, str) and result == "next":
            return jsonify({"error": "This is a private account"}), 400
    else:
        return jsonify({"error": "Unknown error occurred"}), 500