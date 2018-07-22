import urllib.request
from bs4 import BeautifulSoup
import time
import re

def run():
    i = 1
    while i > 0:
        url = "https://blog.csdn.net/weixin_28750267"
        user_agent = "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
        request = urllib.request.Request(url)
        request.add_header("User-Agent", user_agent)
        content = urllib.request.urlopen(request)
        soup = BeautifulSoup(content,'html.parser')

        titles = soup.find_all(href=re.compile("details"))
        for title in titles:

            url = title.get('href')
            print(url)
            try:
                request1 = urllib.request.Request(url)
                request1.add_header("User-Agent", user_agent)
                content = urllib.request.urlopen(request1)
            except:
                print("NO %s time out" % i)
                continue
            print("NO %s complete" % i)
            i = i + 1
            time.sleep(15)

if __name__ == '__main__':
    run()