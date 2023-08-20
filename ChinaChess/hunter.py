from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://www.xqbase.com/xqbase/?")
driver.find_element(By.TAG_NAME, 'input').submit()

a_list = driver.find_elements(By.TAG_NAME, 'a')


def get_next_btn(c):
    driver.get("https://www.xqbase.com/xqbase/?")
    driver.find_element(By.TAG_NAME, 'input').submit()


def get_before_download_list(c):
    before_download_list = []
    s = set()

    driver.get("https://www.xqbase.com/xqbase/?")
    driver.find_element(By.TAG_NAME, 'input').submit()
    a_list = driver.find_elements(By.TAG_NAME, 'a')

    for a in a_list:
        if str(c) == str(a.text):
            print("c = {}, 第{}此".format(c, i))
            a.click()
            break

    time.sleep(1)
    a_list = driver.find_elements(By.TAG_NAME, 'a')
    for a in a_list:
        if "?gameid=" in str(a.get_attribute('href')) and a.get_attribute("href") not in s:
            s.add(a.get_attribute("href"))
            before_download_list.append(a)
    return before_download_list


def download_a_file(to):
    to.click()
    driver.get(str(to.get_attribute('href')))
    download_list = driver.find_elements(By.TAG_NAME, "a")
    for d in download_list:
        try:
            img = d.find_element(By.TAG_NAME, "img")
            if img and str(img.get_attribute("src")).endswith("/images/pgn.gif"):
                d.click()
                win = driver.window_handles
                driver.switch_to.window(win[1])
                driver.close()
                driver.switch_to.window(win[0])
                driver.back()
                break
        except:
            pass


for j in range(50):
    for i in range(20):
        before_download_list = get_before_download_list(j)
        download_a_file(before_download_list[i])


time.sleep(10)
driver.quit()
