# 由于没有系统学过selenium，因此这个爬虫不稳定
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


def get_next_btn(c):
    driver.get("https://www.xqbase.com/xqbase/?")
    driver.find_element(By.TAG_NAME, 'input').submit()


def jump_main_page():
    driver.get("https://www.xqbase.com/xqbase/?")
    driver.find_element(By.TAG_NAME, 'input').submit()


def jump_to_next_paeg(j):
    a_list = driver.find_elements(By.TAG_NAME, 'a')
    for i in range(len(a_list)):
        a = a_list[i]
        if '>' == str(a.text):
            a.click()
            break
        a_list = driver.find_elements(By.TAG_NAME, 'a')
    time.sleep(1)


def get_before_download_list():
    before_download_list = []
    s = set()
    a_list = driver.find_elements(By.TAG_NAME, 'a')

    for i in range(len(a_list)):
        a = a_list[i]
        if "?gameid=" in str(a.get_attribute('href')) and a.get_attribute("href") not in s:
            s.add(a.get_attribute("href"))
            before_download_list.append(a)
        a_list = driver.find_elements(By.TAG_NAME, 'a')
    return before_download_list


def download_a_file(to):
    to.click()
    driver.switch_to.window(driver.window_handles[1])
    download_list = driver.find_elements(By.TAG_NAME, "a")
    for i in range(len(download_list)):
        try:
            d = download_list[i]
            img = d.find_element(By.TAG_NAME, "img")
            if img and str(img.get_attribute("src")).endswith("/images/pgn.gif"):
                d.click()
                win = driver.window_handles
                driver.switch_to.window(win[1])
                driver.close()
                driver.switch_to.window(win[0])
                break
        except:
            pass
        download_list = driver.find_elements(By.TAG_NAME, "a")


driver = webdriver.Chrome()
driver.get("https://www.xqbase.com/xqbase/?")
driver.find_element(By.TAG_NAME, 'input').submit()

for j in range(50):
    for i in range(20):
        before_download_list = get_before_download_list()
        download_a_file(before_download_list[i])
    jump_to_next_paeg(j)

time.sleep(10)
driver.quit()
