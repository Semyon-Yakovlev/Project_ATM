from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
from time import sleep
from selenium.webdriver.common.by import By
from streets import *
from places import *
import json_pattern
import util_module
from infogetter import InfoGetter


class GrabberApp:

    def __init__(self, city, street, org_type):
        self.city = city
        self.street = street
        self.org_type = org_type

    def grab_data(self):

        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get('https://yandex.ru/maps')
        sleep(4)
        # Вводим данные поиска
        driver.find_element(By.CLASS_NAME,'input__control._bold').send_keys(self.city + ' ' + self.street + ' ' + self.org_type)

        # Нажимаем на кнопку поиска
        driver.find_element(By.CLASS_NAME,'small-search-form-view__button').click()
        sleep(4)

        slider = driver.find_element(By.CLASS_NAME,'scroll__scrollbar-thumb')
        # Основная вкладка со списком всех организаций
        parent_handle = driver.window_handles[0]

        org_id = 0
        organizations_href = ""
        try:
            for i in range(10000):
                # Симулируем прокрутку экрана на главной странице поиска
                try:
                    ActionChains(driver).click_and_hold(slider).move_by_offset(0, 120).release().perform()
                except:
                    pass
                # Подгружаем ссылки на организации каждые 5 итераций
                if (org_id == 0) or (org_id % 5 == 0):
                    organizations_href = driver.find_elements(By.CLASS_NAME,'search-snippet-view__link-overlay')
                organization_url = organizations_href[i].get_attribute("href")

                # Открываем карточку организации в новой вкладке
                driver.execute_script(f'window.open("{organization_url}","org_tab");')
                child_handle = [x for x in driver.window_handles if x != parent_handle][0]
                driver.switch_to.window(child_handle)
                sleep(1)

                soup = BeautifulSoup(driver.page_source, "lxml")
                org_id += 1
                name = InfoGetter.get_name(soup)
                address = InfoGetter.get_address(soup)
                coords = driver.current_url.split('/')[-1]
                rating = InfoGetter.get_rating(soup)
                current_url_split = driver.current_url.split('/')
                #  Переходим на вкладку "Отзывы"
                reviews_url = 'https://yandex.ru/maps/org/' + current_url_split[5] + '/' + current_url_split[6] + \
                              '/reviews'
                driver.get(reviews_url)
                sleep(2)

                reviews_count,reviews_rating = InfoGetter.get_ratings(soup, driver)

                # Записываем данные в OUTPUT.json
                output = json_pattern.into_json(org_id, self.org_type, name, address, self.street, coords, rating, reviews_count, reviews_rating)
                util_module.JSONWorker("set", output)
                print(f'Данные добавлены, id - {org_id}')

                # Закрываем вторичную вкладу и переходим на основную
                driver.close()
                driver.switch_to.window(parent_handle)
                sleep(1)
        except Exception:
            pass

        driver.quit()
        print('Данные сохранены в OUTPUT.json')
        return




def main():
    city = 'Москва'
    for street in streets:
        for org_type in places:
            grabber = GrabberApp(city, street, org_type)
            grabber.grab_data()


if __name__ == '__main__':
    main()
