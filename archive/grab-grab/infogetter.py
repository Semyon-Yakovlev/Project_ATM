from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    MoveTargetOutOfBoundsException,
    NoSuchElementException,
)
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By


class InfoGetter(object):
    """Класс с логикой парсинга данных из объекта BeautifulSoup"""

    @staticmethod
    def get_name(soup_content):
        """Получение названия организации"""

        try:
            for data in soup_content.find_all(
                "h1", {"class": "orgpage-header-view__header"}
            ):
                name = data.getText()

            return name
        except Exception:
            return ""

    @staticmethod
    def get_address(soup_content):
        """Получение адреса организации"""

        try:
            for data in soup_content.find_all(
                "a", {"class": "business-contacts-view__address-link"}
            ):
                address = data.getText()

            return address
        except Exception:
            return ""

    @staticmethod
    def get_rating(soup_content):
        """Получение рейтинга организации"""

        rating = ""
        try:
            for data in soup_content.find_all(
                "span", {"class": "business-summary-rating-badge-view__rating-text"}
            ):
                rating += data.getText()
            return rating
        except Exception:
            return ""

    @staticmethod
    def get_ratings(soup_content, driver):
        """Получение рейтинга о организации"""

        reviews_rating = []
        try:
            slider = driver.find_element(By.CLASS_NAME, "scroll__scrollbar-thumb")
        except:
            slider = False
            pass

        # Узнаём количество отзывов
        try:
            reviews_count = int(
                soup_content.find_all("div", {"class": "tabs-select-view__counter"})[
                    -1
                ].text
            )

        except ValueError:
            print("get_reviews ValueError")
            reviews_count = 0

        except AttributeError:
            reviews_count = 0

        except Exception:
            return ""
        if reviews_count > 150:
            find_range = range(100)
        else:
            find_range = range(30)

        try:
            soup_content = BeautifulSoup(driver.page_source, "lxml")
            for el in soup_content.find_all("meta", {"itemprop": "ratingValue"}):
                reviews_rating.append(el.get("content"))

            return reviews_count, reviews_rating
        except:
            pass
