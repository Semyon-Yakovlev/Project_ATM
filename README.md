# Project_ATM

Определние популярности геолокации для размещения банкомата

Для сборки проекта необходимо прописать команду docker compose up --build для сборки Docker контейнера.
После сборки контейнера будут доступны приложения:
1) Streamlit - http://localhost:8501
2) FastApi (запросы кэшируются в redis)
3) Telegram Bot

Ссылка на датасеты:
https://drive.google.com/drive/folders/1cpSOWXf-CPhT-gLRaoLVKJqLqe163mvq?usp=drive_link

Для данной задачи мы парсили данные из яндекс карт - спаршенные данные в файле output.json
Это данные по локациям банкоматов с их рейтингом и отзывами, адресом и районом в Москве.
1. ID - номер внутри района (от 1 до бесконечности)
2. name - наименование на яндекс картах (Строка, может быть как названием банка, так и 'Банкомат - "банк"')
3. address - полный адрес банкомата
4. districs - район москвы
5. website - чаще всего сайт банка
6. opening_hours - часы работы банкомата по дням
7. ypage - ссылка на данный банкомат в яндекс картах
8. rating - рейтинг банкомата, если он отображается (Если нет, среднее по reviews_rating)
9. reviews - отзывы в массиве
10. reviews_count - кол-во отзывов
11. reviews_rating - рейтинги по отзывам (от 1 до 5)

Второй датасет лежит в папке another dataset.
Он был найден по названию контеста от росбанка, на котором основана данная тема.

Текущая версия датасета содержит следующие поля:
1. lat - широта
2. long - долгота
3. Переменные, в которых указано количество торговых центров, банков, универмагов, станций метро, магазинов с алкоголем, полицейских участков, университетов, ЖД станций, аэропртов в радиусе 100 и 300 метров
4. population - население региона
5. density - плотность населения
6. salary - средняя заработная плата в регионе
7. OHE регионов
8. target - рейтинг банкомата (целевая переменная)


Plan:
1. Data parsing. +
2. Feature Engineering. +
3. Train ML models. +
4. Design and train DL models. -in progress
5. Create web application. +
