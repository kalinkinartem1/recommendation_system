# Рекомендательная система постов для пользователей

## Описание проекта

Данный проект представляет собой рекомендательную систему для постов пользователей. В работе использован контентный подход для рекомендации постов юзерам. Для этого были использованы следующие библиотеки 
- **pandas**
- **numpy**
- **scikit-learn**
- **catboost**
- **nltk**
- **sqlalchemy**
- **fastapi**

## Основные компоненты проекта

В проекте используются следующие таблицы:

1. Таблица с информацией о пользователях.
2. Таблица с информацией о постах.
3. Таблица с информацией о взаимодействии пользователя с постами.

На основе этих таблиц были выделены дополнительные признаки для обучения моделей.
Первоначалные таблицы и измененные можно скачать [здесь](https://drive.google.com/file/d/1aZAJr18KRblh9guoCxc8DNaIyXp-mXOm/view?usp=sharing)


## Цель проекта
Достижение качества **hitrate@5=0.57**. 

## Достижения
После обучения моделей рекомендательной системы удалось достичь качества **hitrate@5=0.607**. 

Как запустить программу для запуска программы необходимо запустить 

