# Putoshka x НМИЦ (Сервис классификации кортикограмм)
**Описание проекта:**

Мы представляем Desktop-приложение для автоматической разметки и классификации электрокортикограмм (ЭКоГ), используемых в доклинических исследованиях. Приложение на основе искусственного интеллекта определяет эпилептическую активность и ключевые фазы сна, обеспечивая точный и автоматизированный анализ данных ЭКоГ.

**Стек:** PyEDFlib, Scikit-learn, TensorFlow, Keras, NumPy, PyQT.

## [Screencast](-)
## [Desktop application](-)

## Команда
  
  Игнатенок Филипп (Frontend)

  Данила Ильин (Backend)
  
  Назарова Полина (ML-Специалист)
  
  Илья Богданов (Product Manager)

# Запуск

  1. Скачать exe-файл по ссылке: https://disk.yandex.ru/d/tbq_oAmPsUkW_A
  2. Запустить установленный файлик


---

# Архитектура
В данном проекте реализовано Desktop-приложение для автоматической классификации электрокортикограмм, а также генерации отчетов о работе модели ИИ. В качестве выходных файлов мы представляем: размеченный EDF-файл (электрокортикограмма), PDF-файл (отчет о работе ИИ), txt-файл (файлик с координатами меток).

## [Frontend + Backend]

Пользовательский интерфейс реализован на PyQT и обеспечивает полный контроль как при исследовании электрокортикограмм, так и при редактировании уже размеченных EDF-файлов. 

## [ML]

Приложение анализирует данные о сне, чтобы определить фазы промежуточного и глубокого сна, испытание эпилептического принципа или какие-то необозначенные аномалии. Оно использует четыре алгоритма, каждый из которых специализируется на определении одного из этих состояний. Алгоритмы анализируют небольшие части данных и выдают результат, который затем объединяется в единый вывод с учетом настроек пользователя (EDF, PDF, TXT файлы).