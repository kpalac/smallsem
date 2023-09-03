#!/usr/bin/python3
# -*- coding: utf-8 -*-

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#  Author: Karol Pałac (palac.karol@gmail.com)



"""
Sample language model (Russian) for developers
Change model data and run the script to generate header file, and then copy it to language models dir

"""

import pickle 
import snowballstemmer





# Modify data in the dictionary below
model = {
# List of possible names (e.g. if given in feed's meta info. First name on this list is a model name, so they need to be unique
# First item on the list is used as a model name in SmallSem
'names' : ('ru','RU','ru-*','Ru-*','RU-*', 'rus', 'Rus', 'RUS', 'Russian', 'russ', 'Русский', 'Рус', 'Русс'),

# Stemmer and pyphen modules
'stemmer' : 'russian', #Must be in snowballstemmer language list or None for no stemming
'pyphen' : 'ru_RU',


# This data is useful for language detection
'alphabet' : 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя',
'numerals' : '1234567890',
'capitals' : '', # This is useful to detect capitalized items but not essential
'vowels' : '',
'consonants' : '',
'unique_chars' : 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя',
'charsets' : ('ISO-8859-5','Cp1251','windows-1251'),

# Linguistic info
'writing_system' : 1, # 1 - alphabetic, 2 - logographic
'bicameral' : 1, # is language 'case-sensitive'?
'name_cap' : 1, # 1 - names capitalized, 2 - all nouns capitalized (e.g. German), 0 - no capitals (non-bicameral)

# Tokenizer
# It is advised to include patterns for emails, urls and other useful features for tokenisation
'REGEX_tokenizer' : '',

# Stop words and swadesh list (latter used extensively for language detection and keyword extraction when Xapian DB is not present)
'stops' : ("е", "и", "ж", "м", "о", "на", "не", "ни", "об", "но", "он", "мне", "мои", "мож", "она", "они", "оно", "мной", "много", "многочисленное", "многочисленная", "многочисленные", "многочисленный", "мною", "мой", "мог", "могут", "можно", "может", "можхо", "мор", "моя", "моё", "мочь", "над", "нее", "оба", "нам", "нем", "нами", "ними", "мимо", "немного", "одной", "одного", "менее", "однажды", "однако", "меня", "нему", "меньше", "ней", "наверху", "него", "ниже", "мало", "надо", "один", "одиннадцать", "одиннадцатый", "назад", "наиболее", "недавно", "миллионов", "недалеко", "между", "низко", "меля", "нельзя", "нибудь", "непрерывно", "наконец", "никогда", "никуда", "нас", "наш", "нет", "нею", "неё", "них", "мира", "наша", "наше", "наши", "ничего", "начала", "нередко", "несколько", "обычно", "опять", "около", "мы", "ну", "нх", "от", "отовсюду", "особенно", "нужно", "очень", "отсюда", "в", "во", "вон", "вниз", "внизу", "вокруг", "вот", "восемнадцать", "восемнадцатый", "восемь", "восьмой", "вверх", "вам", "вами", "важное", "важная", "важные", "важный", "вдали", "везде", "ведь", "вас", "ваш", "ваша", "ваше", "ваши", "впрочем", "весь", "вдруг", "вы", "все", "второй", "всем", "всеми", "времени", "время", "всему", "всего", "всегда", "всех", "всею", "всю", "вся", "всё", "всюду", "г       год", "говорил", "говорит", "года", "году", "где", "да", "ее", "за", "из", "ли", "же", "им", "до", "по", "ими", "под", "иногда", "довольно", "именно", "долго", "позже", "более", "должно", "пожалуйста", "значит", "иметь", "больше", "пока", "ему", "имя", "пор", "пора", "потом", "потому", "после", "почему", "почти", "посреди", "ей", "два", "две", "двенадцать", "двенадцатый", "двадцать", "двадцатый", "двух", "его", "дел", "или", "без", "день", "занят", "занята", "занято", "заняты", "действительно", "давно", "девятнадцать", "девятнадцатый", "девять", "девятый", "даже", "алло", "жизнь", "далеко", "близко", "здесь", "дальше", "для", "лет", "зато", "даром", "первый", "перед", "затем", "зачем", "лишь", "десять", "десятый", "ею", "её", "их", "бы", "еще", "при", "был", "про", "процентов", "против", "просто", "бывает", "бывь", "если", "люди", "была", "были", "было", "будем", "будет", "будете", "будешь", "прекрасно", "буду", "будь", "будто", "будут", "ещё", "пятнадцать", "пятнадцатый", "друго", "другое", "другой", "другие", "другая", "других", "есть", "пять", "быть", "лучше", "пятый", "к", "ком", "конечно", "кому", "кого", "когда", "которой", "которого", "которая", "которые", "который", "которых", "кем", "каждое", "каждая", "каждые", "каждый", "кажется", "как", "какой", "какая", "кто", "кроме", "куда", "кругом", "с     т", "у", "я", "та", "те", "уж", "со", "то", "том", "снова", "тому", "совсем", "того", "тогда", "тоже", "собой", "тобой", "собою", "тобою", "сначала", "только", "уметь", "тот", "тою", "хорошо", "хотеть", "хочешь", "хоть", "хотя", "свое", "свои", "твой", "своей", "своего", "своих", "свою", "твоя", "твоё", "раз", "уже", "сам", "там", "тем", "чем", "сама", "сами", "теми", "само", "рано", "самом", "самому", "самой", "самого", "семнадцать", "семнадцатый", "самим", "самими", "самих", "саму", "семь", "чему", "раньше", "сейчас", "чего", "сегодня", "себе", "тебе", "сеаой", "человек", "разве", "теперь", "себя", "тебя", "седьмой", "спасибо", "слишком", "так", "такое", "такой", "такие", "также", "такая", "сих", "тех", "чаще", "четвертый", "через", "часто", "шестой", "шестнадцать", "шестнадцатый", "шесть", "четыре", "четырнадцать", "четырнадцатый", "сколько", "сказал", "сказала", "сказать", "ту", "ты", "три", "эта", "эти", "что", "это", "чтоб", "этом", "этому", "этой", "этого", "чтобы", "этот", "стал", "туда", "этим", "этими", "рядом", "тринадцать", "тринадцатый", "этих", "третий", "тут", "эту", "суть", "чуть", "тысяч"), 
'swadesh' : ('я', 'ты', 'он', 'мы', 'вы', 'они', 'это', 'то', 'здесь, тут', 'там', 'кто', 'что', 'где', 'когда', 'как', 'не', 'все', 'многие', 'несколько', 'немногие', 'другой, иной', 'один', 'два', 'три', 'четыре', 'пять', 'большой', 'длинный', 'широкий', 'толстый', 'тяжёлый', 'маленький', 'короткий', 'узкий', 'тонкий', 'женщина', 'мужчина', 'человек', 'ребёнок, дитя', 'жена', 'муж', 'мать', 'отец', 'зверь, животное', 'рыба', 'птица', 'собака, пёс', 'вошь', 'змея', 'червь', 'дерево', 'лес', 'палка', 'плод', 'семя, семена', 'лист', 'корень', 'кора', 'цветок', 'трава', 'верёвка', 'кожа', 'мясо', 'кровь', 'кость', 'жир', 'яйцо', 'рог', 'хвост', 'перо', 'волосы', 'голова', 'ухо', 'глаз', 'нос', 'рот', 'зуб', 'язык', 'ноготь', 'стопа', 'нога', 'колено', 'рука', 'крыло', 'живот', 'внутренности, кишки', 'шея', 'спина', 'грудь', 'сердце', 'печень', 'пить', 'есть, кушать', 'грызть, кусать', 'сосать', 'плевать', 'рвать, блевать', 'дуть', 'дышать', 'смеяться', 'видеть', 'слышать', 'знать', 'думать', 'нюхать, чуять', 'бояться', 'спать', 'жить', 'умирать', 'убивать', 'бороться', 'охотиться', 'ударить', 'резать, рубить', 'разделить', 'кольнуть', 'царапать', 'копать, рыть', 'плавать', 'летать', 'ходить, идти', 'приходить, прийти', 'лежать (state)', 'сидеть (state)', 'стоять (state)', 'вращать, вертеть', 'падать', 'давать', 'держать', 'сжимать', 'тереть', 'мыть, умывать', 'вытирать', 'тянуть', 'толкать, пихать', 'бросать, кидать', 'вязать, связывать', 'шить', 'считать', 'говорить, сказать', 'петь', 'играть', 'плыть', 'течь', 'замёрзнуть', 'пухнуть', 'солнце', 'луна', 'звезда', 'вода', 'дождь', 'река', 'озеро', 'море', 'соль', 'камень', 'песок', 'пыль', 'земля', 'туча, облако', 'туман', 'небо', 'ветер', 'снег', 'лёд', 'дым', 'огонь', 'зола, пепел', 'жечь', 'дорога, путь', 'гора', 'красный', 'зелёный', 'жёлтый', 'белый', 'чёрный', 'ночь', 'день', 'год', 'тёплый', 'холодный', 'полный', 'новый', 'старый', 'хороший, добрый', 'злой, плохой', 'гнилой', 'грязный', 'прямой', 'круглый', 'острый', 'тупой', 'гладкий, ровный', 'мокрый', 'сухой', 'правильный', 'близкий', 'далёкий, дальний', 'правый', 'левый', 'при, у, возле', 'в', 'с', 'и', 'если', 'потому что', 'имя'),

'copulas' : (),


# Aliases will be expanded upon tokenization, case sensitive
'aliases' :          {},


# Morphology type and -fixes
'morphology' : 1,
'prefixes' : (),
'suffixes' : (),
'name_suffixes' : (),
'infixes' : (),



# List of common words for lookups (faster than dictionaries in most cases), easily found on the internet
'commons' : 
(
'я',
'не',
'что',
'в',
'и',
'ты',
'это',
'на',
'с',
'он',
'вы',
'как',
'мы',
'да',
'а',
'мне',
'меня',
'у',
'нет',
'так',
'но',
'то',
'все',
'тебя',
'его',
'за',
'о',
'она',
'тебе',
'если',
'они',
'бы',
'же',
'ну',
'здесь',
'к',
'из',
'есть',
'чтобы',
'для',
'хорошо',
'когда',
'вас',
'только',
'по',
'вот',
'просто',
'был',
'знаю',
'нас',
'всё',
'было',
'от',
'может',
'кто',
'вам',
'очень',
'их',
'там',
'будет',
'уже',
'почему',
'еще',
'быть',
'где',
'спасибо',
'ничего',
'сейчас',
'или',
'могу',
'хочу',
'нам',
'чем',
'мой',
'до',
'надо',
'этого',
'ее',
'теперь',
'давай',
'знаешь',
'нужно',
'больше',
'этом',
'нибудь',
'раз',
'со',
'была',
'этот',
'ему',
'ладно',
'эй',
'время',
'тоже',
'даже',
'хочешь',
'сказал',
'ли',
'себя',
'думаю',
'пока',
'должен',
'потому',
'никогда',
'ни',
'тут',
'ещё',
'её',
'пожалуйста',
'сюда',
'привет',
'тогда',
'конечно',
'моя',
'него',
'сегодня',
'один',
'тобой',
'правда',
'лучше',
'об',
'были',
'того',
'можно',
'мной',
'всегда',
'сказать',
'день',
'сэр',
'без',
'можешь',
'чего',
'эти',
'дело',
'значит',
'лет',
'много',
'во',
'делать',
'буду',
'порядке',
'должны',
'такой',
'ведь',
'ним',
'всего',
'сделать',
'хотел',
'твой',
'жизнь',
'ей',
'мистер',
'потом',
'через',
'себе',
'них',
'всех',
'такое',
'им',
'куда',
'том',
'мама',
'после',
'человек',
'люди',
'слишком',
'иди',
'зачем',
'этим',
'немного',
'сколько',
'этой',
'знаете',
'боже',
'ней',
'эту',
'который',
'отец',
'свою',
'деньги',
'два',
'под',
'твоя',
'мои',
'никто',
'моей',
'думаешь',
'друг',
'жизни',
'эта',
'назад',
'видел',
'кажется',
'точно',
'вместе',
'люблю',
'мог',
'случилось',
'сам',
'нравится',
'черт',
'какой',
'людей',
'папа',
'домой',
'тот',
'скажи',
'которые',
'должна',
'три',
'всем',
'сделал',
'возможно',
'прошу',
'будем',
'дома',
'парень',
'снова',
'говорит',
'место',
'отсюда',
'можем',
'будешь',
'пошли',
'делаешь',
'совсем',
'говорил',
'понимаю',
'завтра',
'хочет',
'простите',
'разве',
'давайте',
'хотите',
'отлично',
'сказала',
'туда',
'прямо',
'времени',
'вами',
'лишь',
'своей',
'хватит',
'думал',
'можете',
'дом',
'дела',
'знать',
'дай',
'понял',
'помочь',
'говорить',
'слушай',
'свои',
'поэтому',
'прости',
'знает',
'именно',
'знал',
'тем',
'кого',
'смотри',
'каждый',
'ваш',
'похоже',
'найти',
'моего',
'наш',
'мать',
'одна',
'имя',
'про',
'говорю',
'будут',
'оно',
'свой',
'нельзя',
'извините',
'стоит',
'действительно',
'зовут',
'поговорить',
'доктор',
'перед',
'несколько',
'нужен',
'происходит',
'ко',
'господи',
'возьми',
'мою',
'тех',
'нами',
'вижу',
'должно',
'наверное',
'откуда',
'понимаешь',
'верно',
'скоро',
'уж',
'деле',
'твои',
'пусть',
'всю',
'хотела',
'при',
'более',
'ребята',
'нее',
'быстро',
'подожди',
'идти',
'надеюсь',
'чём',
'работу',
'видеть',
'такая',
'этих',
'уверен',
'нужна',
'года',
'раньше',
'такие',
'руки',
'видишь',
'какая',
'посмотри',
'сын',
'самом',
'ваша',
'послушай',
'равно',
'наши',
'другой',
'ага',
'мир',
'извини',
'минут',
'против',
'твоей',
'пор',
'жить',
'ж',
'жаль',
'вообще',
'могли',
'хотя',
'человека',
'пора',
'ради',
'говорят',
'почти',
'твою',
'могут',
'над',
'весь',
'первый',
'чёрт',
'слышал',
'собой',
'брат',
'вещи',
'дня',
'скажу',
'говоришь',
'нормально',
'своего',
'мое',
'ваше',
'итак',
'будь',
'ночь',
'хоть',
'ясно',
'плохо',
'дверь',
'вопрос',
'господин',
'давно',
'денег',
'ваши',
'ка',
'мисс',
'одну',
'глаза',
'пять',
'будто',
'между',
'пойду',
'опять',
'работа',
'самое',
'иногда',
'детей',
'этому',
'рад',
'здорово',
'бог',
'одного',
'ночи',
'готов',
'номер',
'которая',
'машину',
'любовь',
'дорогая',
'виду',
'одно',
'прекрасно',
'вон',
'своих',
'быстрее',
'отца',
'женщина',
'достаточно',
'рядом',
'убить',
'таким',
'пойдем',
'смерти',
'дети',
'такого',
'правильно',
'месте',
'никаких',
'сказали',
'здравствуйте',
'пару',
'две',
'видела',
'долго',
'хороший',
'ах',
'кроме',
'алло',
'нашей',
'прав',
'вчера',
'вечером',
'жена',
'миссис',
'чтоб',
'друга',
'нужны',
'кем',
'какие',
'те',
'увидеть',
'утро',
'смогу',
'неё',
'сама',
'моему',
'большой',
'сразу',
'работать',
'сердце',
'стал',
'своим',
'сначала',
'могла',
'вроде',
'ними',
'говори',
'голову',
'дальше',
'помнишь',
'либо',
'ума',
'одной',
'вечер',
'случае',
'взять',
'проблемы',
'помощь',
'добрый',
'год',
'думала',
'делает',
'скорее',
'слова',
'капитан',
'последний',
'важно',
'дней',
'помню',
'ночью',
'утром',
'моих',
'произошло',
'которую',
'боюсь',
'также',
'вашей',
'ой',
'стой',
'твоего',
'никого',
'дорогой',
'убил',
'насчет',
'друзья',
'самый',
'проблема',
'видели',
'вперед',
'дерьмо',
'понятно',
'чувствую',
'наша',
'будете',
'тому',
'имею',
'вернуться',
'придется',
'пришел',
'спать',
'стать',
'столько',
'говорила',
'пойти',
'иначе',
'работает',
'девушка',
'час',
'момент',
'моим',
'умер',
'думаете',
'доброе',
'слово',
'новый',
'часов',
'мире',
'знаем',
'твое',
'мальчик',
'однажды',
'интересно',
'конец',
'играть',
'a',
'заткнись',
'сделали',
'посмотреть',
'идет',
'узнать',
'свое',
'права',
'хорошая',
'город',
'джон',
'долларов',
'парни',
'идем',
'говорите',
'уйти',
'понять',
'знала',
'поздно',
'нашли',
'работы',
'скажите',
'сделаю',
'увидимся',
'какого',
'другие',
'идея',
'пошел',
'доме',
'дочь',
'имеет',
'приятно',
'лицо',
'наших',
'обо',
'понимаете',
'руку',
'часть',
'смотрите',
'вся',
'собираюсь',
'четыре',
'прежде',
'хотят',
'скажешь',
'чувак',
'дайте',
'сделала',
'кофе',
'джек',
'верю',
'ждать',
'затем',
'большое',
'сами',
'неужели',
'моё',
'любит',
'мужчина',
'дать',
'господа',
'таких',
'осталось',
'которой',
'далеко',
'вернусь',
'сильно',
'ох',
'сможешь',
'кому',
'вашего',
'посмотрим',
'машина',
'подождите',
'свет',
'чуть',
'серьезно',
'пришли',
'оружие',
'решил',
'смысле',
'видите',
'тихо',
'нашел',
'свидания',
'путь',
'той',
'совершенно',
'следующий',
'которого',
'места',
'парня',
'вдруг',
'пути',
'мадам',
'какое',
'шанс',
'сестра',
'нашего',
'ужасно',
'минуту',
'вокруг',
'другом',
'иду',
'других',
'хотели',
'нем',
'смерть',
'подумал',
'фильм',
'оставь',
'делаете',
'уверена',
'кровь',
'говорили',
'внимание',
'помогите',
'идите',
'держи',
'получить',
'оба',
'взял',
'спокойно',
'обычно',
'мало',
'забыл',
'странно',
'смотреть',
'поехали',
'дал',
'часа',
'прекрати',
'посмотрите',
'готовы',
'вернулся',
'поверить',
'позже',
'милая',
'женщины',
'любишь',
'довольно',
'обратно',
'остаться',
'думать',
'та',
'стороны',
'полиция',
'тело',
'тысяч',
'делал',
'машины',
'угодно',
'муж',
'году',
'неплохо',
'бога',
'некоторые',
'конце',
'милый',
'the',
'рождения',
'трудно',
'добро',
'любви',
'больно',
'невозможно',
'спокойной',
'слышишь',
'типа',
'получил',
'которое',
'приятель',
'хуже',
'никому',
'честь',
'успокойся',
'вашу',
'маленький',
'выглядит',
'чарли',
'сына',
'неделю',
'i',
'девочка',
'делаю',
'шесть',
'ноги',
'история',
'рассказать',
'послушайте',
'часто',
'кстати',
'двух',
'забудь',
'которых',
'следует',
'знают',
'пришла',
'семья',
'станет',
'матери',
'ребенок',
'план',
'проблем',
'например',
'сделай',
'воды',
'немедленно',
'мира',
'сэм',
'телефон',
'перестань',
'правду',
'второй',
'прощения',
'ту',
'наше',
'уходи',
'твоих',
'помоги',
'пол',
'внутри',
'нему',
'смог',
'десять',
'нашу',
'около',
'бывает',
'самого',
'большая',
'леди',
'сможем',
'вниз',
'легко',
'делай',
'единственный',
'рада',
'меньше',
'волнуйся',
'хотим',
'полагаю',
'мам',
'иметь',
'своими',
'мере',
'наконец',
'начала',
'минутку',
'работе',
'пожаловать',
'другого',
'двое',
'никакого',
'честно',
'школе',
'лучший',
'умереть',
'дам',
'насколько',
'всей',
'малыш',
'оставить',
'безопасности',
'ненавижу',
'школу',
'осторожно',
'сынок',
'джо',
'таки',
'пытался',
'другое',
'б',
'клянусь',
'машине',
'недели',
'стало',
'истории',
'пришлось',
'выглядишь',
'чему',
'сможет',
'купить',
'слышала',
'знали',
'настоящий',
'сих',
'выйти',
'людям',
'замечательно',
'полиции',
'огонь',
'пойдём',
'спросить',
'дядя',
'детка',
'среди',
'особенно',
'твоим',
'комнате',
'шоу',
'выпить',
'постоянно',
'делают',
'позвольте',
'родители',
'письмо',
'городе',
'случай',
'месяцев',
'мужик',
'благодарю',
'o',
'ребенка',
'смешно',
'ответ',
'города',
'образом',
'любой',
'полностью',
'увидел',
'еду',
'имени',
'вместо',
'абсолютно',
'обязательно',
'улице',
'твоё',
'убили',
'ваших',
'ехать',
'крови',
'решение',
'вина',
'поможет',
'своё',
'секунду',
'обещаю',
'начать',
'голос',
'вещь',
'друзей',
'показать',
'нечего',
'э',
'месяц',
'подарок',
'приехал',
'самая',
'молодец',
'сделаем',
'крайней',
'женщин',
'собираешься',
'конца',
'страшно',
'новости',
'идиот',
'потерял',
'спасти',
'вернуть',
'узнал',
'слушайте',
'хотелось',
'сон',
'поняла',
'прошло',
'комнату',
'семь',
'погоди',
'главное',
'рано',
'корабль',
'пытаюсь',
'игра',
'умерла',
'повезло',
'всему',
'возьму',
'таком',
'моем',
'глаз',
'настолько',
'идём',
'удачи',
'готова',
'семьи',
'садись',
'гарри',
'держись',
'звучит',
'мило',
'война',
'человеком',
'право',
'такую',
'вопросы',
'представить',
'работаю',
'имеешь',
'красивая',
'идёт',
'никакой',
'профессор',
'думает',
'войны',
'стала',
'стали',
'оттуда',
'известно',
'слышу',
'начал',
'подумать',
'позвонить',
'старый',
'придётся',
'историю',
'вести',
'твоему',
'последнее',
'хочется',
'миллионов',
'нашла',
'способ',
'отношения',
'земле',
'фрэнк',
'получится',
'говоря',
'связи',
'многие',
'пошёл',
'пистолет',
'убью',
'руках',
'получилось',
'президент',
'остановить',
'тьi',
'оставил',
'одним',
'you',
'утра',
'боль',
'хорошие',
'пришёл',
'открой',
'брось',
'вставай',
'находится',
'поговорим',
'кино',
'людьми',
'полицию',
'покажу',
'волосы',
'последние',
'брата',
'месяца',
'круто',
'игры',
'сторону',
'неважно',
'другим',
'вид',
'искать',
'сидеть',
'познакомиться',
'встречи',
'забрать',
'придет',
'покое',
'ха',
'чувствуешь',
'оставьте',
'слышали',
'маленькая',
'ублюдок',
'землю',
'король',
'существует',
'девочки',
'богу',
'р',
'беги',
'ок',
'расскажи',
'рот',
'своему',
'стоять',
'впервые',
'сообщение',
'принять',
'становится',
'живет',
'работал',
'любом',
'однако',
'слава',
'возможность',
'встретиться',
'жив',
'секунд',
'гораздо',
'будьте',
'попробуй',
'объяснить',
'навсегда',
'замуж',
'похож',
'прошлой',
'насчёт',
'никуда',
'дороги',
'здравствуй',
'блин',
'пройти',
'господь',
'минуты',
'пап',
'словно',
'ждет',
'силы',
'мужчины',
'означает',
'земли',
'ух',
'пить',
'т',
'неделе',
'использовать',
'счет',
'держать',
'выбор',
'согласен',
'месье',
'дэвид',
'является',
'задницу',
'наверно',
'джимми',
'невероятно',
'будущее',
'музыка',
'покажи',
'нe',
'радио',
'пошла',
'компании',
'хватает',
'речь',
'любил',
'перевод',
'каким',
'которым',
'остальные',
'го',
'намного',
'понравится',
'новые',
'вперёд',
'помните',
'м',
'решили',
'самой',
'восемь',
'чувство',
'считаю',
'стоп',
'вода',
'мужа',
'девушки',
'всеми',
'школы',
'увидишь',
'вашим',
'называется',
'окей',
'едем',
'двери',
'видно',
'взяли',
'нём',
'отцом',
'босс',
'дни',
'побери',
'дорогу',
'никак',
'большие',
'первым',
'голова',
'найду',
'предложение',
'случится',
'to',
'прошлом',
'новая',
'ужин',
'тяжело',
'имел',
'своем',
'первая',
'воду',
'хорошее',
'майкл',
'отпусти',
'ходить',
'получается',
'мэм',
'жду',
'наверняка',
'серьёзно',
'убийство',
'алекс',
'другую',
'учитель',
'убийца',
'любить',
'н',
'другу',
'жену',
'просил',
'заниматься',
'макс',
'й',
'мысли',
'каждого',
'поводу',
'сукин',
'ушел',
'старик',
'свете',
'солнце',
'позвоню',
'стороне',
'народ',
'нравятся',
'одном',
'поздравляю',
'ключ',
'прийти',
'майк',
'дамы',
'внизу',
'оставаться',
'концов',
'джордж',
'общем',
'виноват',
'вовсе',
'стойте',
'такси',
'уверены',
'генерал',
'начали',
'мистера',
'красиво',
'идут',
'джонни',
'and',
'разница',
'уехать',
'море',
'черта',
'котором',
'госпожа',
'блядь',
'сара',
'порядок',
'очередь',
'позволь',
'голове',
'понятия',
'днем',
'ушла',
'касается',
'отдай',
'разумеется',
'приехали',
'нашёл',
'купил',
'раза',
'части',
'двадцать',
'причина',
'звонил',
'какую',
'сука',
'помощи',
'нашим',
'проверить',
'этими',
'удалось',
'необходимо',
'наверху',
'целый',
'течение',
'верить',
'вероятно',
'вечно',
'молодой',
'поезд',
'вполне',
'годы',
'нового',
'другому',
'ничто',
'поехать',
'тюрьме',
'дороге',
'остался',
'придурок',
'рассказал',
'хорошего',
'близко',
'вашем',
'случайно',
'попробовать',
'поскольку',
'написал',
'язык',
'тысячи',
'цель',
'бабушка',
'слышать',
'постой',
'вверх',
'связь',
'книги',
'выше',
'чувства',
'женой',
'играет',
'девушку',
'анна',
'делали',
'готово',
'счастлив',
'мертв',
'договорились',
'питер',
'пара',
'позволить',
'попал',
'останется',
'удар',
'живу',
'смотрю',
'ник',
'забыть',
'приказ',
'доктора',
'понадобится',
'открыть',
'женщину',
'сержант',
'сша',
'ключи',
'дали',
'понравилось',
'сожалею',
'каком',
'могло',
'джеймс',
'нашем',
'делаем',
'сделает',
'выйдет',
'подумай',
'игру',
'одни',
'маме',
'стране',
'выходит',
'сил',
'тела',
'помощью',
'ближе',
'занят',
'возьмите',
'изменить',
'очевидно',
'ввиду',
'ждут',
'смысл',
'трубку',
'новое',
'лично',
'друзьями',
'слушать',
'платье',
'благодаря',
'бери',
'убирайся',
'убивать',
'ищу',
'состоянии',
'поверь',
'первое',
'устал',
'написано',
'менее',
'самые',
'провести',
'думают',
'стрелять',
'боб',
'забыла',
'опасно',
'отличная',
'книгу',
'ухожу',
'группа',
'тише',
'новую',
'часы',
'пришло',
'подумала',
'сделаешь',
'выход',
'обед',
'сможете',
'стану',
'вышел',
'страны',
'лица',
'комната',
'магазин',
'расскажу',
'команда',
'жизнью',
'билли',
'решила',
'головы',
'обещал',
'недавно',
'отличный',
'боишься',
'главный',
'вашему',
'нечто',
'слов',
'лейтенант',
'солдат',
'семью',
'сто',
'чё',
'попасть',
'адрес',
'получили',
'внимания',
'г',
'эдди',
'дурак',
'помогу',
'вопросов',
'примерно',
'собака',
'продолжай',
'беспокойся',
'полный',
'поняли',
'х',
'правила',
'света',
'энди',
'суд',
'жены',
'музыку',
'первой',
'весело',
'каждую',
'сильнее',
'собирался',
'член',
'веришь',
'бойся',
'времена',
'полковник',
'писать',
'услышать',
'земля',
'наверх',
'сложно',
'чушь',
'память',
'холодно',
'подальше',
'этo',
'бен',
'двигаться',
'поле',
'семье',
'везде',
'глупо',
'твоем',
'роль',
'е',
'любят',
'сюрприз',
'делу',
'читать',
'плевать',
'прочь',
'билл',
'собирается',
'окно',
'мужчин',
'прекрасный',
'садитесь',
'позвони',
'идешь',
'звонок',
'яйца',
'класс',
'джейн',
'вещей',
'сан',
'одному',
'скажем',
'хозяин',
'вернись',
'начинается',
'полно',
'ошибка',
'весьма',
'генри',
'рождество',
'получишь',
'черту',
'встретимся',
'разговор',
'жил',
'совет',
'де',
'закон',
'кончено',
'чудо',
'большинство',
'продолжать',
'фотографии',
'сердца',
'телефону',
'скажет',
'лучшее',
'тони',
'спросил',
'пожалуй',
'мысль',
'дочери',
'родителей',
'агент',
'карл',
'зато',
'души',
'убери',
'недель',
'кольцо',
'поеду',
'вернулась',
'другая',
'пойдет',
'войти',
'спустя',
'тюрьму',
'бросил',
'единственная',
'джим',
'чтo',
'подождать',
'сигнал',
'мсье',
'прощай',
'детектив',
'дождь',
'выбора',
'любовью',
'слушаю',
'встреча',
'первого',
'думали',
'поцелуй',
'скажете',
'парней',
'дэнни',
'желание',
'уходите',
'платить',
'книга',
'снаружи',
'бизнес',
'вышла',
'приходит',
'поедем',
'вернется',
'маму',
'песня',
'дел',
'разговаривать',
'такими',
'подруга',
'удовольствием',
'секс',
'взяла',
'больницу',
'момента',
'парнем',
'бобби',
'настоящая',
'ребёнок',
'нос',
'считаешь',
'плохой',
'шаг',
'счастлива',
'забавно',
'правы',
'иисус',
'единственное',
'отцу',
'заметил',
'легче',
'выходи',
'мария',
'имеете',
'самых',
'родился',
'спит',
'документы',
'думай',
'белый',
'большую',
'полицейский',
'вечера',
'неправильно',
'курсе',
'дедушка',
'список',
'настоящее',
'следующей',
'снять',
'бежать',
'шутка',
'шеф',
'уходить',
'несмотря',
'мальчика',
'ждал',
'держите',
'новой',
'смотрел',
'джей',
'ешь',
'армии',
'запах',
'сожалению',
'бросить',
'ти',
'слышите',
'считает',
'закрой',
'найди',
'подходит',
'уходим',
'хорош',
'марк',
'мнение',
'детьми',
'душу',
'джейк',
'сила',
'планы',
'секрет',
'самолет',
'срочно',
'заходи',
'убийства',
'внутрь',
'обоих',
'церкви',
'рука',
'великолепно',
'переведено',
'джентльмены',
'пытается',
'офицер',
'танцевать',
'заняться',
'одежду',
'цветы',
'стоило',
'повсюду',
'делала',
'девять',
'войну',
'закончить',
'свидание',
'принеси',
'проклятье',
'принести',
'красивый',
'попросить',
'мэри',
'звук',
'стол',
'заставить',
'каких',
'всём',
'отдать',
'долг',
'туалет',
'счастье',
'стыдно',
'приходится',
'центр',
'вряд',
'уехал',
'казалось',
'сядь',
'закончил',
'вышло',
'прекратите',
'квартиру',
'фбр',
'остается',
'мимо',
'встречу',
'группы',
'дон',
'шутишь',
'попросил',
'зависит',
'врач',
'силу',
'приду',
'медленно',
'власти',
'чай',
'смогла',
'штука',
'бой',
'самым',
'ждёт',
'неприятности',
'песню',
'другими',
'умею',
'судьба',
'увижу',
'потеряли',
'взглянуть',
'неправда',
'знакомы',
'сумасшедший',
'многое',
'великий',
'настоящему',
'трогай',
'играл',
'скорей',
'люси',
'женщиной',
'передать',
'спас',
'йорк',
'найдем',
'еда',
'разу',
'ого',
'опасности',
'знак',
'узнает',
'товарищ',
'ребёнка',
'руками',
'хм',
'глазами',
'бежим',
'убей',
'остальное',
'работаешь',
'столь',
'сомневаюсь',
'добраться',
'больнице',
'позвонил',
'принцесса',
'получше',
'откройте',
'душа',
'дружище',
'правительство',
'мамой',
'система',
'видимо',
'принесу',
'захочешь',
'третий',
'первых',
'умеешь',
'пиво',
'узнали',
'желаю',
'взгляд',
'начнем',
'чисто',
'глазах',
'док',
'дала',
'расслабься',
'впереди',
'остались',
'вовремя',
'здоровье',
'вечеринку',
'мозги',
'кусок',
'воздух',
'мяч',
'церковь',
'принес',
'ребят',
'ногу',
'начало',
'томми',
'обожаю',
'смотришь',
'нож',
'c',
'письма',
'мамы',
'даст',
'любила',
'адвокат',
'ситуация',
'украл',
'смогли',
'посмотрю',
'последняя',
'вкус',
'мозг',
'командир',
'короля',
'головой',
'святой',
'билет',
'болит',
'рук',
'постели',
'быстрей',
'клуб',
'ветер',
'пойдешь',
'зря',
'величество',
'приехала',
'старая',
'it',
'живут',
'чувствовать',
'старые',
'лучшие',
'возможности',
'уничтожить',
'прием',
'едва',
'информацию',
'кое',
'чарльз',
'прежнему',
'делом',
'мартин',
'вечеринка',
'баксов',
'окна',
'даю',
'послал',
'миль',
'представляешь',
'маленькие',
'искал',
'дух',
'многих',
'куча',
'похожи',
'умру',
'оказался',
'звали',
'братья',
'выпьем',
'видит',
'прекрасная',
'моём',
'банк',
'собираетесь',
'называют',
'придёт',
'упал',
'гости',
'отпустите',
'останусь',
'удовольствие',
'появился',
'уходит',
'разные',
'герой',
'директор',
'of',
'петь',
'живо',
'пыталась',
'начинает',
'узнала',
'страшного',
'хорошим',
'жене',
'продолжайте',
'ушли',
'волнуйтесь',
'джерри',
'любимый',
'написать',
'потрясающе',
'вновь',
'доверять',
'держу',
'жениться',
'меч',
'отель',
'миллион',
'чудесно',
'положи',
'брать',
'сзади',
'вкусно',
'власть',
'умоляю',
'хрена',
'дерьма',
'дает',
'спал',
'говорим',
'офис',
'уши',
'состояние',
'возвращайся',
'ад',
'ответить',
'недостаточно',
'отвали',
'ложь',
'чертовски',
'вернулись',
'закончится',
'принадлежит',
'аминь',
'волнует',
'прекратить',
'возле',
'костюм',
'подойди',
'улицу',
'заплатить',
'сказано',
'сидит',
'пива',
'читал',
'началось',
'получила',
'oн',
'мальчики',
'тип',
'предложить',
'способен',
'продать',
'живой',
'вне',
'встретил',
'станешь',
'вернемся',
'лицом',
'погиб',
'маленькой',
'артур',
'помогать',
'встать',
'карты',
'живёт',
'фото',
'человеку',
'рэй',
'страну',
'трех',
'животных',
'камеру',
'трое',
'кейт',
'бля',
'увидела',
'зайти',
'учиться',
'устала',
'звонит',
'сотни',
'страх',
'сделано',
'похожа',
'достать',
'хрен',
'богом',
'мечты',
'спрашиваю',
'избавиться',
'судья',
'смотрит',
'автобус',
'качестве',
'страна',
'девушкой',
'работают',
'компания',
'двумя',
'собираемся',
'миру',
'сообщить',
'com',
'увидите',
'душе',
'помог',
'небольшой',
'центре',
'президента',
'защитить',
'задание',
'шлюха',
'матерью',
'удивительно',
'сестры',
'школа',
'чувствовал',
'едет',
'доказать',
'ушёл',
'классно',
'положение',
'плохая',
'лучшая',
'считать',
'двоих',
'оставлю',
'спроси',
'фунтов',
'угу',
'лео',
'внимательно',
'телевизор',
'команды',
'потеряла'
)

}


# Stem commons
if model.get('stemmer') != None:
    stemmer = snowballstemmer.stemmer(model.get('stemmer','english'))
    commons_stemmed = []
    for w in model.get('commons',''):
        w = stemmer.stemWord(w)
        commons_stemmed.append(w)
    model['commons_stemmed'] = tuple(commons_stemmed)

# Generate pickle file
filename = model['names'][0] + '_model.pkl'

with open(filename, "wb") as write_file:
    pickle.dump(model, write_file)

