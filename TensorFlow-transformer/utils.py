#coding=utf-8
import jieba
import unicodedata
import sys,re,collections,nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#import nlgeval
class rule:
    # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("(?<=[a-zA-Z])\'s")  # 找出字母后面的字母
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")  # not的缩写
    pat_would = re.compile("(?<=[a-zA-Z])\'d")  # would的缩写
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")  # will的缩写
    pat_am = re.compile("(?<=[I|i])\'m")  # am的缩写
    pat_are = re.compile("(?<=[a-zA-Z])\'re")  # are的缩写
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")  # have的缩写

#delete the chinese sign backspace number
def del_chin_sign(str):
    str = str.strip()
    re_han_default = re.compile(u"([^\u4E00-\u9FD5])")
    str=re_han_default.sub('',str)
    str=str.strip()
    # str = re.sub(r'([。！？，“”「」：、 . ",!?])'.decode('utf-8'), r"", str)
    # str = re.sub(r'([0-9])'.decode('utf-8'), r"", str)
    #str = jieba.cut(str)
    return str


def replace_abbreviations(text):
    new_text = text
    new_text = rule.pat_letter.sub(' ', new_text).strip().lower()
    new_text = rule.pat_is.sub(r"\1 is", new_text)
    new_text = rule.pat_s.sub("", new_text)
    new_text = rule.pat_s2.sub("", new_text)
    new_text = rule.pat_not.sub(" not", new_text)
    new_text = rule.pat_would.sub(" would", new_text)
    new_text = rule.pat_will.sub(" will", new_text)
    new_text = rule.pat_am.sub(" am", new_text)
    new_text = rule.pat_are.sub(" are", new_text)
    new_text = rule.pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text

# pos和tag有相似的地方，通过tag获得pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

def merge(words):
    lmtzr = WordNetLemmatizer()
    new_words = ''
    words = nltk.pos_tag(word_tokenize(words))  # tag is like [('bigger', 'JJR')]
    for word in words:
        pos = get_wordnet_pos(word[1])
        if pos:
            # lemmatize()方法将word单词还原成pos词性的形式
            word = lmtzr.lemmatize(word[0], pos)
            new_words+=' '+word
        else:
            new_words+=' '+word[0]
    return new_words

def clear_data(text):
    text=replace_abbreviations(text)
    text=merge(text)
    text=text.strip()
    return text

def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text
