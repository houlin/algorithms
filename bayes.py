import pandas as pd
import numpy as np


class NBModel:
    'NBModel'
    testStr = 'This is NBModel.'

    def __init__(self, pc0, pc1, pVocabList, pc0VocabList, pc1VocabList):
        self.pc0 = pc0
        self.pc1 = pc1
        self.pVocabList = pVocabList
        self.pc0VocabList = pc0VocabList
        self.pc1VocabList = pc1VocabList

    def displayCount(self):
        print
        "NBModel %s" % NBModel.testStr


#
# def loadDataSet():
#     postingList = [['my', 'dog', 'has', 'flea', \
#                     'problems', 'help', 'please'],
#                    ['maybe', 'not', 'take', 'him', \
#                     'to', 'dog', 'park', 'stupid'],
#                    ['my', 'dalmation', 'is', 'so', 'cute', \
#                     'I', 'love', 'him'],
#                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
#                     'to', 'stop', 'him'],
#                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
#                    ]
#     classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 is not
#     return postingList, classVec


def loadDataSet2():
    import string
    factor = 0.2
    doc_list = ["Get the perfect shot with this $7 Aukey tripod stand for your phone ",
                "How we're continuing our support of @eji_org's mission to fight racial ",
                "This $258 Google Wi-Fi 3-pack means never worrying about a weak signal again https://www.",
                "Searching to help Mexico City, celebrating tennis history, and other top trends this week.",
                "Today's tech deals you don't want to miss include a sale on home and kitchen gadgets, a $16 laptop stand, and more ",
                "Ready, set, hut! With the #GoogleAssistant as your quarterback, keeping track of your favorite team is a snap",
                "Get this Sketch freebie - iPhone X Mockup designed",
                "Expand your Fire Tablet's memory with one of these great ",
                "Your feed, featuring you. See all the things that make you, you—now in the Google app.",
                "this podcast has leaks, hot takes, and rants and may not be suitable for all ages",
                "Samsung's great Galaxy Note 8 order freebies end Sept 24, so act fast!",
                "Wishing the north a cozy fall, and the south a colorful spring. ",
                "Thermostat E is just as good as its more expensive sibling — it's the perfect downgrade!",
                "Best New Android Games: This week, check out Data Wing, Push & Pop, and Stormbound: Kingdom Wars!",
                "Get this Sketch freebie - iOS 11 Grid Template",
                "If you're not happy with #Sprint, it may be time to switch. Here are a few reason you might wanna look elsewhere.",
                "Check out the new Fast and Furious mode in Anki Overdrive!",
                "Get this Sketch freebie - 60 Apple Device Icons designed by",
                "You can now race Fast and Furious characters in Anki Overdrive!",
                "Apple wins round one of legal battle with Qualcomm",
                "You’re so cute when you try to concentrate! Look at you trying to think.",
                "I can’t believe I love a stupid jerk.",
                "Aw, come on, can’t you take a joke?",
                "You should know how to please me by now.",
                "I hoped you were less experienced.",
                "Stop acting like a whore.",
                "You are going to nickel and dime me to death!",
                "In what world does buying that make sense?",
                "Fine. You handle your finances. Let me know when things go to hell.",
                "How dare you spread around our private business!",
                "Let me do the talking; people listen to men.",
                "You took a vow in front of God and everybody and I expect you to honor it!",
                "If you don’t train that dog I’m going to rub your nose in its mess.",
                "I will take our kids if you leave me.",
                "You’re scared?! This isn’t angry! You will KNOW when I’m ANGRY!",
                "Keep your stupid beliefs to yourself.",
                "God will find a way to get you back, and it ain’t gonna be pretty.",
                "I can feel myself being pulled into hell just listening to your nonsense!",
                "your partner admits they have a problem AND",
                "he or she acts on that statement by going to individual therapy AND"
                ]
    class_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                 ]  # 1 is abusive, 0 is not

    doc_words_list = []
    for doc in doc_list:
        lower_doc = doc.lower()
        sans_punc_lower_doc = lower_doc.translate(str.maketrans('', '', string.punctuation))
        doc_words_list.append(sans_punc_lower_doc.split(' '))

    # shuff
    df = pd.DataFrame({'X': doc_words_list, 'Y': class_vec})
    df = df.iloc[np.random.permutation(len(df))]

    # identify train dataset, test dataset size

    all_size = len(df)
    train_size = int(all_size * (1 - factor))
    test_size = all_size - train_size

    print("all_size len is %s" % all_size)
    print("train_size len is %s" % train_size)
    print("test_size len is %s" % test_size)

    X_train = df.iloc[0:train_size, 0]
    y_train = df.iloc[0:train_size, 1]
    X_test = df.iloc[train_size:, 0]
    y_test = df.iloc[train_size:, 1]
    return X_train, y_train, X_test, y_test


def trainNBModel(metricVecList, classList):
    vocabLength = len(metricVecList[0])
    docNum = len(classList)

    c0_num = 0
    c1_num = 0
    for i, c in enumerate(classList):
        if c == 0:
            c0_num = c0_num + 1
        if c == 1:
            c1_num = c1_num + 1

    pc0 = c0_num / docNum
    pc1 = c1_num / docNum

    pVocabList = [0.0] * vocabLength
    pc0VocabList = [0.0] * vocabLength
    pc1VocabList = [0.0] * vocabLength

    for i in list(range(vocabLength)):
        w_cur_num = 0
        w_c0_cur_num = 0
        w_c1_cur_num = 0
        c0_cur_num = 0
        c1_cur_num = 0

        for j, c in enumerate(classList):
            if c == 0:
                c0_cur_num = c0_cur_num + 1
            if c == 1:
                c1_cur_num = c1_cur_num + 1

            if metricVecList[j][i] == 1:
                w_cur_num = w_cur_num + 1
                if c == 0:
                    w_c0_cur_num = w_c0_cur_num + 1
                if c == 1:
                    w_c1_cur_num = w_c1_cur_num + 1

        pVocabList[i] = w_cur_num / docNum
        pc0VocabList[i] = w_c0_cur_num / c0_cur_num
        pc1VocabList[i] = w_c1_cur_num / c1_cur_num

    nbModel = NBModel(pc0, pc1, pVocabList, pc0VocabList, pc1VocabList)
    return nbModel


def predictData(nbModel, metricVecList):
    predictList = []
    for mv in metricVecList:
        pw = 1
        pw_c0 = 1
        pw_c1 = 1

        for i, v in enumerate(mv):
            if v:
                pw = pw * nbModel.pVocabList[i]
                pw_c0 = pw_c0 * nbModel.pc0VocabList[i]
                pw_c1 = pw_c1 * nbModel.pc1VocabList[i]

        pc0_w = pw_c0 * nbModel.pc0 / pw
        pc1_w = pw_c1 * nbModel.pc1 / pw
        if pc0_w > pc1_w:
            predictList.append(0)
        else:
            predictList.append(1)
    return predictList


def creatVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    vocabListLength = len(vocabList)
    returnVec = [0] * vocabListLength
    for w in inputSet:
        try:
            index = vocabList.index(w)
            if index >= 0 and index < vocabListLength:
                returnVec[index] = 1
        except:
            pass

    return returnVec
