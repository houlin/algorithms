import bayes

if True:
    X_train, y_train, X_test, y_test = bayes.loadDataSet()
    myVocabList = bayes.creatVocabList(X_train)

    trainVecList = []
    for doc in X_train:
        docVec = bayes.setOfWords2Vec(myVocabList, doc)
        trainVecList.append(docVec)

    nbModel = bayes.trainNBModel(trainVecList, y_train)

    testVecList = []
    for doc in X_test:
        docVec = bayes.setOfWords2Vec(myVocabList, doc)
        testVecList.append(docVec)

    predictList = bayes.predictData(nbModel, testVecList)
    print("predictList is ")
    print(predictList)

    print("testList is ")
    print(y_test.values)

    print("score is :")
    print(bayes.score(predictList, y_test.values))

