import bayes

if True:
    X_train, y_train, X_test, y_test = bayes.loadDataSet2()
    myVocabList = bayes.creatVocabList(X_train)
    # print(len(myVocabList))
    metricVecList = []
    for doc in X_train:
        metricVec = bayes.setOfWords2Vec(myVocabList, doc)
        metricVecList.append(metricVec)
    # print(len(metricVecList))
    nbModel = bayes.trainNBModel(metricVecList, y_train)
    # testListOPost = []
    testMetricVecList = []
    for doc in X_test:
        metricVec = bayes.setOfWords2Vec(myVocabList, doc)
        testMetricVecList.append(metricVec)
    predictList = bayes.predictData(nbModel, testMetricVecList)
    print("predictList is ")
    print(predictList)

    print("testList is ")
    print(y_test)


