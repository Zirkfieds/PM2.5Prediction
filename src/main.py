from src.MachineLearning.model_test import model_test
from src.MachineLearning.model_train import model_train

if __name__ == '__main__':
    print('PM 2.5 Prediction')

    step = 1

    model = model_train('../dataset/data.xls', 1)
    clf = model.train_model()
    model.self_evaluation()

    # load the data from 4 random test points
    for i in range(0, 5):
        test = model_test('../dataset/data_pred_' + str(i) + '.xls', 6)
        test.test_model(clf)

    # single prediction test for any step X
    test = model_test('../dataset/data_pred_0.xls', 6)
    test.single_test(clf, 6,
                     [[
                         19.3,
                         19.1,
                         19.0,
                         20.1,
                         19.2,
                         18.6
                     ]])
    '''
    19.3550761851852,
    19.1471226666667,
    19.4044148113208,
    19.5288545925926,
    19.2584365,
    18.6199421296296
    '''
