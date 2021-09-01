from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


def run_logistic_regression(x, y):
    logistic_reg_model = LogisticRegression(max_iter=10000)
    rfe = RFE(logistic_reg_model, n_features_to_select=10)
    rfe = rfe.fit(x, y)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)

    # Fit model
    # logistic_reg_model.fit(train_x, train_y)

    print(logistic_reg_model.summary())

    return logistic_reg_model
