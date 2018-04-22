import data.ff as data
import models.train_model as model
import visualization.visualize as vis

if __name__ == "__main__":
    fantasy2013 = data.load_data("2013_Fantasy")
    fantasy2014 = data.load_data("2014_Fantasy")
    fantasy2015 = data.load_data("2015_Fantasy")

    QB_2013_2014_index = data.get_index_for_position(fantasy2013,fantasy2014)
    QB_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015)

    QBstats2013, QBpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014,  QB_2013_2014_index, 'QB')
    QBstats2014, QBpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, QB_2014_2015_index, 'QB')

    QB_linear_model, QB_linear_preds, QB_linear_mse = model.linear_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_ridge_model, QB_ridge_preds, QB_ridge_mse = model.ridge_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_lasso_model, QB_lasso_preds, QB_lasso_mse = model.lasso_regression(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)
    QB_elasticnet_model, QB_elasticnet_preds, QB_elasticnet_mse = model.elasticnet_regression(QBstats2013, QBpoints2014, QBstats2014,                                                          QBpoints2015)
    QB_knn_model, QB_knn_preds, QB_knn_mse = model.knn(QBstats2013, QBpoints2014, QBstats2014, QBpoints2015)

    #vis.plot_regression_coefs(QB_linear_model.coef_, QB_ridge_model.coef_, QB_lasso_model.coef_, QB_elasticnet_model.coef_, 'QB')
    vis.plot_pred_vs_actual(QB_elasticnet_preds, QBpoints2015, 'QB', ' Elastic Net')