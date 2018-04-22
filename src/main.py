import data.ff as data
import models.train_model as model
import visualization.visualize as vis

if __name__ == "__main__":
    #load datasets
    fantasy2013 = data.load_data("2013_Fantasy")
    fantasy2014 = data.load_data("2014_Fantasy")
    fantasy2015 = data.load_data("2015_Fantasy")

    #Quarterbacks
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
    #vis.plot_pred_vs_actual(QB_elasticnet_preds, QBpoints2015, 'QB', ' Elastic Net')

    #Runningbacks
    RB_2013_2014_index = data.get_index_for_position(fantasy2013,fantasy2014, pos = 'RB')
    RB_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos = 'RB')

    RBstats2013, RBpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014,  RB_2013_2014_index, 'RB')
    RBstats2014, RBpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, RB_2014_2015_index, 'RB')

    RB_linear_model, RB_linear_preds, RB_linear_mse = model.linear_regression(RBstats2013, RBpoints2014, RBstats2014, RBpoints2015)
    RB_ridge_model, RB_ridge_preds, RB_ridge_mse = model.ridge_regression(RBstats2013, RBpoints2014, RBstats2014, RBpoints2015)
    RB_lasso_model, RB_lasso_preds, RB_lasso_mse = model.lasso_regression(RBstats2013, RBpoints2014, RBstats2014, RBpoints2015)
    RB_elasticnet_model, RB_elasticnet_preds, RB_elasticnet_mse = model.elasticnet_regression(RBstats2013, RBpoints2014, RBstats2014, RBpoints2015)
    RB_knn_model, RB_knn_preds, RB_knn_mse = model.knn(RBstats2013, RBpoints2014, RBstats2014, RBpoints2015)

    #Wide Recievers
    WR_2013_2014_index = data.get_index_for_position(fantasy2013, fantasy2014, pos='WR')
    WR_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos='WR')

    WRstats2013, WRpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014, WR_2013_2014_index, 'WR')
    WRstats2014, WRpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, WR_2014_2015_index, 'WR')

    WR_linear_model, WR_linear_preds, WR_linear_mse = model.linear_regression(WRstats2013, WRpoints2014, WRstats2014,WRpoints2015)
    WR_ridge_model, WR_ridge_preds, WR_ridge_mse = model.ridge_regression(WRstats2013, WRpoints2014, WRstats2014,WRpoints2015)
    WR_lasso_model, WR_lasso_preds, WR_lasso_mse = model.lasso_regression(WRstats2013, WRpoints2014, WRstats2014, WRpoints2015)
    WR_elasticnet_model, WR_elasticnet_preds, WR_elasticnet_mse = model.elasticnet_regression(WRstats2013, WRpoints2014,WRstats2014, WRpoints2015)
    WR_knn_model, WR_knn_preds, WR_knn_mse = model.knn(WRstats2013, WRpoints2014, WRstats2014, WRpoints2015)

    #Tight Ends
    TE_2013_2014_index = data.get_index_for_position(fantasy2013, fantasy2014, pos='TE')
    TE_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos='TE')

    TEstats2013, TEpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014, TE_2013_2014_index, 'TE')
    TEstats2014, TEpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, TE_2014_2015_index, 'TE')

    TE_linear_model, TE_linear_preds, TE_linear_mse = model.linear_regression(TEstats2013, TEpoints2014, TEstats2014,TEpoints2015)
    TE_ridge_model, TE_ridge_preds, TE_ridge_mse = model.ridge_regression(TEstats2013, TEpoints2014, TEstats2014, TEpoints2015)
    TE_lasso_model, TE_lasso_preds, TE_lasso_mse = model.lasso_regression(TEstats2013, TEpoints2014, TEstats2014, TEpoints2015)
    TE_elasticnet_model, TE_elasticnet_preds, TE_elasticnet_mse = model.elasticnet_regression(TEstats2013, TEpoints2014,TEstats2014, TEpoints2015)
    TE_knn_model, TE_knn_preds, TE_knn_mse = model.knn(TEstats2013, TEpoints2014, TEstats2014, TEpoints2015)