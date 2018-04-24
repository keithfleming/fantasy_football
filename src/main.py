# import data.ff as data
import fantasy_football.src.data.ff as data
import models.train_model as model
import pandas as pd
import visualization.visualize as vis

if __name__ == "__main__":
    #load datasets
    fantasy2012 = data.load_data("2012_Fantasy")
    fantasy2013 = data.load_data("2013_Fantasy")
    fantasy2014 = data.load_data("2014_Fantasy")
    fantasy2015 = data.load_data("2015_Fantasy")

    #Quarterbacks
    QB_2012_2013_index = data.get_index_for_position(fantasy2012, fantasy2013)
    QB_2013_2014_index = data.get_index_for_position(fantasy2013,fantasy2014)
    QB_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015)

    QBstats2012, QBpoints2013 = data.prepare_input_data(fantasy2012, fantasy2013, QB_2012_2013_index, 'QB')
    QBstats2013, QBpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014,  QB_2013_2014_index, 'QB')
    QBstats2014, QBpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, QB_2014_2015_index, 'QB')

    #QBstats2012 = data.df_norm(QBstats2012)
    #QBstats2013 = data.df_norm(QBstats2013)
    #QBstats2014  =data.df_norm(QBstats2014)

    QB_training_stats = pd.concat([QBstats2012, QBstats2013])
    QB_training_points = pd.concat([QBpoints2013, QBpoints2014])

    QB_linear_model, QB_linear_preds, QB_linear_mse = model.linear_regression(QB_training_stats, QB_training_points, QBstats2014, QBpoints2015)
    QB_ridge_model, QB_ridge_preds, QB_ridge_mse = model.ridge_regression(QB_training_stats, QB_training_points, QBstats2014, QBpoints2015)
    QB_lasso_model, QB_lasso_preds, QB_lasso_mse = model.lasso_regression(QB_training_stats, QB_training_points, QBstats2014, QBpoints2015)
    QB_elasticnet_model, QB_elasticnet_preds, QB_elasticnet_mse = model.elasticnet_regression(QB_training_stats, QB_training_points, QBstats2014, QBpoints2015)
    QB_knn_model, QB_knn_preds, QB_knn_mse = model.knn(QB_training_stats, QB_training_points, QBstats2014, QBpoints2015)

    #vis.plot_regression_coefs(QB_linear_model.coef_, QB_ridge_model.coef_, QB_lasso_model.coef_, QB_elasticnet_model.coef_, 'QB')
    vis.plot_pred_vs_actual(QB_knn_preds, QBpoints2015, 'QB', 'KNN')

    #Runningbacks
    RB_2012_2013_index = data.get_index_for_position(fantasy2012, fantasy2013, pos='RB')
    RB_2013_2014_index = data.get_index_for_position(fantasy2013,fantasy2014, pos = 'RB')
    RB_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos = 'RB')

    RBstats2012, RBpoints2013 = data.prepare_input_data(fantasy2012, fantasy2013, RB_2012_2013_index, 'RB')
    RBstats2013, RBpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014,  RB_2013_2014_index, 'RB')
    RBstats2014, RBpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, RB_2014_2015_index, 'RB')

    #RBstats2012 = data.df_norm(RBstats2012)
    #RBstats2013 = data.df_norm(RBstats2013)
    #RBstats2014  =data.df_norm(RBstats2014)

    RB_training_stats = pd.concat([RBstats2012, RBstats2013])
    RB_training_points = pd.concat([RBpoints2013, RBpoints2014])

    RB_linear_model, RB_linear_preds, RB_linear_mse = model.linear_regression(RB_training_stats, RB_training_points, RBstats2014, RBpoints2015)
    RB_ridge_model, RB_ridge_preds, RB_ridge_mse = model.ridge_regression(RB_training_stats, RB_training_points, RBstats2014, RBpoints2015)
    RB_lasso_model, RB_lasso_preds, RB_lasso_mse = model.lasso_regression(RB_training_stats, RB_training_points, RBstats2014, RBpoints2015)
    RB_elasticnet_model, RB_elasticnet_preds, RB_elasticnet_mse = model.elasticnet_regression(RB_training_stats, RB_training_points, RBstats2014, RBpoints2015)
    RB_knn_model, RB_knn_preds, RB_knn_mse = model.knn(RB_training_stats, RB_training_points, RBstats2014, RBpoints2015)

    #Wide Recievers
    WR_2012_2013_index = data.get_index_for_position(fantasy2012, fantasy2013, pos='WR')
    WR_2013_2014_index = data.get_index_for_position(fantasy2013, fantasy2014, pos='WR')
    WR_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos='WR')

    WRstats2012, WRpoints2013 = data.prepare_input_data(fantasy2012, fantasy2013, WR_2012_2013_index, 'WR')
    WRstats2013, WRpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014, WR_2013_2014_index, 'WR')
    WRstats2014, WRpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, WR_2014_2015_index, 'WR')

    #WRstats2012 = data.df_norm(WRstats2012)
    #WRstats2013 = data.df_norm(WRstats2013)
    #WRstats2014 = data.df_norm(WRstats2014)

    WR_training_stats = pd.concat([WRstats2012, WRstats2013])
    WR_training_points = pd.concat([WRpoints2013, WRpoints2014])

    WR_linear_model, WR_linear_preds, WR_linear_mse = model.linear_regression(WR_training_stats, WR_training_points, WRstats2014,WRpoints2015)
    WR_ridge_model, WR_ridge_preds, WR_ridge_mse = model.ridge_regression(WR_training_stats, WR_training_points, WRstats2014,WRpoints2015)
    WR_lasso_model, WR_lasso_preds, WR_lasso_mse = model.lasso_regression(WR_training_stats, WR_training_points, WRstats2014, WRpoints2015)
    WR_elasticnet_model, WR_elasticnet_preds, WR_elasticnet_mse = model.elasticnet_regression(WR_training_stats, WR_training_points,WRstats2014, WRpoints2015)
    WR_knn_model, WR_knn_preds, WR_knn_mse = model.knn(WR_training_stats, WR_training_points, WRstats2014, WRpoints2015)

    #Tight Ends
    TE_2012_2013_index = data.get_index_for_position(fantasy2012, fantasy2013, pos='TE')
    TE_2013_2014_index = data.get_index_for_position(fantasy2013, fantasy2014, pos='TE')
    TE_2014_2015_index = data.get_index_for_position(fantasy2014, fantasy2015, pos='TE')

    TEstats2012, TEpoints2013 = data.prepare_input_data(fantasy2012, fantasy2013, TE_2012_2013_index, 'TE')
    TEstats2013, TEpoints2014 = data.prepare_input_data(fantasy2013, fantasy2014, TE_2013_2014_index, 'TE')
    TEstats2014, TEpoints2015 = data.prepare_input_data(fantasy2014, fantasy2015, TE_2014_2015_index, 'TE')

    #TEstats2012 = data.df_norm(TEstats2012)
    #TEstats2013 = data.df_norm(TEstats2013)
    #TEstats2014 = data.df_norm(TEstats2014)

    TE_training_stats = pd.concat([TEstats2012, TEstats2013])
    TE_training_points = pd.concat([TEpoints2013, TEpoints2014])

    TE_linear_model, TE_linear_preds, TE_linear_mse = model.linear_regression(TE_training_stats, TE_training_points, TEstats2014,TEpoints2015)
    TE_ridge_model, TE_ridge_preds, TE_ridge_mse = model.ridge_regression(TE_training_stats, TE_training_points, TEstats2014, TEpoints2015)
    TE_lasso_model, TE_lasso_preds, TE_lasso_mse = model.lasso_regression(TE_training_stats, TE_training_points, TEstats2014, TEpoints2015)
    TE_elasticnet_model, TE_elasticnet_preds, TE_elasticnet_mse = model.elasticnet_regression(TE_training_stats, TE_training_points,TEstats2014, TEpoints2015)
    TE_knn_model, TE_knn_preds, TE_knn_mse = model.knn(TE_training_stats, TE_training_points, TEstats2014, TEpoints2015)