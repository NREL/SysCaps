from sklearn.model_selection import GridSearchCV
import numpy as np
from lightgbm import *
import os
from pathlib import Path
from lightgbm import LGBMRegressor
from shaphypetune import BoostRFE
import pickle
import argparse 

parameters = {
    'metric': ["rmse"], 
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 50, 100],
    'num_leaves': [20, 50, 100],
    'colsample_bytree': [0.2, 0.6, 1.0],
    'reg_lambda': [0.01, 0.1],
    'reg_alpha':  [0.01, 0.1]
}

def normalize_rmse(Y_pred, y_true):
    rmse = np.sqrt(np.mean((Y_pred - y_true) ** 2)) / np.mean(y_true)
    return rmse

def eval(lgbm, X_train, y_train, X_val, y_val):
    rmse_train = normalize_rmse(lgbm.predict(X_train), y_train)
    rmse_val   = normalize_rmse(lgbm.predict(X_val), y_val)

    print("train rmse", rmse_train)
    print("val rmse", rmse_val)

def grid_search(X_train, y_train, X_val, y_val, save_path):
    print("grid search")
    grid_cv = GridSearchCV(estimator=LGBMRegressor(verbose=-1), param_grid=parameters, n_jobs=50, verbose=2, cv=3)
    grid_cv.fit(X_train, y_train)
    lgbm = grid_cv.best_estimator_

    with open(save_path, 'wb') as f:
        pickle.dump(lgbm, f, pickle.HIGHEST_PROTOCOL)
        print("tuned lightgbm saved")

    eval(lgbm, X_train, y_train, X_val, y_val)

    return lgbm

def main(args):
    SYSCAPS_PATH = Path(os.environ.get('SYSCAPS', ''))
    metadata_path = SYSCAPS_PATH / "metadata"

    # set random seed
    np.random.seed(args.random_seed)

    # load data
    X_train = np.load(args.X_train)
    X_val   = np.load(args.X_val)
    y_train = np.load(args.y_train)
    y_val   = np.load(args.y_val)

    print("X_train", X_train.shape)
    print("X_val", X_val.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)

    # dataset meta data
    attributes = open(metadata_path / f'attributes_{args.resstock_comstock}.txt', 'r').read().split('\n')
    attributes = [x.strip('"') for x in attributes]
    attributes = [x for x in attributes if x != ""]
    feature_names = [
        "day_of_year",
        "day_of_week",
        "hour_of_day",
    ] + attributes + [
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "global_horizontal_radiation",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation"
    ]
    feature_names = np.array(feature_names)

    # obtain tuned lightgbm
    if args.grid_search_resume:
        print("loading saved model")
        with open(args.lightgbm_path, 'rb') as f:
            lgbm = pickle.load(f)
    else:
        lgbm = grid_search(X_train, y_train, X_val, y_val, args.lightgbm_path)

    # feature selection
    print("feature selection")
    model_rfe = BoostRFE(lgbm, min_features_to_select=20 if args.resstock_comstock == "resstock" else 10)
    model_rfe.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    with open(f'{args.resstock_comstock}_boost_rfe.pkl', 'wb') as f:
        pickle.dump(model_rfe, f, pickle.HIGHEST_PROTOCOL)

    eval(model_rfe, X_train, y_train, X_val, y_val)
    
    feature_selected = feature_names[model_rfe.ranking_ == 1]
    feature_rejected = feature_names[model_rfe.ranking_ != 1]

    print(f"{len(feature_selected)} features selected")
    print(feature_selected, "\n")
    print(f"{len(feature_rejected)} features rejected")
    print(feature_rejected)
    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--resstock_comstock', type=str, default="resstock")
    parser.add_argument('--X_train', type=str, required=True, help="path to X_train .npy file")
    parser.add_argument('--X_val', type=str, required=True, help="path to X_val .npy file")
    parser.add_argument('--y_train', type=str, required=True, help="path to y_train .npy file")
    parser.add_argument('--y_val', type=str, required=True, help="path to y_val .npy file")
    parser.add_argument('--lightgbm_path', type=str, default="lightgbm.pkl", help="path to save hyper-param tuned lightgbm")
    parser.add_argument('--grid_search_resume', action="store_true", help="whether load saved lightgbm without grid search, default = False")
    args = parser.parse_args()
    main(args)