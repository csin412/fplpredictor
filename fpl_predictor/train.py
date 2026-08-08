import joblib
from lightgbm import LGBMClassifier
from features import FEATURE_COLS

BEST_PARAMS_CLF = {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 10, 'min_child_samples': 20}

def train_classifiers(model_df, thresholds=(5, 6), save_dir='models'):
    classifiers = {}
    for t in thresholds:
        model_df[f'hit_{t}plus'] = (model_df['total_points'] >= t).astype(int)
        train_t = model_df[model_df['season'] != '2025-26']
        x_train, y_train = train_t[FEATURE_COLS], train_t[f'hit_{t}plus']

        clf = LGBMClassifier(random_state=42, **BEST_PARAMS_CLF)
        clf.fit(x_train, y_train, categorical_feature=['position'])
        classifiers[t] = clf

        joblib.dump(clf, f'{save_dir}/clf_{t}plus.pkl')
        print(f"Saved {save_dir}/clf_{t}plus.pkl")
    return classifiers